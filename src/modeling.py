from typing import List, Literal, Optional, Union
from torch import (
    nn,
    optim
)
import torch
from transformers import (
    AutoModel,
    LlamaForCausalLM,
    AutoTokenizer, 
    PretrainedConfig,
    get_linear_schedule_with_warmup
)
from pytorch_lightning import LightningModule


class ProteinAdapter(nn.Module):
    def __init__(
        self,
        plm_config: PretrainedConfig,
        lm_config: PretrainedConfig,
        intermediate_size: int = 256
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(plm_config.hidden_size, intermediate_size, False)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(intermediate_size, lm_config.hidden_size, False)
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.linear1(hidden_states))

        return self.linear2(hidden_states)


class ProteinTextLanguageModel(LightningModule):
    def __init__(self,
        protein_encoder: str,
        language_model: str,
        adapter_hidden_size: int = 256,
        max_steps: int = 10000,
        warmup_steps: int = 1000,
        mode: Literal['finetuning', 'pretraining'] = 'pretraining',
        learning_rate: float = 3e-4,
        beta1: float = 0.99,
        beta2: float = 0.98
    ):
        super().__init__()
        
        self.protein_tower: AutoModel = AutoModel.from_pretrained(protein_encoder)
        self.language_model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(language_model)
        
        self.protein_adapter = ProteinAdapter(
            plm_config=self.protein_tower.config,
            lm_config=self.language_model.config,
            intermediate_size=adapter_hidden_size
        )
        
        if mode == 'pretraining':
            self.protein_tower = self.protein_tower.requires_grad_(False)
            self.language_model = self.language_model.requires_grad_(False)
            self.protein_adapter = self.protein_adapter.requires_grad_(True)
        else:
            self.protein_tower = self.protein_tower.requires_grad_(False)
            self.language_model = self.language_model.requires_grad_(True)
            self.protein_adapter = self.protein_adapter.requires_grad_(True)
        
        self.tokenizers = {}
        self.tokenizers['protein'] = AutoTokenizer.from_pretrained(protein_encoder)
        self.tokenizers['text'] = AutoTokenizer.from_pretrained(language_model)
        
        if self.tokenizers['text'].convert_tokens_to_ids('<protein>') == 0:
            raise ValueError('')
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.adapter_hidden_size = adapter_hidden_size
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.save_hyperparameters()


    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.language_model.parameters()) + list(self.protein_adapter.parameters()) + list(self.protein_adapter.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2)
        )
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.max_steps,
            num_warmup_steps=self.warmup_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
    
    def _calculate_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
        shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1):]
        shift_logits = logits[..., :-1, :][shift_attention_mask != 0]
        shift_labels = labels[..., 1:][shift_attention_mask != 0]
                
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )
        
    def _get_inputs_embeds(self,
        input_ids: torch.Tensor,
        protein_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        
        protein_outputs = self.protein_tower(
            input_ids=protein_ids,
            attention_mask=protein_mask
        )
        
        # [bs, protein_hidden_size]
        protein_embeds = protein_outputs.last_hidden_state[:, 0, :] # use <cls> token as protein-level representation 
        
        # [bs, hidden_size]
        protein_embeds = self.protein_adapter(protein_embeds)
        
        # Inject protein representation into text embeddings
        protein_token_id = self.tokenizers['text'].convert_tokens_to_ids('<protein>')
        
        special_protein_mask = (input_ids == protein_token_id).unsqueeze(-1)
        special_protein_mask = special_protein_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        
        inputs_embeds = inputs_embeds.masked_scatter(special_protein_mask, protein_embeds)
        
        return inputs_embeds
        
    def forward(
        self,
        input_ids: torch.Tensor,
        protein_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **lm_kwargs,
    ):
        inputs_embeds = self._get_inputs_embeds(
            input_ids=input_ids,
            protein_ids=protein_ids,
            protein_mask=protein_mask,
            attention_mask=attention_mask
        )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **lm_kwargs,
        )
        
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        protein_input_ids = batch["protein_input_ids"]
        protein_attention_mask = batch["protein_attention_mask"]
        
        labels = batch['input_ids'].clone()
        im_start_id = self.tokenizers['text'].convert_tokens_to_ids('<|im_start|>')
        mask = (labels == im_start_id).cumsum(dim=1) > 0
        labels[~mask] = -100
        
        outputs = self(
            input_ids=input_ids,
            protein_ids=protein_input_ids,
            attention_mask=attention_mask,
            protein_mask=protein_attention_mask
        )
        
        # Calculate loss
        loss = self._calculate_loss(outputs.logits, labels, attention_mask)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        protein_input_ids = batch["protein_input_ids"]
        protein_attention_mask = batch["protein_attention_mask"]
        labels = batch['input_ids'].clone()
        
        im_start_id = self.tokenizers['text'].convert_tokens_to_ids('<|im_start|>')
        mask = (labels == im_start_id).cumsum(dim=1) > 0
        labels[~mask] = -100
        
        outputs = self(
            input_ids=input_ids,
            protein_ids=protein_input_ids,
            attention_mask=attention_mask,
            protein_mask=protein_attention_mask
        )

        # Calculate loss
        loss = self._calculate_loss(outputs.logits, labels, attention_mask)
        
        self.log('valid_loss', loss)
        return loss
    
    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def predict_step(
        self,
        sequences: Union[str, List[str]],
        user_input: str = 'Describe protein name'
    ):
        
        if isinstance(sequences, str):
            sequences = [sequences]
            
        outputs = []
        
        for seq in sequences:
            protein_inputs = self.tokenizers['protein'](seq, return_tensors='pt', padding=True)
            
            prompt = f"USER: <protein> \n {user_input} <|im_start|>"
            prompt = self.tokenizers['text'](prompt, return_tensors='pt')
            
            inputs = {
                'protein_ids': protein_inputs['input_ids'],
                'protein_mask': protein_inputs['attention_mask'],
                'input_ids': prompt['input_ids'],
                'attention_mask': prompt['attention_mask']
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            inputs_embeds = self._get_inputs_embeds(**inputs)
            
            output = self.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])

            output = self.tokenizers['text'].batch_decode(output[0], skip_special_tokens=False)
            output = "".join(output)
            output = output.replace('ASSISTANT:', '').strip()
            
            outputs.append(output)
    
        return outputs
