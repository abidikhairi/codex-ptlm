import torch
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


class ProtreinTextDataModule(LightningDataModule):
    train_data: Dataset = None
    valid_data: Dataset = None
    test_data: Dataset = None
    
    def __init__(
        self,
        protein_tokenizer: str,
        text_tokenizer: str,
        dataset_id: str,
        protein_max_length: int = 256,
        batch_size: int = 4,
        num_proc: int = 2
    ):
        super().__init__()
        
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_tokenizer)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
        self.dataset_id = dataset_id
        self.protein_max_length = protein_max_length
        self.batch_size = batch_size
        self.num_proc = num_proc
    
    
    def _tokenize_protein(self, examples):
        inputs = self.protein_tokenizer(
            examples['Sequence'],
            return_tensors='pt',
            padding=True,
            max_length=self.protein_max_length,
            truncation=True
        )
        inputs = {f'protein_{k}': v for k, v in inputs.items()}
        
        return inputs
    
    def _tokenize_text(self, example):
        text_inputs = []
        for x, y in zip(example['Instruction'], example['Answer']):
            text_inputs.append(f'{x} <|im_start|> {y} <|im_end|>')
        
        inputs = self.text_tokenizer(
            text_inputs,
            return_tensors='pt',
            padding=True,
            max_length=self.protein_max_length,
            truncation=True
        )
        
        return inputs
    
    def _prepare_dataset(self, dataset: Dataset):
        return dataset.map(self._tokenize_protein, batch_size=self.batch_size, batched=True) \
            .map(self._tokenize_text, batch_size=self.batch_size, batched=True) \
            .remove_columns('Entry')

    def setup(self, stage=None):
        dataset = load_dataset(self.dataset_id)
        
        self.train_data = self._prepare_dataset(dataset['train']).with_format('torch')
        self.valid_data = self._prepare_dataset(dataset['validation']).with_format('torch')
        self.test_data = self._prepare_dataset(dataset['test']).with_format('torch')

        return super().setup(stage)
    
    def _collate_fn(self, rows):
        input_ids = []
        attention_mask = []
        protein_input_ids = []
        protein_attention_mask = []
        
        for row in rows:
            input_ids.append(row['input_ids'])
            attention_mask.append(row['attention_mask'])
            protein_input_ids.append(row['protein_input_ids'])
            protein_attention_mask.append(row['protein_attention_mask'])
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.text_tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        protein_input_ids = torch.nn.utils.rnn.pad_sequence(protein_input_ids, batch_first=True, padding_value=self.protein_tokenizer.pad_token_id)
        protein_attention_mask = torch.nn.utils.rnn.pad_sequence(protein_attention_mask, batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'protein_input_ids': protein_input_ids,
            'protein_attention_mask': protein_attention_mask
        }
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_proc,
            collate_fn=self._collate_fn
        )
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_proc
        )
    
    def test_dataloader(self):
            return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_proc
        )
    