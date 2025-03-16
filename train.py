from src.data import ProtreinTextDataModule
from src.modeling import ProteinTextLanguageModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger 


def main():
    dataset_id = "khairi/ptlm-tiny-dataset"
    protein_encoder_id = "khairi/Esm2-8M"
    language_model_id = "khairi/SmolLM-135M"
    num_proc = 4
    batch_size = 8
    protein_max_length = 512
    adapter_hidden_size = 256
    max_steps = 10000
    warmup_steps = 2000
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0.1
    mode ='pretraining'
    log_every_n_steps = 50
    gradient_clip_val = 0.1
    
    
    data_module = ProtreinTextDataModule(
        protein_tokenizer=protein_encoder_id,
        text_tokenizer=language_model_id,
        dataset_id=dataset_id,
        num_proc=num_proc,
        batch_size=batch_size,
        protein_max_length=protein_max_length
    )

    model = ProteinTextLanguageModel(
        protein_encoder=protein_encoder_id,
        language_model=language_model_id,
        adapter_hidden_size=adapter_hidden_size,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        mode=mode,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay
    )
    
    csv_logger = CSVLogger(save_dir="experim", name="logs", flush_logs_every_n_steps=log_every_n_steps)
    # wandb_logger = WandbLogger(name="codex-ptlm", save_dir="experim/wandb", project="codex")
    model_checkpoint = ModelCheckpoint(dirpath='experim/ckpt', monitor='valid_loss', save_top_k=1, save_last=True, mode='min')

    model_summary = ModelSummary()
    progress_bar = RichProgressBar()
    
    trainer = Trainer(
        max_steps=max_steps,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=log_every_n_steps,
        logger=[csv_logger],# wandb_logger],
        callbacks=[model_checkpoint, model_summary, progress_bar],
        gradient_clip_val=gradient_clip_val
    )
    
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
