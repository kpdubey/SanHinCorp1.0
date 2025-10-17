from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk

# Load dataset
dataset_path = "/media/Maith 1.0/data/hf_dataset"
datasets = load_from_disk(dataset_path)

# Load Model & Tokenizer
model_name = "/media/Maith 1.0/nllb_finetune_mai_hin_mansyn/model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize function
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source"], 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        examples["target"], 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_datasets = datasets.map(preprocess_function, batched=True)

# Define training arguments using eval_loss as the metric for best model
training_args = TrainingArguments(
    output_dir="/media/Maith 1.0/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="/media/Maith 1.0/data/logs",
    logging_steps=500,
    fp16=True,
    load_best_model_at_end=True,        # Automatically load the best model at the end
    metric_for_best_model="eval_loss",   # Use evaluation loss to select the best model
    greater_is_better=False              # Lower eval_loss is better
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize Trainer (using eval_loss; no compute_metrics needed)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save final model (the best checkpoint will be loaded automatically)
trainer.save_model("/media/Maith 1.0/nllb_finetuned_maithili_hindi")
tokenizer.save_pretrained("/media/Maith 1.0/nllb_finetuned_maithili_hindi")

print("Fine-tuning completed and model saved!")

