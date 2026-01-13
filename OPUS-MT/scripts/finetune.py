# scripts/finetune_with_bleu_keep_last_two_tensorboard.py

import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
from tqdm import tqdm
import sacrebleu
from torch.utils.tensorboard import SummaryWriter

# Paths to your data files
train_src_file = "../train/train.sanskrit"
train_tgt_file = "../train/train.hindi"
valid_src_file = "../train/valid.sanskrit"
valid_tgt_file = "../train/valid.hindi"

# Load tokenizer and model
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
batch_size = 32
num_epochs = 7
learning_rate = 5e-5
max_length = 256

# Directories
save_dir = "./opus-mt-en-hi-finetuned-sanskrit-hindi"
epoch_ckpt_dir = os.path.join(save_dir, "checkpoints")
os.makedirs(epoch_ckpt_dir, exist_ok=True)

# TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length):
        with open(src_file, "r", encoding="utf-8") as f:
            self.src_texts = f.read().strip().split("\n")
        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_texts = f.read().strip().split("\n")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.src_texts[idx]
        tgt = self.tgt_texts[idx]
        src_enc = self.tokenizer(src, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        tgt_enc = self.tokenizer(tgt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        input_ids = src_enc["input_ids"].squeeze()
        attention_mask = src_enc["attention_mask"].squeeze()
        labels = tgt_enc["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "src_text": src,
            "tgt_text": tgt
        }

# Create datasets and dataloaders
train_dataset = TranslationDataset(train_src_file, train_tgt_file, tokenizer, max_length)
valid_dataset = TranslationDataset(valid_src_file, valid_tgt_file, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Track best BLEU
best_bleu = 0.0
last_two_ckpts = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}")
    writer.add_scalar("Loss/train", avg_train_loss, epoch+1)

    # Validation BLEU
    model.eval()
    generated_texts = []
    references = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validation BLEU"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4
            )
            generated_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_texts.extend(generated_batch)
            references.extend(batch["tgt_text"])

    bleu = sacrebleu.corpus_bleu(generated_texts, [references])
    print(f"Epoch {epoch+1} - Validation BLEU: {bleu.score:.2f}")
    writer.add_scalar("BLEU/valid", bleu.score, epoch+1)

    # Save epoch checkpoint
    epoch_dir = os.path.join(epoch_ckpt_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    print(f"✅ Model checkpoint for epoch {epoch+1} saved to {epoch_dir}")

    # Manage last two checkpoints
    last_two_ckpts.append(epoch_dir)
    if len(last_two_ckpts) > 2:
        # Remove the oldest checkpoint
        oldest_ckpt = last_two_ckpts.pop(0)
        shutil.rmtree(oldest_ckpt)
        print(f"🗑️ Oldest checkpoint {oldest_ckpt} removed to keep only the last two.")

    # Save best model based on BLEU
    if bleu.score > best_bleu:
        best_bleu = bleu.score
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"🎉 Best model updated based on BLEU {best_bleu:.2f} and saved to {save_dir}")

writer.close()
print("✅ Training complete.")

