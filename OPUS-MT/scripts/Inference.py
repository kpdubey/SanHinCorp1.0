#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: infer_translate_with_bleu.py
Description: Translate test data with fine-tuned model and calculate BLEU score.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sacrebleu import corpus_bleu

# File paths
test_src_file = "../test/test.sanskrit"
test_ref_file = "../test/test.hindi"
output_file = "../test/test_translations.hindi"

# Load the fine-tuned model and tokenizer
model_dir = "opus-mt-en-hi-finetuned-sanhin_mydata"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Translation hyperparameters
batch_size = 16
max_length = 128
beam_size = 4

# Load test source data
with open(test_src_file, "r", encoding="utf-8") as f:
    test_src_texts = f.read().strip().split("\n")

# Load reference translations
with open(test_ref_file, "r", encoding="utf-8") as f:
    ref_texts = f.read().strip().split("\n")

assert len(test_src_texts) == len(ref_texts), "Mismatch in test source and reference lines."

# Translate in batches
translations = []
for i in tqdm(range(0, len(test_src_texts), batch_size), desc="Translating"):
    batch_texts = test_src_texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=beam_size,
            early_stopping=True
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translations.extend(decoded)

# Save translations
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(translations))

print(f"✅ Translations saved to: {output_file}")

# Calculate BLEU score
bleu = corpus_bleu(translations, [ref_texts])
print(f"🎯 BLEU score: {bleu.score:.2f}")

