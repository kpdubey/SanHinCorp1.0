from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from tqdm import tqdm
import torch

# Paths
model_dir = "/media/Maith 1.0/nllb_finetuned_maithili_hindi"
dataset_path = "/media/Maith 1.0/data/hf_dataset"
output_file = "/media/Maith 1.0/output/translated_test_output.txt"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test dataset
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]

# Translate and save outputs
translated_sentences = []
for example in tqdm(test_dataset, desc="Translating"):
    source_text = example["source"]
    inputs = tokenizer(source_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_sentences.append(translated_text)

# Save to file
with open(output_file, "w", encoding="utf-8") as f:
    for line in translated_sentences:
        f.write(line.strip() + "\n")

print(f"Translation completed. Output saved to: {output_file}")

