from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define model name
model_name = "facebook/nllb-200-distilled-600M"

# Define local directory to save the model
save_directory = "/media/Maith 1.0/nllb-200_model"  # Change to your desired path

# Load and save model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model and tokenizer locally
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and Tokenizer saved at: {save_directory}")

