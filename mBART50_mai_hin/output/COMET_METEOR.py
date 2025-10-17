import json
import torch
from comet import download_model, load_from_checkpoint
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# --------- NLTK setup for METEOR ---------
nltk.download('punkt')  # Tokenizer models for word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')

# --------- File reading helper ---------
def read_lines(filepath):
    """Reads a file and returns a list of lines without trailing newline characters."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# --------- Tokenization Function ---------
def tokenize_sentences(sentences):
    """Tokenizes sentences into word lists."""
    return [word_tokenize(sentence) for sentence in sentences]

# --------- METEOR Score Function ---------
def calculate_average_meteor(reference_file, hypothesis_file):
    references = read_lines(reference_file)
    hypotheses = read_lines(hypothesis_file)

    if len(references) != len(hypotheses):
        raise ValueError("Mismatch in number of lines between reference and hypothesis.")

    references_tokenized = tokenize_sentences(references)
    hypotheses_tokenized = tokenize_sentences(hypotheses)

    total_score = 0.0
    for ref_tokens, hyp_tokens in zip(references_tokenized, hypotheses_tokenized):
        total_score += meteor_score([ref_tokens], hyp_tokens)

    return total_score / len(hypotheses)

# --------- COMET Score Function ---------
def calculate_comet_score(source_file, reference_file, hypothesis_file):
    source = read_lines(source_file)
    reference = read_lines(reference_file)
    hypothesis = read_lines(hypothesis_file)

    if not (len(source) == len(reference) == len(hypothesis)):
        raise ValueError("Mismatch in number of lines among source, reference, and hypothesis.")

    print("Loading COMET model (this might take a few seconds)...")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    # Move the model to the appropriate device (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prepare data for COMET
    data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(source, hypothesis, reference)]

    # Predict COMET scores
    comet_output = model.predict(data, batch_size=8)

    return comet_output["system_score"], comet_output["scores"]

# --------- File Paths (Update these if needed) ---------
source_path = "/home/user/Desktop/Maith 1.0/mBART50_mai_hin/output/test.mai_Deva"
reference_path = "/home/user/Desktop/Maith 1.0/mBART50_mai_hin/output/test.hin_Deva"
hypothesis_path = "/home/user/Desktop/Maith 1.0/mBART50_mai_hin/output/generated_predictions.txt"

# --------- Run Evaluation ---------
if __name__ == "__main__":
    print("\n--- METEOR Evaluation ---")
    avg_meteor = calculate_average_meteor(reference_path, hypothesis_path)
    print(f"Average METEOR Score: {avg_meteor:.4f}")

    print("\n--- COMET Evaluation ---")
    comet_score, comet_sentence_scores = calculate_comet_score(source_path, reference_path, hypothesis_path)
    print(f"System-level COMET Score: {comet_score:.4f}")

    # Show a few sentence-level COMET scores
    print("\nFirst 5 Sentence-level COMET Scores:")
    for i, score in enumerate(comet_sentence_scores[:5]):
        print(f"Sentence {i+1}: COMET = {score:.4f}")

