import sacrebleu
import pyter
from bert_score import score

# Function to read text files line by line
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# Function to calculate chrF2 score
def calculate_chrf2(reference_texts, hypothesis_texts):
    chrf2_score = sacrebleu.corpus_chrf(hypothesis_texts, [reference_texts], beta=2)
    return chrf2_score.score

# Function to calculate TER score safely and log empty rows
def calculate_ter(reference_texts, hypothesis_texts):
    ter_scores = []
    skipped = 0
    empty_indices = []

    for idx, (ref, hyp) in enumerate(zip(reference_texts, hypothesis_texts)):
        if len(ref.strip()) == 0 or len(hyp.strip()) == 0:
            skipped += 1
            empty_indices.append(idx + 1)  # 1-based line number
            continue
        ter_score = pyter.ter(ref.split(), hyp.split())
        ter_scores.append(ter_score)
    
    if empty_indices:
        print(f"\nSkipped {skipped} sentence pairs due to empty reference or hypothesis.")
        print(f"Empty rows (1-based indices): {empty_indices}\n")
    
    if not ter_scores:
        raise ValueError("No valid sentence pairs to compute TER.")

    return sum(ter_scores) / len(ter_scores)

# Function to calculate BERTScore
def calculate_bertscore(reference_texts, hypothesis_texts, lang_code='hi'):
    P, R, F1 = score(hypothesis_texts, reference_texts, lang=lang_code, model_type='xlm-roberta-base')
    return P.mean().item(), R.mean().item(), F1.mean().item()

# File paths
reference_file = "/home/user/Desktop/Maith 1.0/mBART50_mai_hin/output/test.hin_Deva"
hypothesis_file = "/home/user/Desktop/Maith 1.0/mBART50_mai_hin/output/generated_predictions.txt"

# Read files
reference_texts = read_text_file(reference_file)
hypothesis_texts = read_text_file(hypothesis_file)

# Sanity check
if len(reference_texts) != len(hypothesis_texts):
    raise ValueError("The number of sentences in the reference and hypothesis files do not match!")

# Calculate and print chrF2
chrf2 = calculate_chrf2(reference_texts, hypothesis_texts)
print(f"chrF2 Score: {chrf2:.2f}")

# Calculate and print TER (as percentage)
ter = calculate_ter(reference_texts, hypothesis_texts)
print(f"TER Score: {ter * 100:.2f}%")

# Calculate and print BERTScore
P, R, F1 = calculate_bertscore(reference_texts, hypothesis_texts, lang_code='hi')
print(f"BERTScore - Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")

