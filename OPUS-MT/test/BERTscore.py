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

# Final robust function to calculate TER score
def calculate_ter(reference_texts, hypothesis_texts):
    ter_scores = []
    skipped = 0
    for idx, (ref, hyp) in enumerate(zip(reference_texts, hypothesis_texts)):
        ref_tokens = ref.strip().split()
        hyp_tokens = hyp.strip().split()
        # Skip if either reference or hypothesis is empty
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            print(f"Skipped pair at line {idx+1}: Ref='{ref}', Hyp='{hyp}'")
            skipped += 1
            continue
        ter_score = pyter.ter(ref_tokens, hyp_tokens)
        ter_scores.append(ter_score)
    if len(ter_scores) == 0:
        raise ValueError("All reference/hypothesis pairs are empty or have no tokens; cannot compute TER.")
    print(f"Skipped {skipped} pairs with zero tokens out of {len(reference_texts)} sentences.")
    return sum(ter_scores) / len(ter_scores)

# Function to calculate BERTScore
def calculate_bertscore(reference_texts, hypothesis_texts, lang_code='hi'):
    P, R, F1 = score(hypothesis_texts, reference_texts, lang=lang_code, model_type='xlm-roberta-base')
    return P.mean().item(), R.mean().item(), F1.mean().item()

# File paths
reference_file = '/SanHinCorp1.0/OPUS-MT/test/test.hindi'  # Replace with your reference file path
hypothesis_file = '/SanHinCorp1.0/OPUS-MT/test/test_translations.hindi'  # Replace with your hypothesis file path

# Read files
reference_texts = read_text_file(reference_file)
hypothesis_texts = read_text_file(hypothesis_file)

# Sanity check
if len(reference_texts) != len(hypothesis_texts):
    raise ValueError("The number of sentences in the reference and hypothesis files do not match!")

# Print out empty or whitespace-only references for debugging
for i, ref in enumerate(reference_texts):
    if len(ref.strip().split()) == 0:
        print(f"Empty/whitespace-only reference at line {i+1}: Hypothesis is '{hypothesis_texts[i]}'")

# Calculate and print chrF2
chrf2 = calculate_chrf2(reference_texts, hypothesis_texts)
print(f"chrF2 Score: {chrf2:.2f}")

# Calculate and print TER (as percentage)
ter = calculate_ter(reference_texts, hypothesis_texts)
print(f"TER Score: {ter * 100:.2f}%")

# Calculate and print BERTScore
P, R, F1 = calculate_bertscore(reference_texts, hypothesis_texts, lang_code='hi')  # Use 'hi' for Hindi
print(f"BERTScore - Precision: {P:.4f}, Recall: {R:.4f}, F1: {F1:.4f}")

