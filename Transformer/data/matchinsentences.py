def count_and_save_matching_sentences(file1_path, file2_path, output_file):
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        sentences1 = [line.strip() for line in f1]
        sentences2 = [line.strip() for line in f2]

    # Compare up to the minimum length
    min_len = min(len(sentences1), len(sentences2))
    matching_sentences = []

    for i in range(min_len):
        if sentences1[i] == sentences2[i]:
            matching_sentences.append(sentences1[i])

    # Save matching sentences
    with open(output_file, 'w', encoding='utf-8') as out:
        for sentence in matching_sentences:
            out.write(sentence + '\n')

    print(f"Total matching sentences: {len(matching_sentences)} out of {min_len}")
    print(f"Matching sentences saved to: {output_file}")

# Example usage
file1 = 'sys.out'
file2 = 'ref.out'
output = 'matching_sentences.txt'
count_and_save_matching_sentences(file1, file2, output)

