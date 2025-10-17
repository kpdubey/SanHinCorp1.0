import json

def txt_to_json(hindi_file, english_file, output_json):
    """
    Convert two text files (Hindi and English) into a JSON file.
    Each line in the Hindi file corresponds to the same line in the English file.

    :param hindi_file: Path to the Hindi text file
    :param english_file: Path to the English text file
    :param output_json: Path to the output JSON file
    """
    with open(hindi_file, 'r', encoding='utf-8') as h_file, \
         open(english_file, 'r', encoding='utf-8') as e_file:
        
        # Read lines from both files
        hindi_sentences = h_file.readlines()
        english_sentences = e_file.readlines()
        
        # Check if both files have the same number of lines
        if len(hindi_sentences) != len(english_sentences):
            raise ValueError("The number of lines in the Hindi file does not match the English file.")
        
        # Create a list of dictionaries with Hindi and English pairs
        data = []
        for hindi, english in zip(hindi_sentences, english_sentences):
            data.append({"translation":{"ma": hindi.strip(),"hi": english.strip()}})
        
        # Write the data to a JSON file
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f"JSON file created successfully at {output_json}")

# Example Usage
hindi_file = "/media/kpdubey/43a28c87-1876-4ae5-a360-9029ca34f6cd/nmt/hgftransformer/data/test/test.mai_Deva"      # Path to Hindi text file
english_file = "/media/kpdubey/43a28c87-1876-4ae5-a360-9029ca34f6cd/nmt/hgftransformer/data/test/test.hin_Deva"  # Path to English text file
output_json = "/media/kpdubey/43a28c87-1876-4ae5-a360-9029ca34f6cd/nmt/hgftransformer/data/test/test.json"   # Output JSON file path

txt_to_json(hindi_file, english_file, output_json)
