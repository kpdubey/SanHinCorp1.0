import csv
import json

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        sentences = [row[0] for row in reader]
    return sentences

def merge_to_json(maithili_file, hindi_file, output_file):
    maithili_sentences = read_csv(maithili_file)
    hindi_sentences = read_csv(hindi_file)
    #maithili_sentences=maithili_sentences.iloc[0:53570]
    #hindi_sentences=hindi_sentences.iloc[0:53570]

    if len(maithili_sentences) != len(hindi_sentences):
        raise ValueError("Number of sentences in Sanskrit and Hindi files do not match.")
        print(len(maithili_sentences),len(hindi_sentences))

    data = []
    for mai, hin in zip(maithili_sentences, hindi_sentences):
        data.append({"translation": {"ma": mai, "hi": hin}})

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    maithili_csv = "/home/phd2201101001/hgftransformer/data_2/test/test_mai.csv" #maithili
    hindi_csv = "/home/phd2201101001/hgftransformer/data_2/test/test_hin.csv" #hindi
    output_json = "/home/phd2201101001/hgftransformer/data_2/test/test_mai_hin.json"

    merge_to_json(maithili_csv, hindi_csv, output_json)

