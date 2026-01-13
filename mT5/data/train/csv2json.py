import csv
import json

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        sentences = [row[0] for row in reader]
    return sentences

def merge_to_json(hindi_file, english_file, output_file):
    hindi_sentences = read_csv(hindi_file)
    english_sentences = read_csv(english_file)
    #hindi_sentences=hindi_sentences.iloc[0:53570]
    #english_sentences=english_sentences.iloc[0:53570]

    if len(hindi_sentences) != len(english_sentences):
        raise ValueError("Number of sentences in Sanskrit and Hindi files do not match.")
        print(len(hindi_sentences),len(english_sentences))

    data = []
    for mai, hin in zip(hindi_sentences, english_sentences):
        data.append({"translation": {"hi": hin, "en": eng}})

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    hindi_csv = "/home/kpdubey/NMT/mBART50_chh_eng_exp1/data/train/train_hin.csv" #hindi
    english_csv = "/home/kpdubey/NMT/mBART50_chh_eng_exp1/data/train/train_eng.csv" #english
    output_json = "/home/kpdubey/NMT/mBART50_chh_eng_exp1/data/train/train_hin_eng.json"

    merge_to_json(hindi_csv, english_csv, output_json)

