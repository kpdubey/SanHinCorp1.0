import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

sentence = "Hello world! How are you today?"
tokens = word_tokenize(sentence)
print(tokens)

