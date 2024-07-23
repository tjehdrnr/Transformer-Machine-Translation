import sys, fileinput
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

if __name__ == "__main__":

    for line in fileinput.input():
        if line.strip() != "":
            tokens = word_tokenize(line.strip())

            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')