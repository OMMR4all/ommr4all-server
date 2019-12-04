from argparse import ArgumentParser
from tqdm import tqdm

from database import DatabaseBook
from database.file_formats import PcGts


parser = ArgumentParser()
parser.add_argument("--books", default=None, nargs="+")
parser.add_argument("--output", required=True, type=str)

args = parser.parse_args()

dictionary = {}

def normalize(s: str):
    for c in ';.[]()<>':
        s = s.replace(c, '')
    return s

def extract_from_pcgts(pcgts: PcGts):
    for l in pcgts.page.all_text_lines():
        hyphenated = l.sentence.text()
        for word in hyphenated.split():
            word = normalize(word)
            dictionary[word.replace("-", "")] = word

def extract_from_book(book: DatabaseBook):
    for page in tqdm(book.pages(), desc="Processing {}".format(book.book)):
        extract_from_pcgts(page.pcgts())


if args.books is None:
    books = DatabaseBook.list_available()
else:
    books = [DatabaseBook(b) for b in args.books]

print("Processing {} books".format(len(books)))

for book in books:
    print("Processing book {}".format(book.book))
    extract_from_book(book)

print("Extracted {} words".format(len(dictionary)))

with open(args.output, 'w') as f:
    for word, hyphen in dictionary.items():
        f.write("{:20s} {}\n".format(word, hyphen))
