import argparse
import os
import shutil
import tqdm
from database.database_book import DatabaseBook
from database.database_page import DatabasePage
from omr.dataset import LyricsNormalization
from omr.steps.text.dataset import TextDataset, DatasetParams
from skimage.io import imsave
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--books", default=None, nargs="+")
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--n_pages", default=-1, type=int)
parser.add_argument("--swash_capitals", action="store_true")

args = parser.parse_args()
args.output = os.path.expanduser(args.output)


def extract_text_lines_of_page(page: DatabasePage, output_dir: str):
    params = DatasetParams()
    params.lyrics_normalization.lower_only = False
    params.lyrics_normalization.lyrics_normalization = LyricsNormalization.WORDS
    params.lyrics_normalization.unified_u = False
    params.lyrics_normalization.remove_brackets = True
    params.cut_region = True

    dataset = TextDataset([page.pcgts(create_if_not_existing=False)], params)
    raw_data = dataset.to_text_line_calamari_dataset(train=True).samples()
    for d, l in zip(raw_data, dataset.load()):
        if len(d['text']) == 0:
            continue

        if not args.swash_capitals and len(d['text']) <= 1:
            continue

        filename = "{}_{}_{:03d}".format(page.book.book, page.page, int(d['id']))
        imsave(os.path.join(output_dir, filename + '.bin.png'), d['image'])
        with open(os.path.join(output_dir, filename + '.gt.txt'), 'w') as f:
            f.write(d['text'])


def extract_text_lines_of_book(book: DatabaseBook):
    path = os.path.join(args.output, book.book)
    for page in tqdm.tqdm(book.pages() if args.n_pages < 0 else book.pages()[:args.n_pages], desc="Processing book {}".format(book.book)):
        page_path = os.path.join(path, page.page)
        os.makedirs(page_path, exist_ok=True)
        extract_text_lines_of_page(page, page_path)


for book in args.books:
    extract_text_lines_of_book(DatabaseBook(book))




