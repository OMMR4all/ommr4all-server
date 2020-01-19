from argparse import ArgumentParser
from tqdm import tqdm
import os

import django

from database.database_book_meta import DatabaseBookMeta
from database.file_formats.pcgts import Coords, Line, Block

os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
django.setup()
from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts.page import BlockType, Sentence
from omr.steps.algorithmpreditorparams import AlgorithmPredictorSettings
from omr.steps.step import Step, AlgorithmMeta, AlgorithmTypes


parser = ArgumentParser()
parser.add_argument("--book", required=True, type=str)
parser.add_argument("--output_book", required=True, type=str)
parser.add_argument("--type", type=lambda t: AlgorithmTypes[t], required=True, choices=list(AlgorithmTypes))

args = parser.parse_args()

book = DatabaseBook(args.book)
out_book = DatabaseBook(args.output_book)
out_book.create(DatabaseBookMeta(out_book.book, out_book.book))


meta = Step.meta(args.type)
settings = AlgorithmPredictorSettings(meta.best_model_for_book(book))
predictor = meta.create_predictor(settings)

if args.type == AlgorithmTypes.LAYOUT_SIMPLE_LYRICS:
    for prediction in tqdm(predictor.predict([p.copy_to(out_book) for p in book.pages()]), total=len(book.pages())):
        pcgts: PcGts = prediction.pcgts

        # simply override layout for staff lines
        for line in prediction.blocks[BlockType.MUSIC]:
            l = pcgts.page.line_by_id(line.id)
            l.coords = line.coords

        # find best matches for lyric regions
        all_text_lines = sum([b.lines for b in pcgts.page.blocks_of_type(BlockType.LYRICS)], [])
        new_text_blocks = []
        for id_coords in prediction.blocks[BlockType.LYRICS]:
            coords: Coords = id_coords.coords
            aabb = coords.aabb()
            text_lines = []
            for at in all_text_lines[:]:
                if at.aabb.intersects(aabb):
                    all_text_lines.remove(at)
                    text_lines.append(at)

            text_lines.sort(key=lambda l: l.aabb.left())
            new_line = Line(coords=coords, sentence=Sentence.from_string(" ".join([tl.sentence.text(with_drop_capital=True) for tl in text_lines])))
            new_block = Block(BlockType.LYRICS, lines=[new_line])
            # maybe also keep annotations by moving them to the new line
            new_text_blocks.append(new_block)

        pcgts.page.clear_text_blocks()
        pcgts.page.blocks.extend(new_text_blocks)

        pcgts.to_file(pcgts.dataset_page().file('pcgts').local_path())
