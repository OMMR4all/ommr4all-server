import argparse
from database import DatabaseBook
from database.file_formats.pcgts import PcGts, SymbolType, NoteComponent, Neume, Clef, Accidental
from enum import IntEnum
import numpy as np
from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", required=True, type=str)

args = parser.parse_args()


class Counts(IntEnum):
    Pages = 0
    StaffLines = 1
    Staves = 2
    Symbols = 3
    NoteComponents = 4
    Clefs = 5
    Accidentals = 6

    Size = 7


global_counts = np.zeros(Counts.Size, dtype=np.uint32)


table = PrettyTable(["Dataset"] + [str(Counts(i))[7:] for i in range(Counts.Size)])
for book_name in args.datasets:
    book = DatabaseBook(book_name)
    counts = np.zeros(Counts.Size, dtype=np.uint32)
    if not book.exists():
        raise ValueError("Dataset '{}' does not exist at '{}'".format(book.book, book.local_path()))

    for page in book.pages():
        pcgts = PcGts.from_file(page.file('pcgts'))
        counts[Counts.Pages] += 1

        for mr in pcgts.page.music_regions:
            for ml in mr.staffs:
                counts[Counts.Staves] += 1
                counts[Counts.StaffLines] += len(ml.staff_lines)

                for s in ml.symbols:
                    if isinstance(s, Neume):
                        n: Neume = s
                        counts[Counts.Symbols] += len(n.notes)
                        counts[Counts.NoteComponents] += len(n.notes)
                    else:
                        counts[Counts.Symbols] += 1
                        if isinstance(s, Clef):
                            counts[Counts.Clefs] += 1
                        elif isinstance(s, Accidental):
                            counts[Counts.Accidentals] += 1
                        else:
                            raise TypeError("Unknown type of {}: {}".format(s, type(s)))

    global_counts += counts
    table.add_row([book_name] + list(counts))

table.add_row(["Total"] + list(global_counts))
print(table.get_string())