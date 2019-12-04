from argparse import ArgumentParser
from typing import List, NamedTuple, Set
import numpy as np
import matplotlib.pyplot as plt

from attr import dataclass
from tqdm import tqdm

from database import DatabaseBook
from database.file_formats import PcGts
from database.file_formats.pcgts import MusicSymbol, SymbolType, GraphicalConnectionType, MusicSymbolPositionInStaff
from omr.dataset.datastructs import CalamariCodec, CalamariSequence


parser = ArgumentParser()
parser.add_argument("--books", default=None, nargs="+")
parser.add_argument("--output", required=True, type=str)

args = parser.parse_args()

codec = CalamariCodec()


class Neume(NamedTuple):
    nc: List[MusicSymbol]


@dataclass
class Stats:
    clefs: List[MusicSymbol]
    accids: List[MusicSymbol]
    neumes: List[Neume]
    n_neumes: int = 0
    n_clefs: int = 0
    n_ncs: int = 0
    n_accids: int = 0


stats = Stats([], [], [])


def convert_symbols_to_neumes(symbols: List[MusicSymbol]):
    last_neume = None
    out = []
    for s in symbols:
        if s.symbol_type != SymbolType.NOTE:
            last_neume = None

        if s.symbol_type == SymbolType.CLEF:
            out.append(s)
        elif s.symbol_type == SymbolType.ACCID:
            out.append(s)
        elif s.symbol_type == SymbolType.NOTE:
            if not last_neume or s.graphical_connection == GraphicalConnectionType.NEUME_START:
                last_neume = Neume([s])
                out.append(last_neume)
            else:
                last_neume.nc.append(s)

    return out


def extract_from_pcgts(pcgts: PcGts):
    for ml in pcgts.page.all_music_lines():
        for s in convert_symbols_to_neumes(ml.symbols):
            if isinstance(s, MusicSymbol):
                if s.symbol_type == SymbolType.CLEF:
                    stats.n_clefs += 1
                    stats.clefs.append(s)
                else:
                    stats.n_accids += 1
                    stats.accids.append(s)
            elif isinstance(s, Neume):
                stats.neumes.append(s)
                stats.n_neumes += 1
                stats.n_ncs += len(s.nc)


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


all_symbols = {CalamariSequence(codec, [s]).calamari_str for s in stats.clefs + stats.accids}

all_derived_neumes = set()
unique_neumes = set()
unique_normalized_neumes = set()
for neume in stats.neumes:
    unique_neumes.add(CalamariSequence(codec, neume.nc).calamari_str)

    # compute derived neumes (take care of bounds)
    poss = [nc.position_in_staff for nc in neume.nc]
    low, high = min(poss), max(poss)
    dlow = MusicSymbolPositionInStaff.SPACE_0
    dhigh = MusicSymbolPositionInStaff.SPACE_7
    min_add = dlow - low
    max_add = dhigh - high
    for nc in neume.nc:
        nc.position_in_staff = MusicSymbolPositionInStaff(min_add + nc.position_in_staff)
        assert(nc.position_in_staff >= dlow)
        assert(nc.position_in_staff <= dhigh)

    unique_normalized_neumes.add(CalamariSequence(codec, neume.nc).calamari_str)

    all_derived_neumes.add(CalamariSequence(codec, neume.nc).calamari_str)
    for d in range(max_add - min_add):
        for nc in neume.nc:
            nc.position_in_staff = MusicSymbolPositionInStaff(nc.position_in_staff + 1)
            assert (nc.position_in_staff >= dlow)
            assert (nc.position_in_staff <= dhigh)
        all_derived_neumes.add(CalamariSequence(codec, neume.nc).calamari_str)


print(stats)
print("Codec size: ", len(codec.codec))
print("Total number of neumes: ", stats.n_neumes)
print("Number of unique neumes: ", len(unique_neumes))
print("Number of unique normalized neumes: ", len(unique_normalized_neumes))
print("Number of unique derived neumes: ", len(all_derived_neumes))
print("Longest ncs per neume", max(len(n) for n in unique_neumes))

neume_lengths = np.zeros(max(len(n) for n in unique_neumes))
for n in stats.neumes:
    neume_lengths[len(n.nc) - 1] += 1

plt.bar(range(1, len(neume_lengths) + 1), neume_lengths)
# plt.show()

with open(args.output + '_unique.txt', 'w') as f:
    f.write("\n".join(sorted(all_symbols.union(unique_neumes))))

with open(args.output + '_all.txt', 'w') as f:
    f.write("\n".join(sorted(all_symbols.union(all_derived_neumes))))

with open(args.output + '_codec.json', 'w') as f:
    import json
    json.dump(codec.to_dict(), f, indent=2)
