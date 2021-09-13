import enum
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List
import numpy as np

from database.file_formats.pcgts import MusicSymbol, SymbolType, NoteType, GraphicalConnectionType, NoteName, AccidType
from omr.steps.symboldetection.evaluator import SequenceDiffs, Codec


def load_json(path):
    json_data = None

    with open(path, 'r') as fp:
        json_data = json.load(fp)
    return json_data


def populate(path):
    sentence = simple_monodi_data_importer(load_json(path))

    return sentence


class NoteTypeSimple(enum.IntEnum):
    GAPED = 0
    LOOPED = 1
    NeumeStart = 2


@dataclass
class SimpleNote:
    base: str
    octave: int
    type: NoteTypeSimple = NoteTypeSimple.NeumeStart
    note_type: NoteType = NoteType.NORMAL
    symbol_type: SymbolType = SymbolType.NOTE
    accid_type: AccidType = AccidType.NATURAL


def get_melody_sequence_from_monodi_json(json):
    sentence = []
    melody_sequence = []
    for x in json["children"]:
        for y in x["children"]:
            if y["kind"] == "ZeileContainer":
                for z in y["children"]:
                    if z["kind"] == "Syllable":
                        sentence.append(z["text"])
                    elif z["kind"] == "LineChange":
                        pass
                        # sentence.append("\n")
                    elif z["kind"] == "FolioChange":
                        pass
                        # sentence.append("\n")
                    else:
                        pass
                        # print(z["kind"])
                    for note in z["notes"]:
                        for spaced in note["spaced"]:
                            for nonSpaced in spaced["nonSpaced"]:
                                for grouped in nonSpaced["grouped"]:
                                    melody_sequence.append(SimpleNote(grouped["base"], grouped["octave"]))

    return


def getRowContainer(dict, list):
    if "children" in dict:
        for x in dict["children"]:
            if "kind" in x and x["kind"] != "ZeileContainer":
                getRowContainer(x, list=list)
            else:
                list.append(x)
    else:
        pass
        # print(dict)


def simple_monodi_data_importer(json, type=False):
    sentence = []
    row_container = []
    melody_sequence = []

    getRowContainer(json, row_container)
    for x in row_container:
        for z in x["children"]:
            if z["kind"] == "Syllable":
                sentence.append(z["text"])
                for spaced in z["notes"]["spaced"]:
                    neumeStart = True
                    for nonSpaced in spaced["nonSpaced"]:
                        grouped_c = False
                        for grouped in nonSpaced["grouped"]:
                            if neumeStart:
                                melody_sequence.append(
                                    SimpleNote(grouped["base"], grouped["octave"], NoteTypeSimple.NeumeStart,
                                               NoteType.from_string(grouped["noteType"]),
                                               SymbolType.NOTE if grouped["noteType"] == "Normal" or grouped["noteType"] == "Liquescent" else SymbolType.ACCID,
                                               AccidType.from_string(grouped["noteType"])))
                                neumeStart = False
                                grouped_c = True
                            else:
                                if grouped_c:
                                    melody_sequence.append(SimpleNote(grouped["base"], grouped["octave"],
                                                                      NoteTypeSimple.LOOPED,
                                                                      NoteType.NORMAL, #NoteType.from_string(grouped["noteType"]),
                                                                      SymbolType.NOTE if grouped[
                                                                                             "noteType"] == "Normal" or grouped["noteType"] == "Liquescent" else SymbolType.ACCID,
                                                                      AccidType.from_string(grouped["noteType"])))
                                else:
                                    melody_sequence.append(SimpleNote(grouped["base"], grouped["octave"],
                                                                      NoteTypeSimple.GAPED,
                                                                      NoteType.NORMAL, #NoteType.from_string(grouped["noteType"]),
                                                                      SymbolType.NOTE if grouped[
                                                                                             "noteType"] == "Normal"
                                                                                         or grouped["noteType"] == "Liquescent" else SymbolType.ACCID,
                                                                      AccidType.from_string(grouped["noteType"])))
                                    grouped_c = True



            elif z["kind"] == "LineChange":
                pass
            elif z["kind"] == "FolioChange":
                pass
            else:
                pass

    return melody_sequence


class Codec2:
    def __init__(self):
        self.codec = []
        self.type = True

    def get(self, element):
        if element in self.codec:
            return self.codec.index(element)
        else:
            self.codec.append(element)
        return len(self.codec) - 1

    def symbols_to_sequence(self, symbols: List[SimpleNote]):
        sequence = []
        for x in symbols:
            sequence.append(self.get((x.base, x.octave, True if not self.type else False)))
        return sequence

    def compute_sequence_diffs(self, gt, pred) -> SequenceDiffs:
        sm = SequenceMatcher(a=pred, b=gt, autojunk=False, isjunk=False)
        total = max(len(gt), len(pred))
        missing_accids = 0
        missing_notes = 0
        missing_clefs = 0
        wrong_note_connections = 0
        wrong_position_in_staff = 0

        additional_note = 0
        add_wrong_pos_in_staff = 0
        add_wrong_note_con = 0
        additional_clef = 0
        additional_accid = 0

        total_errors = 0
        true_positives = 0
        # print(list(map(self.codec.__getitem__, pred)))
        # print(list(map(self.codec.__getitem__, gt)))
        # print(sm.get_opcodes())
        for opcode, pred_start, pred_end, gt_start, gt_end in sm.get_opcodes():
            if opcode == 'equal':
                true_positives += gt_end - gt_start
            elif opcode == 'insert' or opcode == 'replace' or opcode == 'delete':
                total_errors += pred_end - pred_start + gt_end - gt_start
                for i, s in enumerate(gt[gt_start:gt_end]):
                    entry = self.codec[s]
                    symbol_type = entry[0]
                    # if symbol_type == SymbolType.ACCID:
                    #    missing_accids += 1
                    # elif symbol_type == SymbolType.NOTE:
                    if opcode == 'replace' and pred_end > pred_start + i:
                        # check for wrong connection
                        p = self.codec[pred[pred_start + i]]
                        if p[0] == symbol_type:
                            if p[3] == entry[3]:
                                wrong_position_in_staff += 1
                            else:
                                wrong_note_connections += 1
                        else:
                            missing_notes += 1
                    else:
                        missing_notes += 1
                    # elif symbol_type == SymbolType.CLEF:
                    #    missing_clefs += 1
                    # else:
                    #    raise ValueError("Unknown symbol type {} of entry {}".format(symbol_type, entry))

                for i, s in enumerate(pred[pred_start:pred_end]):
                    entry = self.codec[s]
                    symbol_type = entry[0]
                    # if symbol_type == SymbolType.ACCID:
                    #    additional_accid += 1
                    # elif symbol_type == SymbolType.NOTE:
                    if opcode == 'replace' and gt_end > gt_start + i:
                        # check for wrong connection
                        p = self.codec[gt[gt_start + i]]
                        if p[0] == symbol_type:
                            if p[3] == entry[3]:
                                add_wrong_pos_in_staff += 1
                            else:
                                add_wrong_note_con += 1
                        else:
                            additional_note += 1
                    else:
                        additional_note += 1
                    # elif symbol_type == SymbolType.CLEF:
                    #    additional_clef += 1
                    # else:
                    #    raise ValueError("Unknown symbol type {} of entry {}".format(symbol_type, entry))

            else:
                raise ValueError(opcode)

        return SequenceDiffs(missing_notes, wrong_note_connections, wrong_position_in_staff, missing_clefs,
                             missing_accids,
                             additional_note, add_wrong_note_con, add_wrong_pos_in_staff,
                             additional_clef, additional_accid,
                             total, total_errors)


class MusicSymbolConverter():
    def __init__(self):
        pass

    @staticmethod
    def convert(symbols: List[SimpleNote]):
        m_symbols: List[MusicSymbol] = []
        for x in symbols:
            m_symbols.append(MusicSymbol(x.symbol_type, None, note_type=x.note_type,
                                         graphical_connection=GraphicalConnectionType(x.type.value),
                                         octave=x.octave, note_name=NoteName.from_string(x.base),
                                         accid_type=x.accid_type))
        pass
        return m_symbols


class MonodiSymbolEvaluator:
    def __init__(self):
        self.codec = Codec()

    def evaluate(self, gt_melody: List[SimpleNote], pred_melody: List[SimpleNote]):
        ms_gt = MusicSymbolConverter.convert(gt_melody)
        ms_pred = MusicSymbolConverter.convert(pred_melody)

        gt = self.codec.symbols_to_label_sequence_melody(ms_gt, True)
        pred = self.codec.symbols_to_label_sequence_melody(ms_pred, True)
        print(gt)
        print(pred)
        diffs = np.asarray(self.codec.compute_sequence_diffs(gt, pred))
        return diffs


def writetoods(gt, pred1o, pred2o, pred1m, pred2m, name1, name2):
    import ezodf
    spreadsheet = ezodf.newdoc(doctype="ods", filename="spreadsheetdip.ods")
    sheets = spreadsheet.sheets
    # append a new sheet
    sheets += ezodf.Table('Sheet1', size=(100, 100))
    sheet = sheets[0]
    starty = 2
    startx = 1
    header = [
        "missing_notes",
        "wrong_note_connections",
        "wrong_octave/basis",
        "missing_clefs",
        "missing_accids",
        "additional_note",
        "add_wrong_note_con",
        "add_octabe/basis",
        "additional_clef",
        "additional_accid",
        "total",
        "total_errors"]

    for ind, x in enumerate(header):
        sheet[starty - 1, startx + 1 + 2 + ind].set_value(x)

    #sheet[starty - 1, startx + 1 + 2 + len(header)].formula =""

    for gts, ao, bo, am, bm in zip(gt, pred1o, pred2o, pred1m, pred2m):
        id = gts.split("/")[-1]
        r1 = test(gts, ao)
        r2 = test(gts, bo)
        r1m = test(gts, am)
        r2m = test(gts, bm)
        sheet[starty, startx].set_value(id)
        sheet[starty, startx + 1].set_value(name1)
        sheet[starty + 2, startx + 1].set_value(name2)
        for ind, x in enumerate(zip(r1, r2, r1m, r2m)):
            c, d, e, f = x
            sheet[starty, startx + 1 + 1].set_value("o")
            sheet[starty, startx + 1 + 2 + ind].set_value(int(c))

            sheet[starty + 1, startx + 1 + 1].set_value("m")
            sheet[starty + 1, startx + 1 + 2 + ind].set_value(int(e))

            sheet[starty + 2, startx + 1 + 1].set_value("o")
            sheet[starty + 2, startx + 1 + 2 + ind].set_value(int(d))

            sheet[starty + 3, startx + 1 + 1].set_value("m")
            sheet[starty + 3, startx + 1 + 2 + ind].set_value(int(f))

        starty += 4

    spreadsheet.save()


def test(doc1, doc2):
    melody = populate(doc1)
    melody2 = populate(doc2)
    print(melody)
    print(melody2)
    ev = MonodiSymbolEvaluator()
    diff = ev.evaluate(melody, melody2)
    return diff
    # print(melody)
    # print(melody2)
    # for ind, m in enumerate(zip(melody, melody2)):
    #    m1, m2 = m
    #    print(ind + 1)
    #    print(m1)
    #    print(m2)
    #    print("-----")


if __name__ == "__main__":
    '''
    writetoods(["/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00045v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00046r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00046v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00047r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00053v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00054r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00054v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00055r.json", ],
               ["/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/45v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/46r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/46v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/47r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/53v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/54r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/54v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/55r.json"],
               ["/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/45v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/46r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/46v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/47r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/53v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/54r.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/54v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/ommr4all/55r.json"],
               ["/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_45v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_46.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_46v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_47.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_53v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_54.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_54v.json",
                "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/monodi/AB_55.json", ],
               [
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.45v_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.46_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.46v_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.47_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.53v_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.54_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.54v_Editorenordner.json",
                   "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Jasmin/monodi/JHS_f.55_Editorenordner.json"],

               "anna", "jasmin")
    '''
    test("/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/gt/00046r.json",
         "/home/alexanderh/Downloads/OMMR4allEvaluationDatenAB/Evaluation/Buck/ommr4all/46r.json")
    pass
