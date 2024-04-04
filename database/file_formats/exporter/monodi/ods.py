from typing import Union

import typing

header = ['Dokumenten_ID', 'Gattung1', 'Gattung2', 'Fest', 'Feier', 'Textinitium Editionseinheit',
          'Überlieferungszustand', 'Zusatz zu Textinitium', 'Bezugsgesang', 'Nachweis_Editionseinheit',
          'Melodiennummer_Katalog', 'Kontrafaktur: Stammtext', 'Melodie Standard', 'Melodie Quelle',
          'Startseite', 'Start Verso-Spalte', 'Startzeile', 'Startposition', 'Endseite', 'End Verso-Spalte',
          'Endzeile', 'Nachtragsschicht', 'Quellensigle', 'Druckausgabe', 'Referenz auf Spiel', 'Editionsstatus',
          'Editor', 'Schreibzugriff', 'Kommentar', 'Doc-Id\' (intern)', 'Quellen-ID (intern)', '', 'Verlinkung']


class OdsCell:
    def __init__(self, row: int, column: Union[str, int]):
        self.row = row
        self.column = column

    def get_entry(self):
        return ''.join([str(self.column), str(self.row)])


class Entry:
    def __init__(self, cell: OdsCell, value: Union[str, int] = None):
        self.cell = cell
        self.value = value
        pass


class MonodiOdsConfig:
    default_config = [
        Entry(OdsCell(1, 'A'), 'Dokumenten_ID'),
        Entry(OdsCell(1, 'B'), 'Gattung1'),
        Entry(OdsCell(1, 'C'), 'Gattung2'),
        Entry(OdsCell(1, 'D'), 'Fest'),
        Entry(OdsCell(1, 'E'), 'Feier'),
        Entry(OdsCell(1, 'F'), 'Textinitium Editionseinheit'),
        Entry(OdsCell(1, 'G'), 'Überlieferungszustand'),
        Entry(OdsCell(1, 'H'), 'Zusatz zu Textinitium'),
        Entry(OdsCell(1, 'I'), 'Bezugsgesang'),
        Entry(OdsCell(1, 'J'), 'Nachweis_Editionseinheit'),
        Entry(OdsCell(1, 'K'), 'Melodiennummer_Katalog'),
        Entry(OdsCell(1, 'L'), 'Kontrafaktur: Stammtext'),
        Entry(OdsCell(1, 'M'), 'Melodie Standard'),
        Entry(OdsCell(1, 'N'), 'Melodie Quelle'),
        Entry(OdsCell(1, 'O'), 'Startseite'),
        Entry(OdsCell(1, 'P'), 'Start Verso-Spalte'),
        Entry(OdsCell(1, 'Q'), 'Startzeile'),
        Entry(OdsCell(1, 'R'), 'Startposition'),
        Entry(OdsCell(1, 'S'), 'Endseite'),
        Entry(OdsCell(1, 'T'), 'End Verso-Spalte'),
        Entry(OdsCell(1, 'U'), 'Endzeile'),
        Entry(OdsCell(1, 'V'), 'Nachtragsschicht'),
        Entry(OdsCell(1, 'W'), 'Quellensigle'),
        Entry(OdsCell(1, 'X'), 'Druckausgabe'),
        Entry(OdsCell(1, 'Y'), 'Referenz auf Spiel'),
        Entry(OdsCell(1, 'Z'), 'Editionsstatus'),
        Entry(OdsCell(1, 'AA'), 'Editor'),
        Entry(OdsCell(1, 'AB'), 'Schreibzugriff'),
        Entry(OdsCell(1, 'AC'), 'Kommentar'),
        Entry(OdsCell(1, 'AD'), 'Doc-Id\' (intern)'),
        Entry(OdsCell(1, 'AE'), 'Quellen-ID (intern)'),
        Entry(OdsCell(1, 'AF'), ''),
        Entry(OdsCell(1, 'AG'), 'Verlinkung'),
    ]

    def __init__(self):
        self.length = len(MonodiOdsConfig.default_config)
        self.dict: typing.Dict[str, Entry] = {x.value: x for x in MonodiOdsConfig.default_config}
        self.entries = MonodiOdsConfig.default_config


class MonodiXlsxConfig:
    default_config = [
        Entry(OdsCell(0, 0), 'Dokumenten_ID'),
        Entry(OdsCell(0, 1), 'Gattung1'),
        Entry(OdsCell(0, 2), 'Gattung2'),
        Entry(OdsCell(0, 3), 'Fest'),
        Entry(OdsCell(0, 4), 'Feier'),
        Entry(OdsCell(0, 5), 'Textinitium Editionseinheit'),
        Entry(OdsCell(0, 6), 'Überlieferungszustand'),
        Entry(OdsCell(0, 7), 'Zusatz zu Textinitium'),
        Entry(OdsCell(0, 8), 'Bezugsgesang'),
        Entry(OdsCell(0, 9), 'Nachweis_Editionseinheit'),
        Entry(OdsCell(0, 10), 'Melodiennummer_Katalog'),
        Entry(OdsCell(0, 11), 'Kontrafaktur: Stammtext'),
        Entry(OdsCell(0, 12), 'Melodie Standard'),
        Entry(OdsCell(0, 13), 'Melodie Quelle'),
        Entry(OdsCell(0, 14), 'Startseite'),
        Entry(OdsCell(0, 15), 'Start Verso-Spalte'),
        Entry(OdsCell(0, 16), 'Startzeile'),
        Entry(OdsCell(0, 17), 'Startposition'),
        Entry(OdsCell(0, 18), 'Endseite'),
        Entry(OdsCell(0, 19), 'End Verso-Spalte'),
        Entry(OdsCell(0, 20), 'Endzeile'),
        Entry(OdsCell(0, 21), 'Nachtragsschicht'),
        Entry(OdsCell(0, 22), 'Quellensigle'),
        Entry(OdsCell(0, 23), 'Druckausgabe'),
        Entry(OdsCell(0, 24), 'Referenz auf Spiel'),
        Entry(OdsCell(0, 25), 'Editionsstatus'),
        Entry(OdsCell(0, 26), 'Editor'),
        Entry(OdsCell(0, 27), 'Schreibzugriff'),
        Entry(OdsCell(0, 28), 'Kommentar'),
        Entry(OdsCell(0, 29), 'Doc-Id\' (intern)'),
        Entry(OdsCell(0, 30), 'Quellen-ID (intern)'),
        Entry(OdsCell(0, 31), ''),
        Entry(OdsCell(0, 32), 'Verlinkung'),
        Entry(OdsCell(0, 33), 'Manuscript'),
        Entry(OdsCell(0, 34), 'Lyric'),
        Entry(OdsCell(0, 35), 'Skip'),
        Entry(OdsCell(0, 36), 'Skip_Symbol'),
        Entry(OdsCell(0, 37), 'Empty_Symbol'),

    ]

    def __init__(self):
        self.length = len(MonodiOdsConfig.default_config)
        self.dict: typing.Dict[str, Entry] = {x.value: x for x in MonodiXlsxConfig.default_config}
        self.entries = MonodiXlsxConfig.default_config


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from PIL import Image
    import numpy as np
    from database import DatabaseBook, DatabaseFile
    from database.file_formats.pcgts import PageScaleReference

    book = DatabaseBook('Annotation___Square_Notation')
    pages = book.pages()[0]

    pcgts = [DatabaseFile(page, 'pcgts', create_if_not_existing=True).page.pcgts() for page in [pages]]
    file = pages.file('color_norm_x2').local_path()
    orig = Image.open(file)
    orig = np.array(orig)
    lines = pcgts[0].page.all_music_lines()
    page = pcgts[0].page
    for p in lines:
        # page = p.line.operation.page

        def p2i(l):
            return page.page_to_image_scale(l, PageScaleReference.NORMALIZED_X2)


        for s in p.symbols:
            c = p2i(s.coord).round().astype(int)
            t, l = c.y, c.x
            orig[t - 2:t + 2, l - 2:l + 2] = 0

    Image.fromarray(orig).save('symbols.png')
    plt.imshow(orig)
    plt.show()
    pass
