from typing import Union

import typing



header = ['Dokumenten_ID', 'Gattung1', 'Gattung2', 'Fest', 'Feier', 'Textinitium Editionseinheit',
          'Überlieferungszustand', 'Zusatz zu Textinitium', 'Bezugsgesang', 'Nachweis_Editionseinheit',
          'Melodiennummer_Katalog', 'Kontrafaktur: Stammtext', 'Melodie Standard', 'Melodie Quelle',
          'Startseite', 'Start Verso-Spalte', 'Startzeile', 'Startposition', 'Endseite', 'End Verso-Spalte',
          'Endzeile', 'Nachtragsschicht', 'Quellensigle', 'Druckausgabe', 'Referenz auf Spiel', 'Editionsstatus',
          'Editor', 'Schreibzugriff', 'Kommentar', 'Doc-Id\' (intern)', 'Quellen-ID (intern)', '', 'Verlinkung']


class OdsCell:
    def __init__(self, row: int, column: str):
        self.row = row
        self.column = column

    def get_entry(self):
        return ''.join([self.column, str(self.row)])


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
