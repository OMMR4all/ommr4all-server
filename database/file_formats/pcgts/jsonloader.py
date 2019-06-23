from typing import Union
import json
from uuid import uuid4
from database.file_formats.pcgts.page.musicsymbol import SymbolType, NoteType, ClefType, AccidType, GraphicalConnectionType
from database.file_formats.pcgts.pcgts import PcGts


def update_step(pcgts: dict):
    version = pcgts.get('version', 0)
    if version == PcGts.VERSION:
        return
    elif version == 0:
        from database.file_formats.pcgts.page.block import BlockType
        pcgts['version'] = 1
        page = pcgts.get('page', {})
        blocks = []
        to_block_type = [BlockType.PARAGRAPH.value, BlockType.HEADING.value, BlockType.LYRICS.value,
                         BlockType.DROP_CAPITAL.value,
                         BlockType.FOLIO_NUMBER.value, BlockType.MUSIC.value]
        if 'textRegions' in page:
            for tr in page.get('textRegions', []):
                lines = []
                for tl in tr.get('textLines', []):
                    lines.append({
                        'id': tl.get('id', str(uuid4())),
                        'coords': tl.get('coords', ''),
                        'sentence': {'syllables': tl.get('syllables', '')},
                        'reconstructed': tl.get('reconstructed', False),
                    })

                block = {
                    'id': tr.get('id', str(uuid4())),
                    'type': to_block_type[tr.get('type', 0)],
                    'coords': tr.get('coords', ''),
                    'lines': lines,
                }
                blocks.append(block)

            del page['textRegions']

        if 'musicRegions' in page:
            to_st = [SymbolType.NOTE, SymbolType.CLEF, SymbolType.ACCID]
            to_nt = [NoteType.NORMAL, NoteType.ORISCUS, NoteType.APOSTROPHA, NoteType.LIQUESCENT_FOLLOWING_U, NoteType.LIQUESCENT_FOLLOWING_D]
            for mr in page.get('musicRegions', []):
                lines = []
                for ml in mr.get('musicLines', []):
                    symbols = []
                    for s in ml.get('symbols', []):
                        st = s['symbol']
                        if st == 0:
                            # neume -> to individual notes
                            for i, nc in enumerate(s.get('nc', [])):
                                if i == 0:
                                    nc['graphicalConnection'] = GraphicalConnectionType.NEUME_START.value
                                    nc['id'] = s['id'].replace('neume', 'note')

                                nc['noteType'] = to_nt[nc.get('type', 0)].value
                                nc['type'] = SymbolType.NOTE.value
                                symbols.append(nc)
                        elif st == 1 or st == 2:
                            # clef or accid
                            ns = {
                                'id': s.get('id', str(uuid4())),
                                'type': to_st[st].value,
                                'fixedSorting': s.get('fixedSorting', False),
                                'coord': s.get('coord', ''),
                                'positionInStaff': s.get('positionInStaff', -1),
                            }
                            if st == 1:
                                ns['clefType'] = [ClefType.F, ClefType.C][s.get('type')].value
                            else:
                                ns['accidType'] = [AccidType.NATURAL, AccidType.SHARP, AccidType.FLAT][s.get('type')].value

                            symbols.append(ns)

                    lines.append({
                        'id': ml.get('id', str(uuid4())),
                        'coords': ml.get('coords', ''),
                        'reconstructed': ml.get('reconstructed', False),
                        'staffLines': ml.get('staffLines', []),
                        'symbols': symbols,
                    })

                block = {
                    'id': mr.get('id', str(uuid4())),
                    'type': BlockType.MUSIC.value,
                    'coords': mr.get('coords', ''),
                    'lines': lines
                }
                blocks.append(block)

            del page['musicRegions']

        page['blocks'] = blocks

        # rename neume -> note in connetions
        for c in page.get('annotations', {}).get('connections', []):
            for sc in c.get('syllableConnectors', []):
                if 'neumeID' in sc:
                    sc['noteID'] = sc['neumeID'].replace('neume', 'note')
                    del sc['neumeID']
    else:
        raise ValueError("Unknown version {}".format(version))


def update_pcgts(pcgts: dict, target_version=PcGts.VERSION) -> bool:
    if pcgts.get('version', 0) >= target_version:
        return False

    while pcgts.get('version', 0) < target_version:
        update_step(pcgts)

    return True
