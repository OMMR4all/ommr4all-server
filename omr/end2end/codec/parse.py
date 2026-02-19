import re

from database.file_formats.pcgts import MusicSymbolPositionInStaff, SymbolType, GraphicalConnectionType, NoteType, \
    AccidType, MusicSymbol, ClefType, NoteName


def symbol_parse(tuple_str, skip_gapped=False):
    parts = tuple_str.split('|')

    if not parts:
        return None

    main_type_str = parts[0]

    s_type = SymbolType.NOTE
    n_type = None
    c_type = None
    a_type = None
    pos = MusicSymbolPositionInStaff.UNDEFINED
    gc = GraphicalConnectionType.NEUME_START
    octave = 0
    notename = NoteName.UNDEFINED

    if main_type_str == 'clef':
        s_type = SymbolType.CLEF

        clef_val = parts[1]
        if clef_val == 'f':
            c_type = ClefType.F
        elif clef_val == 'c':
            c_type = ClefType.C
        elif clef_val == 'g':
            c_type = ClefType.G

        if len(parts) > 2:
            try:
                pos = MusicSymbolPositionInStaff(int(parts[2]))
            except ValueError:
                print("MusicSymbolPositionInStaff2. Error")
                pass

    elif main_type_str == 'note':
        s_type = SymbolType.NOTE

        try:
            n_type = NoteType(int(parts[1]))
        except (ValueError, IndexError):
            n_type = NoteType.NORMAL

        if len(parts) > 4:
            notename = parts[2]
            octave = parts[3]
            if len(parts) > 4:
                try:
                    gc = GraphicalConnectionType(int(parts[4]))

                    if skip_gapped:
                        if gc == GraphicalConnectionType.GAPED:
                            gc = GraphicalConnectionType.NEUME_START
                except ValueError:
                    print("GraphicalConnectionType. Error")
                    pass


        else:
            if len(parts) > 2:
                try:
                    pos = MusicSymbolPositionInStaff(int(parts[2]))
                except ValueError:
                    print("MusicSymbolPositionInStaff. Error")

                    pass

            if len(parts) > 3:
                try:
                    gc = GraphicalConnectionType(int(parts[3]))

                    if skip_gapped:
                        if gc == GraphicalConnectionType.GAPED:
                            gc = GraphicalConnectionType.NEUME_START
                except ValueError:
                    print("GraphicalConnectionType. Error")
                    pass

    elif main_type_str == 'accid':
        s_type = SymbolType.ACCID

        accid_val = parts[1]

        if accid_val == 'flat':
            a_type = AccidType.FLAT
        elif accid_val == 'sharp':
            a_type = AccidType.SHARP
        elif accid_val == 'natural':
            a_type = AccidType.NATURAL
        else:

            try:
                a_type = AccidType(int(accid_val))
            except:
                pass

        if len(parts) > 4:
            notename = parts[2]
            octave = parts[3]
            if len(parts) > 4:
                try:
                    gc = GraphicalConnectionType(int(parts[4]))

                    if skip_gapped:
                        if gc == GraphicalConnectionType.GAPED:
                            gc = GraphicalConnectionType.NEUME_START
                except ValueError:
                    print("GraphicalConnectionType. Error")
                    pass


        else:
            if len(parts) > 2:
                try:
                    pos = MusicSymbolPositionInStaff(int(parts[2]))
                except ValueError:
                    print("MusicSymbolPositionInStaff. Error")

                    pass

            if len(parts) > 3:
                try:
                    gc = GraphicalConnectionType(int(parts[3]))

                    if skip_gapped:
                        if gc == GraphicalConnectionType.GAPED:
                            gc = GraphicalConnectionType.NEUME_START
                except ValueError:
                    print("GraphicalConnectionType. Error")
                    pass

    ms = MusicSymbol(
        symbol_type=s_type,
        clef_type=c_type,
        note_type=n_type,
        accid_type=a_type,
        position_in_staff=pos,
        graphical_connection=gc,
        note_name=notename,
        octave=octave,
    )
    return ms


def parse_neural_output(output_line, skip_gapped=True):
    content_match = re.search(r"<line>\d+\.\s*(.*?)</line>", output_line)
    if content_match:
        content = content_match.group(1)
    else:

        content = output_line

    text_only = re.sub(r"\[.*?\]", "", content)

    text_only = text_only.replace("*", "")

    text_sequence = text_only.strip()

    music_symbols = []

    symbol_blocks = re.findall(r"\[(.*?)\]", content)

    for block in symbol_blocks:

        raw_tuples = re.findall(r"\(([^)]+)\)", block)
        for tuple_str in raw_tuples:
            ms = symbol_parse(tuple_str, skip_gapped)
            if ms:
                music_symbols.append(ms)

    return text_sequence, music_symbols
