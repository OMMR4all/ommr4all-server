from typing import List
import database.file_formats.pcgts as ns_pcgts

from midiutil.MidiFile import MIDIFile

from database.file_formats.book.document import Document
from database.file_formats.pcgts import NoteName


class Pitch:
    def __init__(self, pname: NoteName, octave):
        self.pname = pname
        self.octave = octave

    def __hash__(self):
        return hash((self.pname, self.octave))

    def __eq__(self, other):
        return (self.pname, self.octave) == (other.pname, other.octave)


pitch_midi_table = {
    Pitch(NoteName.A, 0): 21,
    Pitch(NoteName.B, 0): 23,
    Pitch(NoteName.C, 1): 24,
    Pitch(NoteName.D, 1): 26,
    Pitch(NoteName.E, 1): 28,
    Pitch(NoteName.F, 1): 29,
    Pitch(NoteName.G, 1): 31,

    Pitch(NoteName.A, 1): 33,
    Pitch(NoteName.B, 1): 35,
    Pitch(NoteName.C, 2): 36,
    Pitch(NoteName.D, 2): 38,
    Pitch(NoteName.E, 2): 40,
    Pitch(NoteName.F, 2): 41,
    Pitch(NoteName.G, 2): 43,

    Pitch(NoteName.A, 2): 45,
    Pitch(NoteName.B, 2): 47,
    Pitch(NoteName.C, 3): 48,
    Pitch(NoteName.D, 3): 50,
    Pitch(NoteName.E, 3): 52,
    Pitch(NoteName.F, 3): 53,
    Pitch(NoteName.G, 3): 55,

    Pitch(NoteName.A, 3): 57,
    Pitch(NoteName.B, 3): 59,
    Pitch(NoteName.C, 4): 60,
    Pitch(NoteName.D, 4): 62,
    Pitch(NoteName.E, 4): 64,
    Pitch(NoteName.F, 4): 65,
    Pitch(NoteName.G, 4): 67,

    Pitch(NoteName.A, 4): 69,
    Pitch(NoteName.B, 4): 71,
    Pitch(NoteName.C, 5): 72,
    Pitch(NoteName.D, 5): 74,
    Pitch(NoteName.E, 5): 76,
    Pitch(NoteName.F, 5): 77,
    Pitch(NoteName.G, 5): 79,

    Pitch(NoteName.A, 5): 81,
    Pitch(NoteName.B, 5): 83,
    Pitch(NoteName.C, 6): 84,
    Pitch(NoteName.D, 6): 86,
    Pitch(NoteName.E, 6): 88,
    Pitch(NoteName.F, 6): 89,
    Pitch(NoteName.G, 6): 91,

    Pitch(NoteName.A, 6): 93,
    Pitch(NoteName.B, 6): 95,
    Pitch(NoteName.C, 7): 96,
    Pitch(NoteName.D, 7): 98,
    Pitch(NoteName.E, 7): 100,
    Pitch(NoteName.F, 7): 101,
    Pitch(NoteName.G, 7): 103,

    Pitch(NoteName.A, 7): 105,
    Pitch(NoteName.B, 7): 107,
    Pitch(NoteName.C, 8): 108,

}


class SimpleMidiExporter:
    def __init__(self, pcgts: List[ns_pcgts.PcGts]):
        self.pcgts = pcgts
        pass

    def generate_midi(self, output_path: str, track_name: str = 'Sample'):
        mf = MIDIFile(1)  # only 1 track
        track = 0  # the only track
        time_step = 0  # start at the beginning
        # create your MIDI object
        # add some notes
        channel = 0
        volume = 100
        mf.addTrackName(track, time_step, track_name)
        mf.addTempo(track, time_step, 120)
        for pcgts in self.pcgts:
            page = pcgts.page
            music_blocks = page.music_blocks()
            for mb in music_blocks:
                for line in mb.lines:
                    symbols = line.symbols
                    for symbol in symbols:
                        if symbol.symbol_type == symbol.symbol_type.NOTE:
                            duration = 0.5  # 1 1 beat long. Calculate duration based on position in image?
                            mf.addNote(track, channel, pitch_midi_table[Pitch(symbol.note_name.value, symbol.octave)],
                                       time_step, duration, volume)
                            time_step = time_step + duration
        # write it to disk
        with open(output_path, 'wb') as outf:
            mf.writeFile(outf)

    def generate_note_sequence(self, document: Document = None):
        notes = []
        total_duration = 0.0
        document_started = False

        def add_note(lines):
            nonlocal total_duration
            nonlocal notes
            for line in lines:
                symbols = line.symbols
                for symbol in symbols:
                    if symbol.symbol_type == symbol.symbol_type.NOTE:
                        duration = 0.5  # 1 1 beat long. Calculate duration based on position in image?
                        notes.append(
                            {"pitch": pitch_midi_table[Pitch(symbol.note_name.value, symbol.octave)],
                             'startTime': total_duration,
                             'endTime': total_duration + duration})
                        total_duration += duration

        for pcgts in self.pcgts:
            page = pcgts.page
            music_blocks = page.music_blocks()
            for mb in music_blocks:
                connections = [c for c in pcgts.page.annotations.connections if c.music_region == mb]
                if len(connections) > 0:
                    connection = connections[0]
                    if document is not None:
                        line_id_start = document.start.line_id
                        line_id_end = document.end.line_id

                        line_ids = [line.id for line in connection.text_region.lines]
                        if page.p_id == document.end.page_id:
                            if line_id_end in line_ids:
                                break
                        if page.p_id == document.start.page_id or document_started:
                            if line_id_start in line_ids:
                                add_note(mb.lines)
                                document_started = True
                            else:
                                add_note(mb.lines)

                else:
                    add_note(mb.lines)
            else:
                continue
            break
        return {'notes': notes, 'totalTime': total_duration}


if __name__ == "__main__":
    from database import DatabaseBook

    b = DatabaseBook('Pa_14819')
    pcgts = [p.pcgts() for p in b.pages()][0]
    sme = SimpleMidiExporter([pcgts])
    sme.generate_midi("/tmp/test.mid")
