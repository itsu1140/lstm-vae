from collections import deque

import numpy as np
import pretty_midi

PIANO_SIZE = 88
THRESHOLD = 0.05
OFFSET = 24


def get_keyboard() -> list[int]:
    return [0] * PIANO_SIZE


def get_chroma_gram() -> list[int]:
    return [0] * 12


def melo2vec(melody: pretty_midi.Instrument, beats: np.ndarray) -> np.ndarray:
    notes = sorted(melody.notes, key=lambda x: x.start)
    notes = deque(notes)
    if len(notes) == 0:
        return []

    beat_diff = beats[1] - beats[0]
    beat_diff /= 2
    beat_len = len(beats)
    eighth_beat = beats.copy().tolist()
    for i in range(beat_len - 1):
        eighth_beat.append(beats[i] + beat_diff)
    eighth_beat = np.array(sorted(eighth_beat))
    seq = []
    for beat in eighth_beat:
        on_note = get_keyboard()
        while len(notes) > 0 and beat - notes[0].start > THRESHOLD:
            buf = notes.popleft()
            if beat < buf.end:
                buf.start = beat
                notes.appendleft(buf)
        if len(notes) > 0 and (abs(beat - notes[0].start) <= THRESHOLD):
            cur_note = notes.popleft()
            pitch = cur_note.pitch - OFFSET
            if not 0 <= pitch < PIANO_SIZE:
                continue
            on_note[pitch] = 1
            cur_note.start += beat_diff
            if cur_note.start < cur_note.end:
                notes.appendleft(cur_note)
        seq.append(on_note)

    return np.array(seq)


def chord2vec(chord: pretty_midi.Instrument, beats: np.ndarray) -> np.ndarray:
    notes = deque(chord.notes)
    if len(notes) == 0:
        return []

    beat_diff = beats[1] - beats[0]
    beat_diff /= 2
    seq = []
    data = []
    chroma_gram = get_chroma_gram()

    for beat in beats:
        long_note = deque()
        while len(notes) > 0 and beat - notes[0].start > THRESHOLD:
            buf = notes.popleft()
            if beat < buf.end:
                buf.start = beat
                long_note.append(buf)

        while len(long_note):
            notes.appendleft(long_note.popleft())

        while len(notes) > 0 and (abs(beat - notes[0].start) <= THRESHOLD):
            cur_note = notes.popleft()
            pitch = cur_note.pitch - OFFSET
            chroma_gram[pitch % 12] = 1
            if 0 <= pitch <= PIANO_SIZE and len(data) < PIANO_SIZE * 4:
                on_note = get_keyboard()
                on_note[pitch] = 1
                data += on_note

            cur_note.start += beat_diff
            if cur_note.start < cur_note.end:
                long_note.append(cur_note)
        while len(notes) and len(long_note) and notes[0].start < long_note[0].start:
            long_note.append(notes.pop())
        while len(long_note):
            notes.appendleft(long_note.popleft())
        lack_dim = PIANO_SIZE * 4 - len(data)
        data += [0] * lack_dim + chroma_gram
        chroma_gram = get_chroma_gram()
        seq.append(data)
        data = []
    return np.array(seq)
