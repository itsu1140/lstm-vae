from pathlib import Path

import numpy as np
import pretty_midi
import torch
import yaml
from lstmvae import LSTMVAE
from main import Params
from src.data.midi2vec import melo2vec


def get_input_tensor(input_ndarray: np.ndarray, params: Params) -> torch.Tensor:
    seq_len = params.seq_len
    lack_seq = (seq_len - input_ndarray.shape[0] % seq_len) % seq_len
    lack = np.zeros((lack_seq, params.piano_size))
    data = np.concatenate([input_ndarray, lack])
    chord = np.zeros((data.shape[0], 12))
    data = np.block([data, chord]).reshape((-1, seq_len, params.input_size))
    return torch.tensor(data, dtype=torch.float32)


def make_note(
    velocity: int,
    pitch: int,
    start: float,
    unittime: int,
) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=start * unittime,
        end=(start + 1) * unittime,
    )


def connect_same_chord(chord, unittime: int):
    class run_length:
        def __init__(self, chroma, start):
            self.chroma = chroma
            self.start = start
            self.end = start + unittime

        def set_end(self, end):
            self.end = end

    recon_chord = pretty_midi.Instrument(0, is_drum=False, name="chord")
    chroma = [run_length(chord[0], 0)]
    for i, ch in enumerate(chord):
        if i == 0:
            continue
        if ch == chord[i - 1]:
            chroma[-1].set_end((i + 1) * unittime)
        else:
            chroma.append(run_length(ch, i * unittime))

    for c in chroma:
        for j, key in enumerate(c.chroma):
            if not key:
                continue
            recon_chord.notes.append(
                pretty_midi.Note(
                    velocity=70,
                    pitch=j + 48,
                    start=c.start,
                    end=c.end,
                ),
            )
    return recon_chord


def instrument_copy(instrument: pretty_midi.Instrument) -> pretty_midi.Instrument:
    new_instrument = pretty_midi.Instrument(0, "melody")
    for note in instrument.notes:
        new_instrument.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            ),
        )
    return new_instrument


def pred(model_dir: Path, data_path: Path, params: Params) -> None:
    output_dir: Path = Path("outputs/pred") / f"from_{model_dir.name}"
    output_dir.mkdir(exist_ok=False)

    input_mid: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI(data_path)
    melody: pretty_midi.Instrument = instrument_copy(input_mid.instruments[0])
    input_ndarray: np.ndarray = melo2vec(
        input_mid.instruments[0],
        input_mid.get_beats(),
    )
    chord_num: int = 4

    # prepare model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Path.open(model_dir / "params.yaml", "r") as f:
        params: Params = yaml.safe_load(f)
    model: LSTMVAE = LSTMVAE(params=params, device=device).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt"))
    ave_z: np.ndarray = np.load(model_dir / "z.npy")
    input_tensor: torch.Tensor = get_input_tensor(input_ndarray, params)
    batch_size = input_tensor.shape[0]

    model.eval()
    outputs: list[np.ndarray] = []
    threshold = 0.024
    for genre, z in enumerate(ave_z):
        z = (
            torch.tensor(z, dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        with torch.no_grad():
            mu, logvar = model.encoder(input_tensor.to(device))
            z = model.reparameterize(mu, logvar)
            output: torch.Tensor = model.decoder(z, input_tensor.shape[1], z)
            output: np.ndarray = (
                output.view(-1, input_tensor.output).to("cpu").numpy().tolist()
            )
            outputs.append(output)

        chord: list[list[int]] = []
        recon_chord: pretty_midi.Instrument = pretty_midi.Instrument(0, name="chord")
        for i, chroma in enumerate(output):
            chroma_sum: float = sum(chroma)
            num_threshold = sorted(chroma)[-chord_num]
            for j, key in enumerate(chroma):
                if key < threshold * chroma_sum or key < num_threshold:
                    continue
                recon_chord.notes.append(make_note(70, j + 48, i, params.unittime))
            chroma = [
                x >= threshold * chroma_sum and x >= num_threshold for x in chroma
            ]
            chord.append(chroma)

        recon: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        recon.instruments.append(melody)
        recon.instruments.append(recon_chord)
        recon.instruments.append(connect_same_chord(chord, params.unittime))
        recon.write(output_dir / f"genre-{genre}.mid")
