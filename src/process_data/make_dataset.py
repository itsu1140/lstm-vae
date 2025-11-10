"""Make dataset from data path"""

import random
from pathlib import Path

import numpy as np
import torch
from main import Params
from torch.utils.data import Dataset


def dataset_input(params: Params) -> dict[str, object]:
    label = 0
    npy_path = Path("path/to/dataset")
    data = None
    labels = []
    label2genre = {}
    data_lower = 1000
    for genre in npy_path.iterdir():
        genre_vec = np.load(genre)
        if genre_vec.shape[0] // params.seq_len < data_lower:
            continue
        genre_vec = genre_vec.reshape((-1, params.seq_len, params.input_size))
        choice_idx = np.random.Generator.choice(
            genre_vec.shape[0],
            data_lower,
            replace=False,
        )
        genre_vec = genre_vec[choice_idx]
        data = genre_vec if data is None else np.concatenate((data, genre_vec), axis=0)
        labels += [label] * genre_vec.shape[0]
        label2genre[label] = genre.name[:-4]
        label += 1
    data = torch.tensor(data, dtype=torch.float32)
    data_dict = {
        "data": data,
        "chroma": data[:, :, -12:],
        "labels": torch.tensor(np.array(labels), dtype=torch.float32),
        "label2genre": label2genre,
    }
    return data_dict


class MidiDataset(Dataset):
    def __init__(self, params: Params) -> None:
        random.seed(0)
        data_dict = dataset_input(params)
        self.dataset = data_dict["data"]
        self.chroma = data_dict["chroma"]
        self.labels = data_dict["labels"]
        self.label2genre = data_dict["label2genre"]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dataset[index], self.labels[index], self.chroma[index]

    def get_label2genre(self) -> dict[int, str]:
        return self.label2genre
