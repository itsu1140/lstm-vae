from dataclasses import dataclass


@dataclass
class Params:
    # model
    piano_size: int
    input_size: int
    output_size: int
    seq_len: int
    batch_size: int
    hidden_size: int
    latent_size: int
    num_layers: int
    beta_max: int

    # train
    epoch: int

    # midi
    offset: int
    unittime: int
