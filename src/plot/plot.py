from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from train.train import History


def tsne(label2genre: dict, history: History, output_path: Path):
    seed = 0
    cmap = plt.get_cmap("tab20")
    tsne = TSNE(perplexity=30, random_state=seed)
    tsne_z = tsne.fit_transform(history.z)
    for lbl in iter(label2genre):
        idx = lbl == history.label
        plt.scatter(
            tsne_z[idx, 0],
            tsne_z[idx, 1],
            color=cmap(lbl),
            label=label2genre[lbl],
            s=10,
        )
    plt.legend(loc="upper left")
    plt.title("t-SNE")
    plt.savefig(output_path / "tsne.eps")
    plt.savefig(output_path / "tsne.png")
