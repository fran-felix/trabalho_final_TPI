"""Visualiza amostras do WFLW com landmarks, retângulo da face e atributos.

Uso:
    python src/visualize_wflw.py                      # 8 faces aleatórias do treino
    python src/visualize_wflw.py --split test --n 12 --seed 1
    python src/visualize_wflw.py --out amostras.png   # salva em vez de abrir janela
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ATTRIBUTES = ["pose", "expressão", "iluminação", "maquiagem", "oclusão", "desfoque"]
EYES = list(range(60, 76))  # 60–67: olho direito, 68–75: olho esquerdo
PUPILS = [96, 97]


def load_annotations(data_dir: Path, split: str) -> list[dict]:
    path = (data_dir / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
            / f"list_98pt_rect_attr_{split}.txt")
    samples = []
    for line in path.read_text().splitlines():
        fields = line.split()
        samples.append({
            "landmarks": np.array(fields[:196], dtype=np.float32).reshape(98, 2),
            "bbox": np.array(fields[196:200], dtype=np.float32),
            "attributes": np.array(fields[200:206], dtype=int),
            "image": data_dir / "WFLW_images" / fields[206],
        })
    return samples


def draw(ax, sample: dict, margin: float = 0.3) -> None:
    image = cv2.cvtColor(cv2.imread(str(sample["image"])), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = sample["bbox"]
    pts = sample["landmarks"]

    # recorta ao redor da face, com margem, para os landmarks ficarem visíveis
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    cx0, cy0 = int(max(x0 - mx, 0)), int(max(y0 - my, 0))
    cx1 = int(min(x1 + mx, image.shape[1]))
    cy1 = int(min(y1 + my, image.shape[0]))
    ax.imshow(image[cy0:cy1, cx0:cx1])

    pts = pts - [cx0, cy0]
    others = [i for i in range(98) if i not in EYES + PUPILS]
    ax.scatter(*pts[others].T, s=6, c="lime")
    ax.scatter(*pts[EYES].T, s=8, c="red")
    ax.scatter(*pts[PUPILS].T, s=12, c="yellow", marker="x")
    ax.add_patch(plt.Rectangle((x0 - cx0, y0 - cy0), x1 - x0, y1 - y0,
                               fill=False, edgecolor="cyan", linewidth=1))

    active = [a for a, v in zip(ATTRIBUTES, sample["attributes"]) if v]
    ax.set_title(", ".join(active) or "normal", fontsize=8)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--n", type=int, default=8, help="número de faces")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="salva a figura neste arquivo em vez de exibir")
    args = parser.parse_args()

    samples = load_annotations(args.data, args.split)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(samples), size=args.n, replace=False)

    cols = min(args.n, 4)
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, chosen):
        draw(ax, samples[idx])
    fig.suptitle(f"WFLW ({args.split}) — verde: face, vermelho: olhos, "
                 "amarelo: pupilas, ciano: retângulo", fontsize=10)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Figura salva em {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
