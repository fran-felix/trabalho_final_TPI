"""Baixa o dataset WFLW (imagens + anotações).

Uso:
    python src/download_wflw.py [--dest data]

Usa um espelho público do Kaggle (não exige login), pois o Google Drive
oficial costuma bloquear downloads por excesso de acessos.
Fonte oficial: https://wywu.github.io/projects/LAB/WFLW.html
"""

import argparse
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "mrriandmstique/wflw-wider-facial-landmarks-in-the-wild"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()

    path = kagglehub.dataset_download(KAGGLE_DATASET, output_dir=str(args.dest))
    print(f"Dataset pronto em {Path(path).resolve()}")


if __name__ == "__main__":
    main()
