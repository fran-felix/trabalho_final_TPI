"""Verifica se uma entrega segue o formato exigido no enunciado.

Uso (na raiz do projeto, com o ambiente tpi ativo):
    python src/verificar_entrega.py caminho/da/entrega
    python src/verificar_entrega.py caminho/da/entrega --n 300 --device cuda

O script confere os arquivos obrigatórios e os argumentos do app.py, carrega o
modelo pelo predict.py da entrega e o roda em imagens do teste do WFLW,
verificando o formato da saída, o tempo por imagem, a fração de faces
detectadas e o NME. Termina com código 0 se tudo estiver certo.

Associação entre faces: cada face anotada é comparada com todas as faces
previstas na mesma imagem e associada à de menor NME. Se nenhuma face prevista
tiver NME abaixo de MISSED_NME, a face anotada conta como não detectada.
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REQUIRED = ["README.md", "relatorio.pdf", "video_demo.mp4", "predict.py", "app.py",
            "train.py", "pesos/modelo.pt"]
APP_ARGS = ["--source", "--out", "--no-display"]
MAX_WEIGHTS_MB = 200
MAX_SECONDS_PER_IMAGE = 2.0  # em CPU
MISSED_NME = 0.5  # acima disso, a face anotada conta como não detectada


def load_test_images(data_dir: Path, n: int, seed: int = 0) -> dict:
    """Sorteia n imagens do teste do WFLW: {caminho: [landmarks de cada face anotada]}."""
    path = (data_dir / "WFLW_annotations" / "list_98pt_rect_attr_train_test"
            / "list_98pt_rect_attr_test.txt")
    faces = defaultdict(list)
    for line in path.read_text().splitlines():
        fields = line.split()
        faces[data_dir / "WFLW_images" / fields[206]].append(
            np.array(fields[:196], dtype=np.float64).reshape(98, 2))
    images = sorted(faces)
    chosen = np.random.default_rng(seed).choice(len(images), min(n, len(images)),
                                                replace=False)
    return {images[i]: faces[images[i]] for i in sorted(chosen)}


def nme(pred: np.ndarray, gt: np.ndarray) -> float:
    return np.linalg.norm(pred - gt, axis=-1).mean(-1) / np.linalg.norm(gt[60] - gt[72])


def check(ok: bool, message: str, errors: list) -> bool:
    print(("  [ok]   " if ok else "  [ERRO] ") + message)
    if not ok:
        errors.append(message)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--n", type=int, default=100, help="imagens de teste usadas")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sub = args.submission.resolve()
    data_dir = args.data.resolve()  # antes de mudar para a pasta da entrega
    errors = []

    print("1. Arquivos")
    for name in REQUIRED:
        check((sub / name).is_file(), name, errors)
    weights = sub / "pesos" / "modelo.pt"
    if weights.is_file():
        size = weights.stat().st_size / 2 ** 20
        check(size <= MAX_WEIGHTS_MB, f"pesos com {size:.0f} MB (máximo {MAX_WEIGHTS_MB})",
              errors)

    print("2. app.py")
    if (sub / "app.py").is_file():
        help_text = subprocess.run([sys.executable, "app.py", "--help"], cwd=sub,
                                   capture_output=True, text=True, timeout=120).stdout
        for arg in APP_ARGS:
            check(arg in help_text, f"argumento {arg}", errors)

    print("3. predict.py")
    if not ((sub / "predict.py").is_file() and weights.is_file()):
        check(False, "predict.py e pesos/modelo.pt são necessários", errors)
    else:
        os.chdir(sub)  # a entrega usa caminhos relativos à sua pasta
        sys.path.insert(0, str(sub))
        spec = importlib.util.spec_from_file_location("predict", sub / "predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.LandmarkModel(str(weights), device=args.device)

        import cv2
        images = load_test_images(data_dir, args.n)
        matched, missed, times, bad = [], 0, [], 0
        for path, annotated in images.items():
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # predict recebe RGB
            start = time.perf_counter()
            pred = model.predict(image)
            times.append(time.perf_counter() - start)
            pred = np.asarray(pred, dtype=np.float64)
            if pred.ndim != 3 or pred.shape[1:] != (98, 2) or not np.isfinite(pred).all():
                bad += 1
                missed += len(annotated)
                continue
            for gt in annotated:
                best = min((nme(p, gt) for p in pred), default=np.inf)
                if best < MISSED_NME:
                    matched.append(best)
                else:
                    missed += 1

        n_faces = sum(len(a) for a in images.values())
        check(bad == 0, f"saída (N, 98, 2) com valores finitos em todas as {len(images)} "
              f"imagens ({bad} com problema)", errors)
        per_image = float(np.median(times[1:] or times))
        limit = MAX_SECONDS_PER_IMAGE if args.device == "cpu" else float("inf")
        check(per_image <= limit,
              f"tempo por imagem: {1000 * per_image:.0f} ms ({args.device})", errors)
        check(len(matched) >= 0.5 * n_faces,
              f"faces detectadas: {len(matched)} de {n_faces} "
              f"({100 * len(matched) / n_faces:.1f}%)", errors)
        if matched:
            value = 100 * float(np.mean(matched))
            check(value < 20, f"NME nas faces detectadas: {value:.2f}% (acima de 20% "
                  "indica erro de formato, como x/y trocados)", errors)

    print("\nEntrega OK" if not errors else f"\n{len(errors)} problema(s) encontrado(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
