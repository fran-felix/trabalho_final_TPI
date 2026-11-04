from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from visualize_wflw import draw

DATA = Path("/home/francisco/trabalho_final_TPI/data")
ANNOTATIONS = DATA / "WFLW_annotations/list_98pt_rect_attr_train_test"
IMAGES = DATA / "WFLW_images"


class DatasetAnotacoesFaces(Dataset):
  def __init__(self, annotations, img_dir, transform=None):
    self.annotations = Path(annotations)
    self.img_dir = Path(img_dir)
    self.transform = transform
    self.faces = [] # [0, 7499] train [7500, 9999] test

    for split in ("train", "test"):
      arquivo = self.annotations / f"list_98pt_rect_attr_{split}.txt"
      for linha in arquivo.read_text().splitlines():
        campos = linha.split()
        self.faces.append(
          {
            "landmarks": np.array(campos[:196], dtype=np.float32).reshape(98, 2),
            "caixa": np.array(campos[196:200], dtype=np.float32),
            "atributos": np.array(campos[200:206], dtype=int),
            "imagem": self.img_dir / campos[206],
          }
        )

  def __len__(self):
    return len(self.faces)

  def ler_imagem_rgb(self, caminho):
    """Lê uma imagem com OpenCV e converte BGR para RGB."""
    imagem = cv2.imread(str(caminho))
    if imagem is None:
      raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.item()
    if idx is None:
      raise RuntimeError(f"Index inválido, não foi possível acessar a posição: {idx}")

    face = self.faces[int(idx)]
    sample = {
      #"image": self.ler_imagem_rgb(face["imagem"]),
      "landmarks": face["landmarks"],
      "caixa": face["caixa"],
      "atributos": face["atributos"],
      "imagem": face["imagem"],
    }
    if self.transform:
      sample = self.transform(sample)
    
    return sample


def test():
  face_dataset = DatasetAnotacoesFaces(ANNOTATIONS, IMAGES)

  for i, sample in enumerate(face_dataset.faces):
      print(i, sample['landmarks'].shape, sample['caixa'].shape, sample['atributos'].shape, sample['imagem'])


      ax = plt.subplot(1, 4, i + 1)
      #draw(ax, sample)
      plt.tight_layout()
      ax.set_title('Sample #{}'.format(i))
      ax.axis('off')

      if i == 3:
        plt.show()
        break

test()