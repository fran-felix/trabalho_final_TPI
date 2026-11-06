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
    self.faces_treino = [] # [0, 7499] train
    self.faces_teste = [] # [7500, 9999] test
    self.tamanho_recorte = 256

    self.faces_treino = self.ler_anotacoes("train")
    # self.faces_teste = self.ler_anotacoes("test")

  def ler_anotacoes(self, split):
    arquivo = self.annotations / f"list_98pt_rect_attr_{split}.txt"
    faces = []

    for linha in arquivo.read_text().splitlines():
      campos = linha.split()
      faces.append(
        {
          "landmarks": np.array(campos[:196], dtype=np.float32).reshape(98, 2),
          "caixa": np.array(campos[196:200], dtype=np.float32),
          "atributos": np.array(campos[200:206], dtype=int),
          "imagem": self.img_dir / campos[206],
        }
      )

    return faces

  def __len__(self):
    return len(self.faces_treino) + len(self.faces_teste)

  def ler_imagem_rgb(self, caminho):
    """Lê uma imagem com OpenCV e converte BGR para RGB."""
    imagem = cv2.imread(str(caminho))
    if imagem is None:
      raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

  '''
  def imagem_para_vetor(self, caminho):
    imagem = self.ler_imagem_rgb(caminho)

    # OpenCV retorna HWC; modelos geralmente esperam tensores CHW numéricos.
    # No momento não estou convertendo para tensor por padronização, todos os outros valores são
    # retornados como np.array
    return np.transpose(imagem, (2, 0, 1)).copy()
    # imagem = np.transpose(imagem, (2, 0, 1)).copy()
    # return torch.from_numpy(imagem).to(dtype=torch.float32) / 255.0
  '''

  def matriz_recorte(self, caixa, margem=1.25):
    """
    Matriz afim 2x3 que leva a imagem original ao recorte TAMANHO x TAMANHO.
    Implementa p' = s (p - c) + (TAMANHO/2, TAMANHO/2).
    """
    x0, y0, x1, y1 = caixa
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])  # centro da caixa
    lado = margem * max(x1 - x0, y1 - y0)         # lado do quadrado
    s = self.tamanho_recorte / lado               # escala
    t = self.tamanho_recorte / 2 - s * c          # translação

    return np.array([[s, 0, t[0]], [0, s, t[1]]], dtype=np.float32)

  def transformar_pontos(self, pts, M):
      """Aplica a matriz afim 2x3 M a pontos (K, 2) -- a mesma conta que o warpAffine faz."""
      return pts @ M[:, :2].T + M[:, 2]
  
  def recortar(self, face, M=None):
    """Recorte RGB TAMANHO x TAMANHO da face e seus landmarks no sistema do recorte."""

    if M is None:
      M = self.matriz_recorte(face["caixa"])
    recorte = cv2.warpAffine(self.ler_imagem_rgb(face["imagem"]), M, (self.tamanho_recorte, self.tamanho_recorte))

    return recorte, self.transformar_pontos(face["landmarks"], M)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.item()
    if idx is None:
      raise RuntimeError(f"Index inválido, não foi possível acessar a posição: {idx}")

    face = self.faces_treino[int(idx)]
    face['imagem'], face['landmarks'] = self.recortar(face)

    sample = {
      "landmarks": face["landmarks"],
      "caixa": face["caixa"],
      "atributos": face["atributos"],
      "imagem": face["imagem"], # Retornando apenas a FACE, não a imagem inteira
    }
    if self.transform:
      sample = self.transform(sample)
    
    return sample


def test():
  face_dataset = DatasetAnotacoesFaces(ANNOTATIONS, IMAGES)

  print(len(face_dataset))
  for i in range(5):
    sample = face_dataset[i]
    print(i, sample['landmarks'].shape, sample['caixa'].shape, sample['atributos'].shape, sample['imagem'].shape)

test()

def test2():
  face_dataset = DatasetAnotacoesFaces(ANNOTATIONS, IMAGES)

  exemplo = next(f for f in teste if np.allclose(f["caixa"], [692, 90, 854, 326]))
  M = matriz_recorte(exemplo["caixa"])
  recorte, pts_recorte = recortar(exemplo, M)

  print("matriz M:\n", M.round(3))
  print("landmark 60 na imagem original:", exemplo["landmarks"][60].round(1))
  print("landmark 60 no recorte:        ", pts_recorte[60].round(1))

  imagem = face_dataset.ler_imagem_rgb(exemplo["imagem"])
  x0, y0, x1, y1 = exemplo["caixa"]
  c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
  lado = MARGEM * max(x1 - x0, y1 - y0)

  fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
  axs[0].imshow(imagem)
  desenhar_caixa(axs[0], exemplo["caixa"], "lime", lw=2)
  desenhar_caixa(axs[0], [*(c - lado / 2), *(c + lado / 2)], "yellow", lw=2, ls="--")
  axs[0].set_title(f"(a) original {imagem.shape[1]}x{imagem.shape[0]}\n"
                  "verde: caixa anotada | amarelo: quadrado com margem de 25%")
  axs[1].imshow(recorte)
  desenhar_landmarks(axs[1], pts_recorte, s=8)
  axs[1].set_title("(b) recorte 256x256\nlandmarks transformados com a mesma M")
  plt.show()