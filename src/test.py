from dataset import DatasetAnotacoesFaces, ANNOTATIONS, IMAGES, REGIOES, CORES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def desenhar_caixa(ax, caixa, cor, **kwargs):
  x0, y0, x1, y1 = caixa
  ax.add_patch(pat.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cor, **kwargs))

def desenhar_landmarks(ax, pts, s=10, legenda=False):

  """Desenha os 98 pontos, com uma cor por região."""
  for nome, indices in REGIOES.items():
      ax.scatter(*pts[list(indices)].T, s=s, c=CORES[nome], label=nome)
  if legenda:
      ax.legend(fontsize=8, loc="lower right", markerscale=1.5)


def test1():
  face_dataset = DatasetAnotacoesFaces(ANNOTATIONS, IMAGES)

  print(len(face_dataset))
  for i in range(5):
    sample = face_dataset[i]
    print(i, sample['landmarks'].shape, sample['caixa'].shape, sample['atributos'].shape, sample['imagem'].shape)

def test2():
  face_dataset = DatasetAnotacoesFaces(ANNOTATIONS, IMAGES)

  exemplo = next(f for f in face_dataset.faces if np.allclose(f["caixa"], [692, 90, 854, 326]))
  M = face_dataset.matriz_recorte(exemplo["caixa"])
  recorte, pts_recorte = face_dataset.recortar(exemplo, M)

  print("matriz M:\n", M.round(3))
  print("landmark 60 na imagem original:", exemplo["landmarks"][60].round(1))
  print("landmark 60 no recorte:        ", pts_recorte[60].round(1))

  imagem = face_dataset.ler_imagem_rgb(exemplo["imagem"])
  x0, y0, x1, y1 = exemplo["caixa"]
  c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
  lado = 1.25 * max(x1 - x0, y1 - y0)

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


test1()
test2()