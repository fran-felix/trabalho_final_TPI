# Trabalho Prático — Facial Landmark Detection para Detecção de Sonolência

Trabalho final de Visão Computacional. O enunciado completo está em `enunciado.pdf`.

## Estrutura

```
enunciado.pdf     enunciado do trabalho
guia_basico.ipynb notebook introdutório: dataset, recorte, heatmaps, NME, EAR, análise temporal
src/              scripts de apoio (download e visualização do dataset, verificação da entrega)
environment.yml   dependências (conda)
data/             dataset (criado pelo script de download, não versionar)
```

## Ambiente

```bash
conda env create -f environment.yml   # cria o ambiente "tpi"
conda activate tpi
```

Para atualizar um ambiente já existente após mudanças no `environment.yml`:

```bash
conda env update -n tpi -f environment.yml
```

O PyTorch é instalado com suporte a CUDA 12.1 (requer driver NVIDIA >= 530);
sem GPU, ele funciona normalmente em CPU. O OpenCV fica na versão 4.x.

O ambiente define `PYTHONNOUSERSITE=1`, para que pacotes instalados com
`pip install --user` (em `~/.local`) não substituam os do ambiente. A variável
só vale depois de reativar o ambiente (`conda deactivate && conda activate tpi`).

## Dataset: WFLW

O [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) (Wider Facial Landmarks
in the Wild) contém 10.000 faces anotadas com 98 landmarks: 7.500 para treino e
2.500 para teste.

### Download

```bash
python src/download_wflw.py            # salva em ./data
python src/download_wflw.py --dest outro_dir
```

O script baixa o dataset completo (~725 MB) de um
[espelho público no Kaggle](https://www.kaggle.com/datasets/mrriandmstique/wflw-wider-facial-landmarks-in-the-wild),
sem necessidade de login, e extrai tudo no diretório de destino. O Google Drive
oficial costuma bloquear o download por excesso de acessos, por isso não é usado.

Estrutura resultante:

```
data/
├── WFLW_images/                        imagens, organizadas por categoria
└── WFLW_annotations/
    └── list_98pt_rect_attr_train_test/
        ├── list_98pt_rect_attr_train.txt   7.500 faces
        └── list_98pt_rect_attr_test.txt    2.500 faces
```

### Visualização

```bash
python src/visualize_wflw.py                       # 8 faces aleatórias do treino
python src/visualize_wflw.py --split test --n 12 --seed 1
python src/visualize_wflw.py --out amostras.png    # salva em arquivo
```

Mostra cada face recortada com os landmarks (olhos em vermelho, pupilas em
amarelo), o retângulo da face e os atributos marcados.

### Formato das anotações

Cada linha dos arquivos `.txt` descreve uma face, com 207 campos separados por
espaço:

| Campos | Conteúdo |
|---|---|
| 1–196 | coordenadas dos 98 landmarks: `x0 y0 x1 y1 ... x97 y97` |
| 197–200 | retângulo da face: `x_min y_min x_max y_max` |
| 201–206 | atributos binários: pose, expressão, iluminação, maquiagem, oclusão, desfoque |
| 207 | caminho da imagem, relativo a `WFLW_images/` |

Nos atributos, `0` indica condição normal e `1` indica condição difícil (pose
extrema, expressão exagerada, iluminação extrema, maquiagem, oclusão ou
desfoque).

## Guia introdutório (notebook)

O `guia_basico.ipynb` percorre, com código curto e figuras, as ideias básicas do
trabalho: formato do WFLW, recorte da face e transformação dos landmarks,
espelhamento com o `Mirror98.txt`, heatmaps e soft-argmax, NME, um treino mínimo
de demonstração, EAR, detector YuNet e análise temporal. Ele não é a solução: o
modelo treinado nele é propositalmente simples.

O notebook roda no Google Colab sem preparação: a primeira célula baixa o
dataset (pelo mesmo espelho do Kaggle) para `/content/data` e o detector YuNet
para `/content/modelos`, pastas visíveis no painel de arquivos. Escolha um ambiente
de execução com GPU (Ambiente de execução → Alterar o tipo de ambiente de
execução → GPU).

Para rodar localmente, abra o notebook a partir da raiz do projeto, com o kernel
do ambiente `tpi` (VS Code) ou com `jupyter lab`, depois de baixar o dataset.
