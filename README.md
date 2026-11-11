# Projeto Final de IA - Detecção de frutas com YOLO

Atividade Prática Final de Inteligência Artificial (YOLO).
Aluna: Ana Caroline

## Objetivo

Treinar um modelo YOLO para detectar frutas em imagens, avaliar o desempenho nos conjuntos de validação e teste e demonstrar o reconhecimento em imagens inéditas.

## Dataset

- Fonte: [Fruit Object Detection (Kaggle)](https://www.kaggle.com/datasets/eunpyohong/fruit-object-detection)
- Licença no Kaggle: desconhecida. Por isso, **as imagens do dataset não estão incluídas neste repositório**; apenas os rótulos (labels) e o `data.yaml`.
- 11 classes (IDs 0 a 10)
- Divisão usada: 798 imagens de treino, 320 de validação e 319 de teste (arquivos GIF com extensão .jpg foram removidos)

Para reproduzir, baixe o dataset no link acima e coloque as imagens em `dataset/images/{train,val,test}`, mantendo os mesmos nomes dos arquivos de `dataset/labels`.

## Treinamento

- Modelo base: `yolo26n.pt`
- 20 épocas, `imgsz=416`, `batch=4`, CPU

## Resultados

| Conjunto | Precisão | Recall | mAP50 |
|----------|----------|--------|-------|
| Validação | - | - | 0,846 |
| Teste | 0,797 | 0,850 | 0,888 |

## Estrutura do repositório

```
dataset/        data.yaml e labels (sem as imagens)
modelo/         best.pt (melhor modelo treinado)
resultados/     gráficos e métricas de treino, validação e teste
comparar_imagens.py   compara original, anotação e predição de 5 imagens
comandos.txt    todos os comandos utilizados, em ordem
relatorio.pdf   relatório com as 6 questões
```

## Como reproduzir

Os comandos completos (ambiente, preparação do dataset, treino, validação, teste e detecção) estão em [`comandos.txt`](comandos.txt).

## Uso de IA

Conforme solicitado pelo professor: o assistente de IA Claude (Anthropic) foi usado como apoio na execução passo a passo da atividade e na redação do relatório. Os treinos e resultados foram gerados localmente pela aluna.
