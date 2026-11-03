# Classificação da Qualidade Visual de Lâminas de Madeira Compensada com CNNs

Código dos experimentos do trabalho **"Avaliação da Qualidade Visual de Lâminas de Madeira Compensada utilizando CNNs via Transferência de Aprendizado"** (Khauan Gonçalves Barbosa, IFPR – campus Palmas, 2026).

O script classifica imagens de lâminas de *Pinus* em cinco categorias de qualidade (**A, B, C, D e E**), com base nos critérios da norma PS 1-22 e em adaptações industriais, e compara abordagens clássicas e de aprendizado profundo.

## Experimentos

O script `classificar_laminas.py` executa, em sequência, 19 experimentos (com as três arquiteturas):

| Etapa | Experimentos | Descrição |
|---|---|---|
| `baseline` | `Hist RGB + SVM` | Histograma de cor RGB (32 *bins* por canal, imagem redimensionada para 128×128) + SVM (kernel RBF, `C=1.0`) |
| `cnn` | `<Modelo> + TL`, `<Modelo> + TL + DA`, `<Modelo> (Do Zero)`, `<Modelo> + DA (Do Zero)` | ResNet50, EfficientNet-B5 e EfficientNet-B7 treinadas fim a fim (12 experimentos). **TL**: pesos do ImageNet com ajuste fino de *todas* as camadas. **Do Zero**: pesos aleatórios. **DA**: aumento de dados no treino |
| `embeddings` | `Embeddings <Modelo> + TL + SVM`, `Embeddings <Modelo> (pesos aleatórios) + SVM` | Vetores da camada de agrupamento global de uma rede **congelada** (nenhum peso é atualizado), classificados por SVM (6 experimentos) |

## Estrutura esperada dos dados

As imagens **não estão incluídas neste repositório**. Organize-as assim:

```
Base_Original/
├── A/   (imagens da classe A)
├── B/
├── C/
├── D/
└── E/
```

## Instalação

Requer Python 3.9 ou superior e, de preferência, uma GPU com CUDA (as EfficientNets são muito lentas em CPU).

```bash
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Na primeira execução, o `torchvision` baixa os pesos do ImageNet automaticamente (é necessário acesso à internet).

## Uso

```bash
# todos os experimentos
python classificar_laminas.py --base /caminho/para/Base_Original

# apenas uma arquitetura
python classificar_laminas.py --base /caminho/para/Base_Original --modelos resnet50

# apenas o baseline clássico e os embeddings
python classificar_laminas.py --base /caminho/para/Base_Original --etapas baseline embeddings

# reutilizar o mesmo conjunto de teste de uma execução anterior
python classificar_laminas.py --base /caminho/para/Base_Original --split-csv resultados_experimentos/split.csv
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--base` | (obrigatório) | Pasta com as subpastas `A` a `E` |
| `--saida` | `resultados_experimentos` | Pasta onde os resultados são gravados |
| `--modelos` | todos | `resnet50`, `efficientnet_b5`, `efficientnet_b7` |
| `--etapas` | todas | `baseline`, `cnn`, `embeddings` |
| `--epocas` | 100 | Máximo de épocas |
| `--paciencia` | 5 | Paciência do *early stopping* (perda de validação) |
| `--batch` | 16 | Tamanho do lote (reduza se faltar memória da GPU) |
| `--workers` | 4 | Processos do `DataLoader` |
| `--split-csv` | — | Usa um `split.csv` existente em vez de gerar uma nova divisão |

Os experimentos já concluídos são **pulados** (o script verifica a existência do JSON correspondente). Para refazer um experimento, apague os arquivos dele em `--saida`, ou use outra pasta de saída.

## Saídas

Em `--saida` (padrão `resultados_experimentos/`):

| Arquivo | Conteúdo |
|---|---|
| `exp_<experimento>.json` | Acurácia, precisão/*recall*/F1 ponderados, matriz de confusão, métricas por classe e erros por distância entre classes (adjacentes, distância 2 e distância ≥ 3) |
| `cm_<experimento>.png` | Matriz de confusão |
| `resultado_consolidado_geral.csv` | Tabela-resumo de todos os experimentos, ordenada por acurácia |
| `split.csv` | Arquivos de treino e de teste usados na execução |

## Protocolo experimental

- **Divisão:** *holdout* estratificado, 80% treino / 20% teste (`random_state=42`). Para as CNNs, 15% do treino é separado (estratificado) como validação para o *early stopping*; o teste só é usado ao final. A divisão é única, sem validação cruzada.
- **Pré-processamento (todas as imagens):** redimensionamento para 224×224 e normalização com a média e o desvio-padrão do ImageNet. Todas as arquiteturas usam 224×224, abaixo da resolução nativa das EfficientNet-B5 (456) e B7 (600).
- **Aumento de dados (apenas treino das CNNs):** espelhamentos horizontal e vertical (p = 0,5), rotação de até 15° e variação de brilho e contraste (fator 0,1).
- **Treinamento:** Adam, `lr=1e-4`, lote 16, até 100 épocas, *early stopping* com paciência 5 na perda de validação; os pesos da melhor época são restaurados. Sem pesos de classe nem reamostragem.
- **SVM:** kernel RBF, `C=1.0`, sem ajuste de hiperparâmetros e sem normalização dos atributos.
- **Embeddings:** extraídos da camada de agrupamento global (2048 dimensões na ResNet50 e na EfficientNet-B5, 2560 na EfficientNet-B7). Sempre vêm dos pesos originais do ImageNet (ou aleatórios), **nunca** dos modelos ajustados. Treino e teste são extraídos pela **mesma** rede.
- **Métricas:** acurácia e médias ponderadas de precisão, *recall* e F1-score.
- **Aumento de dados nos métodos clássicos:** não é aplicado. Espelhamentos não alteram histogramas de cor, e o aumento não afeta embeddings de uma rede congelada, então essas variações seriam equivalentes às versões sem aumento.

## Reprodutibilidade

- Semente fixa (`42`) para Python, NumPy e PyTorch, com `cudnn` determinístico.
- As imagens são listadas em **ordem alfabética**, o que torna a divisão determinística entre máquinas.
- Mesmo assim, pequenas diferenças de resultado podem ocorrer entre versões de PyTorch/CUDA e tipos de GPU.
- Os pesos pré-treinados são os `weights="DEFAULT"` do `torchvision` (ImageNet-1K); a versão exata dos pesos depende da versão instalada do `torchvision`. Registre as versões que usar.
- Ambiente usado no trabalho: Intel Core i7-13700F, 32 GB de RAM, NVIDIA GeForce RTX 4070 Ti.

## Dados

A base utilizada contém 2.346 imagens reais (A: 220, B: 301, C: 493, D: 813, E: 519), coletadas em linha de produção de uma indústria de compensados em Palmas-PR por Cristian Gotardo (UTFPR – Dois Vizinhos). As imagens **não são distribuídas** neste repositório. Para utilizá-las, consulte os responsáveis pela base.

