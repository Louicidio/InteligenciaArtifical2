# Previsão do Campeão de Fórmula 1 2025 usando Redes Neurais Artificiais

## Autores

[Seus Nomes Aqui]

## Resumo

Este trabalho apresenta a implementação de uma Rede Neural Artificial (RNA) do tipo Multi-Layer Perceptron (MLP) com algoritmo Backpropagation para prever o campeão do Mundial de Fórmula 1 de 2025. Utilizando dados históricos de 2015-2023 para treinamento e dados de 2024 para teste, foram experimentadas diferentes arquiteturas de rede neural variando o número de camadas ocultas, neurônios por camada e taxa de aprendizado. Os resultados demonstram que a MLP consegue capturar padrões complexos de performance dos pilotos e fornecer previsões com alta acurácia.

**Palavras-chave:** Redes Neurais Artificiais, Backpropagation, Fórmula 1, Previsão Esportiva, Machine Learning

## 1. Introdução

A Fórmula 1 é uma das modalidades esportivas mais tecnológicas e competitivas do mundo, onde múltiplos fatores influenciam o desempenho dos pilotos ao longo de uma temporada. A previsão do campeão mundial representa um desafio interessante para técnicas de inteligência artificial, pois envolve a análise de padrões complexos de performance.

Este trabalho propõe o uso de uma Rede Neural Artificial (RNA) do tipo Multi-Layer Perceptron (MLP) para prever qual piloto tem maior probabilidade de ser campeão mundial em 2025, baseando-se em dados históricos de performance.

## 2. Fundamentação Teórica

### 2.1 Redes Neurais Artificiais

As Redes Neurais Artificiais são modelos computacionais inspirados no funcionamento do cérebro humano, compostas por neurônios artificiais interconectados que processam informações através de conexões ponderadas.

### 2.2 Multi-Layer Perceptron (MLP)

O MLP é uma arquitetura de RNA feedforward com uma ou mais camadas ocultas entre a camada de entrada e saída. Cada neurônio aplica uma função de ativação não-linear aos dados de entrada ponderados.

### 2.3 Algoritmo Backpropagation

O Backpropagation é um algoritmo de aprendizado supervisionado que ajusta os pesos da rede através da propagação do erro calculado na saída de volta para as camadas anteriores.

## 3. Metodologia

### 3.1 Coleta e Preparação dos Dados

Os dados utilizados incluem:

-   Resultados de corridas (2015-2024)
-   Informações dos pilotos
-   Classificações de corridas
-   Pontuações por temporada

**Features extraídas:**

-   Taxa de vitórias
-   Taxa de pódios
-   Posição média nas corridas
-   Taxa de abandonos (DNF)
-   Pontos por corrida
-   Número total de corridas

### 3.2 Arquitetura da Rede Neural

Foram testadas diferentes configurações de MLP:

| Configuração | Camadas Ocultas | Neurônios | Taxa de Aprendizado |
| ------------ | --------------- | --------- | ------------------- |
| MLP-1        | 1               | 16        | 0.001               |
| MLP-2        | 1               | 32        | 0.001               |
| MLP-3        | 1               | 64        | 0.001               |
| MLP-4        | 2               | 32-16     | 0.001               |
| MLP-5        | 2               | 64-32     | 0.001               |
| MLP-6        | 3               | 64-32-16  | 0.001               |
| MLP-7        | 2               | 32-16     | 0.01                |
| MLP-8        | 2               | 32-16     | 0.0001              |

### 3.3 Parâmetros de Treinamento

-   **Função de ativação:** ReLU (camadas ocultas), Sigmoid (saída)
-   **Otimizador:** Adam
-   **Função de perda:** Binary Crossentropy
-   **Epochs:** 100
-   **Batch size:** 32
-   **Dropout:** 0.2
-   **Class weight:** {0: 1, 1: 10} (para balancear classes)

## 4. Experimentos e Resultados

### 4.1 Preparação dos Dados

O dataset foi construído com [X] registros de pilotos entre 2015-2023, resultando em [Y] amostras de treinamento com [Z] campeões identificados.

### 4.2 Resultados dos Experimentos

[Aqui você deve incluir a tabela com os resultados dos 8 experimentos]

| Configuração | Acurácia Treino | Acurácia Validação | Observações |
| ------------ | --------------- | ------------------ | ----------- |
| MLP-1        | X.XXXX          | Y.YYYY             |             |
| MLP-2        | X.XXXX          | Y.YYYY             |             |
| ...          | ...             | ...                |             |

### 4.3 Melhor Configuração

A melhor configuração encontrada foi [ESPECIFICAR], com acurácia de validação de [X.XXXX].

### 4.4 Previsão para 2025

Aplicando o melhor modelo aos dados de 2024, obtivemos o seguinte ranking:

1. [Piloto 1] - XX.X% de probabilidade
2. [Piloto 2] - XX.X% de probabilidade
3. [Piloto 3] - XX.X% de probabilidade

## 5. Discussão

### 5.1 Análise dos Resultados

A análise dos experimentos mostra que [DISCUSSÃO DOS RESULTADOS].

### 5.2 Limitações

-   Dataset limitado a dados históricos
-   Não considera fatores externos (mudanças de regulamento, transferências de pilotos)
-   Classes desbalanceadas (poucos campeões por ano)

## 6. Conclusão

Este trabalho demonstrou a viabilidade do uso de MLPs para previsão de campeões de Fórmula 1. A melhor configuração alcançou uma acurácia de [X]%, indicando que a rede neural consegue capturar padrões relevantes nos dados de performance dos pilotos.

## Referências

1. Haykin, S. (2009). Neural Networks and Learning Machines. 3rd Edition, Prentice Hall.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
3. [Adicionar outras referências relevantes]

---

**Nota:** Este é um template. Execute o código e preencha com os resultados reais obtidos nos experimentos.
