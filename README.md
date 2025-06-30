# Projeto de Classificação de Vinhos

Este projeto demonstra um fluxo de trabalho completo de Ciência de Dados, desde a análise exploratória até o treinamento e avaliação de um modelo de Machine Learning para classificar diferentes tipos de vinhos.

## 🎯 Objetivo

O objetivo é construir um modelo preditivo que possa classificar vinhos em uma de três classes (`class_0`, `class_1`, `class_2`) com base em 13 características químicas, como teor alcoólico, acidez, cor, entre outras.

## 💾 Dataset

Utilizei o dataset "Wine", disponível na biblioteca `scikit-learn`. Ele contém 178 amostras e 13 atributos.

## 🛠️ Ferramentas e Bibliotecas

* **Linguagem:** Python 3.11
* **Bibliotecas Principais:**
    * `pandas` para manipulação de dados.
    * `scikit-learn` para o modelo de machine learning e métricas.
    * `matplotlib` e `seaborn` para visualização de dados.

## ⚙️ Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Toni-silva/classificacao_vinhos.git
    cd classificacao_vinhos
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o script principal:**
    ```bash
    python wine_classification.py
    ```

## 📊 Resultados

O modelo `RandomForestClassifier` alcançou uma **acurácia de 97.22%** no conjunto de teste.

### Relatório de Classificação

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| class_0      |      1.00 |   0.92 |     0.96 |      12 |
| class_1      |      0.93 |   1.00 |     0.97 |      14 |
| class_2      |      1.00 |   1.00 |     1.00 |      10 |
|              |           |        |          |         |
| **accuracy** |           |        |   **0.97** |      36 |
| macro avg    |      0.98 |   0.97 |     0.97 |      36 |
| weighted avg |      0.97 |   0.97 |     0.97 |      36 |


### Matriz de Confusão

A matriz de confusão mostra que o modelo teve um desempenho excelente, com apenas um erro de classificação.

![Matriz de Confusão](confusion_matrix.png)

### Mapa de Calor de Correlações

A análise exploratória incluiu um mapa de calor para visualizar a correlação entre as características.

![Mapa de Calor](correlation_heatmap.png)

## 🚀 Próximos Passos

* Testar outros algoritmos de classificação (ex: `XGBoost`, `SVM`).
* Realizar um ajuste fino de hiperparâmetros com `GridSearchCV`.
* Empacotar o modelo treinado para fazer previsões em novos dados.