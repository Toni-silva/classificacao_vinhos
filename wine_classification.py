# -*- coding: utf-8 -*-
"""
Script de Classificação de Vinhos

Este script utiliza o dataset de vinhos do Scikit-learn para treinar um modelo
de Machine Learning capaz de classificar vinhos em três categorias distintas
com base em suas características químicas.

Autor: [Seu Nome]
Data: 30/06/2025
"""

# 1. Importação das bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_wine

def carregar_dados():
    """Carrega o dataset de vinhos e o converte para um DataFrame do Pandas."""
    print("Carregando o dataset de vinhos...")
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print("Dataset carregado com sucesso!")
    return df, data.target_names

def explorar_dados(df):
    """Realiza uma análise exploratória básica dos dados."""
    print("\n--- Análise Exploratória dos Dados ---")
    print("\nPrimeiras 5 linhas do DataFrame:")
    print(df.head())

    print("\nInformações gerais do DataFrame:")
    df.info()

    print("\nEstatísticas descritivas:")
    print(df.describe())

    # Visualização da correlação entre as características
    print("\nGerando mapa de calor de correlações...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor de Correlação das Características do Vinho')
    plt.savefig('correlation_heatmap.png') # Salva a imagem para usar no README
    plt.show()
    print("Mapa de calor salvo como 'correlation_heatmap.png'")


def treinar_modelo(df):
    """Prepara os dados, treina e avalia o modelo de classificação."""
    print("\n--- Treinamento do Modelo de Machine Learning ---")

    # 2. Preparação dos Dados
    # Separando as variáveis independentes (features) da variável dependente (target)
    X = df.drop('target', axis=1)
    y = df['target']

    # Dividindo os dados em conjuntos de treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

    # 3. Treinamento do Modelo
    # Usaremos o RandomForest, um modelo robusto e popular
    print("\nTreinando o modelo RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    # 4. Avaliação do Modelo
    print("\n--- Avaliação do Modelo ---")
    y_pred = model.predict(X_test)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.2%}")

    # Relatório de Classificação (com precisão, recall, f1-score)
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Matriz de Confusão
    print("\nGerando Matriz de Confusão...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Real')
    plt.title('Matriz de Confusão')
    plt.savefig('confusion_matrix.png') # Salva a imagem
    plt.show()
    print("Matriz de Confusão salva como 'confusion_matrix.png'")

if __name__ == '__main__':
    # Bloco principal que executa todas as funções
    df, target_names = carregar_dados()
    explorar_dados(df)
    treinar_modelo(df)