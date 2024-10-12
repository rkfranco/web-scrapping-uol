"""
Bibliotecas para instalar:
pip install spacy
python -m spacy download pt_core_news_lg
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('pt_core_news_lg')
qdt_rows = 30


def calcular_similaridade_tf_idf(data: list[str]) -> pd.DataFrame:
    try:
        # Vetorização com TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data)
        # Calculando a similaridade por cosseno entre os documentos
        cosine_sim = cosine_similarity(tfidf_matrix)
        return criar_dataframe_matriz(cosine_sim)
    except Exception as e:
        print(f"Erro ao calcular similaridade TF-IDF: {e}")
        return pd.DataFrame()


def calcular_similaridade_embedding_spacy(data: list[str]) -> pd.DataFrame:
    try:
        # Aplicando embedding
        embedded_data = [nlp(news) for news in data]
        # Calculando a similaridade por cosseno entre os documentos
        similaridade = [[doc1.similarity(doc2) for doc2 in embedded_data] for doc1 in embedded_data]
        return criar_dataframe_matriz(similaridade)
    except Exception as e:
        print(f"Erro ao calcular similaridade com embeddings do spaCy: {e}")
        return pd.DataFrame()


def criar_dataframe_matriz(matriz: list[list[float]]) -> pd.DataFrame:
    # Convertendo para um DataFrame para melhor visualização
    return pd.DataFrame(matriz, index=[f"Doc{i + 1}" for i in range(0, qdt_rows)],
                        columns=[f"Doc{i + 1}" for i in range(0, qdt_rows)])


def plotar_heatmap(data: pd.DataFrame, title: str):
    # Configurando o tamanho da figura
    plt.figure(figsize=(20, 16))
    # Gerando o heatmap da matriz de similaridade por cosseno
    sns.heatmap(data, annot=True, cmap='Blues', linewidths=0.5)
    # Definindo o título do heatmap
    plt.title(title)
    # Exibindo o gráfico
    plt.show()


if __name__ == '__main__':
    try:
        df = pd.read_csv('uol_news_data.csv', sep=';', encoding='utf-8-sig')
        data = df['content_without_pontuation'][:qdt_rows]

        similaridade_tf_idf = calcular_similaridade_tf_idf(data)
        similaridade_embedding = calcular_similaridade_embedding_spacy(data)

        plotar_heatmap(similaridade_tf_idf, 'Heatmap da Similaridade por Cosseno entre Documentos (TF-IDF)')
        plotar_heatmap(similaridade_embedding, 'Heatmap da Similaridade por Cosseno entre Documentos (SPACY)')
    except Exception as e:
        print(f"Erro ao executar o script: {e}")
