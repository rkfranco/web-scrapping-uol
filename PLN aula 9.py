"""
Bibliotecas para instalar:
pip install numpy==1.24.3 sentence-transformers spacy
python -m spacy download pt_core_news_sm
"""

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, PreTrainedModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

qtd_rows = 30

# Carregar o modelo mBART e o tokenizador
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load("pt_core_news_lg")

# Carregar modelo SBERT
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

tokenizer_t5 = T5Tokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
model_t5 = T5ForConditionalGeneration.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")


def busca_semantica(query: str,
                    model: SentenceTransformer,
                    data: pd.DataFrame,
                    document_embeddings: np.array,
                    top_n: int = 5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, document_embeddings)

    # Encontrar os documentos mais similares à query
    top_doc_indices = np.argsort(similarities[0])[-top_n:]
    for index in reversed(top_doc_indices):
        print(f"Documento similar à query '{query}':\n{data['title'][index]}")
        print(f"Similaridade: {similarities[0][index]}\n")


def resumo_extrativo(doc: str, model: SentenceTransformer, top_n: int = 3) -> str:
    sentences = [sent.text for sent in nlp(doc).sents]
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    # Calcular a similaridade de cosseno entre sentenças e o documento completo
    doc_embedding = model.encode([doc], convert_to_numpy=True)
    similarity_scores = cosine_similarity(doc_embedding, sentence_embeddings)

    # Selecionar as top_n sentenças mais similares ao documento como um todo
    top_sentence_indices = np.argsort(similarity_scores[0])[-top_n:]
    summary = " ".join([sentences[i] for i in sorted(top_sentence_indices)])

    return summary


def resumo_abstrativo(doc: str, model: SentenceTransformer, tokenizer: PreTrainedModel, max_length: int = 40,
                      min_length: int = 10) -> str:
    # Especificar o idioma de origem (português) e destino (português)
    tokenizer.src_lang = "pt_XX"

    # Tokenizar o texto de entrada
    inputs = tokenizer(doc, return_tensors='pt', max_length=512, truncation=True)

    # Garantir que o idioma de saída seja português
    forced_lang_id = tokenizer.lang_code_to_id["pt_XX"]

    # Gerar o resumo usando o modelo
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=3.0,
        num_beams=6,
        early_stopping=True,
        forced_bos_token_id=forced_lang_id  # Força o idioma de saída para português
    )

    # Decodificar o resumo gerado
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


def resumo_abstrativo_t5(
        doc: str,
        model: SentenceTransformer,
        tokenizer: PreTrainedModel,
        max_length: int = 70,
        min_length: int = 10) -> str:
    # Preparar a entrada para o T5
    inputs = tokenizer.encode(doc, return_tensors="pt", max_length=512, truncation=True)

    # Gerar o resumo
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=10.0,
                                 num_beams=20, early_stopping=True)

    # Decodificar o resumo gerado
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


if __name__ == '__main__':
    df = pd.read_csv('uol_news_data.csv', sep=';', encoding='utf-8-sig')
    data = df['content'][:30]

    # Gerar embeddings dos documentos
    doc_embeddings = sbert_model.encode(data, convert_to_numpy=True)

    print('--------------- EXTRATIVO ---------------')
    print(resumo_extrativo(data[0], sbert_model))

    print('--------------- ABSTRATIVO ---------------')
    print(resumo_abstrativo(data[0], mbart_model, tokenizer))
    print('------------- ABSTRATIVO T5 --------------')
    print(resumo_abstrativo_t5(data[0], model_t5, tokenizer_t5))
    print('------------ BUSCA SEMANTICA -------------')
    busca_semantica(
        'Lyne acredita que piloto tentou pousar avião no mar. Segundo ele, os danos nas asas, flaps e flaperons do avião sugerem que a aeronave estava envolvida em um pouso controlado, semelhante ao do Capitão Chesley Sully Sullenberger no Rio Hudson, em 2009 —história que virou até filme. Isso justifica sem sombra de dúvida a alegação original de que o [voo] MH370 tinha combustível e motores funcionando quando sofreu um magistral pouso controlado e não um acidente em alta velocidade com falta de combustível',
        sbert_model, df, doc_embeddings, top_n=1)
