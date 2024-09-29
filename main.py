import string
import time

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

"""
    Trabalho prático 1 - Processamento de Linguagem Natural
    Alunos: Ana Carolina, Gustavo Bruder, Luan Guarnieri, Maria Eduarda, Nicole B., Rodrigo Franco

    * O objetivo deste trabalho é coletar e analisar informações de notícias de forma automatizada, utilizando técnicas de web scraping para extrair dados, sendo escolhido o portal de notícias "UOL". Realizar a extração automática de dados de notícias nos permitiria analisar tendências e monitorar a frequência de certos temas abordados, possibilitando também identificar padrões de comportamento da mídia e acompanhar a evolução de determinados tópicos. 
    * A coleta dos dados pode ser usada, por exemplo, para a detecção de Fake News, analisando o conteúdo das notícias e aplicando técnicas de processamento de linguagem natural (como lematização e stemização), possibilitando detectar padrões textuais que possam indicar a veracidade de uma notícia.
    * Outro exemplo que poderíamos aplicar seria para realizar análise de tendências e padrões, visto que o armazenamento das notícias permite a análise de dados em larga escala, ajudando a identificar tendências de tópicos abordados em determinado período.
"""

# download dos pacotes NLKT, para processar textos
# nltk.download('all')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')

url = 'https://noticias.uol.com.br/'
csv_sep = ';'
stop_words = set(stopwords.words('portuguese'))


# obtém o HTML da URL recebida como parâmetro
def get_html(news_url):
    # define um cabeçalho de uma requisição HTTP, usado para simular um navegador e evitar bloqueio
    header = {
        'referer': 'https://www.scrapingcourse.com/ecommerce/',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/json',
        'accept-encoding': 'gzip, deflate, br',
        'sec-ch-device-memory': '8',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-platform': "Windows",
        'sec-ch-ua-platform-version': '"10.0.0"',
        'sec-ch-viewport-width': '792',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    }

    response = requests.get(news_url, headers=header)
    content = response.content
    # utilizar BeautifulSoup para facilitar a extração dos dados e análise do html
    return BeautifulSoup(content, 'html.parser')


# extrai os links das notícias, dentro do HTML, retornando uma lista de links válidos
def get_news_links_from_html(html):
    links_news = []

    for link in html.findAll('a'):
        href = link.get('href')
        if (href and
                url in href and  # Valida se a noticia e do UOL
                'amp-stories' not in href and  # Valida se nao e um 'story'
                'htm' in href  # Valida se e um arquivo valido
        ):
            links_news.append(href)

    return links_news


# extrai as datas de uma notícia
def get_dates(html):
    # no footer é aonde se encontram as datas
    dates = html.find(attrs={'class': 'headline-footer'}).find_all('time')

    if len(dates) < 2:
        return dates[0].get('datetime'), None
    # se existir mais de uma, há atualização, retornando ambas
    return dates[0].get('datetime'), dates[1].get('datetime')


# atualiza os links salvos
def update_and_get_old_url(html):
    news_links = get_news_links_from_html(html)

    # tentar abrir o arquivo, se não existir cria um novo
    try:
        url_data = open('URL_DATA.txt', 'r+')
    except FileNotFoundError:
        url_data = open('URL_DATA.txt', 'w+')

    old_urls = url_data.readlines()

    # lê todos os links novos. Se encontrar algum novo, salva no arquivo de links
    for link in news_links:
        f_link = f'{link}\n'
        if f_link not in old_urls:
            old_urls.append(f_link)
            url_data.writelines(f_link)

    url_data.close()
    return old_urls


# coleta informações específicas de cada notícia
def scrap_info(news_url, source):
    html = get_html(news_url)
    title = html.find('h1')  # busca o título
    autors = html.find_all(attrs={'class': 'solar-author-name'})  # busca os autores
    date, update_date = get_dates(html)  # busca as datas

    # busca o conteúdo da notícia usando apenas parágrafos com a classe 'bullet'
    content = '\n'.join(map(str, (paragraph.get_text() for paragraph in html.find_all(attrs={'class': 'bullet'}))))

    content_without_pontuation = ''.join(
        [char for char in content if char not in string.punctuation])  # remove pontuação
    tokens = word_tokenize(content_without_pontuation)  # "tokeniza" o conteúdo
    tokens_normalized = [word.lower() for word in tokens]  # normaliza transformando tudo para minúsculo
    tokens_without_stopword = [word for word in tokens_normalized if word not in stop_words]  # remove spotwords

    stemmer = RSLPStemmer()  # alternativas: PorterStemmer, LancasterStemmer, SnowballStemmer
    tokens_stemmed = [stemmer.stem(token) for token in tokens_without_stopword]

    # aplica lematização
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens_without_stopword]

    # retorna em um formato que possa ser salvo para CSV
    return {
        'title': title.get_text(),
        'content': content,
        'content_without_pontuation': content_without_pontuation,
        'tokens': tokens,
        'tokens_normalized': tokens_normalized,
        'tokens_without_stopword': tokens_without_stopword,
        'tokens_stemmed': tokens_stemmed,
        'tokens_lemmatized': tokens_lemmatized,
        'autors': '\n'.join(map(str, (autor.get_text() for autor in autors))),
        'date': date,
        'update_date': update_date,
        'url': news_url,
        'source': source
    }


if __name__ == '__main__':
    source = 'UOL'
    html = get_html(url)
    qtd_urls = 0

    result_list = []
    links = update_and_get_old_url(html)

    # itera sobre todos os links obtidos de notícias
    for link in links:
        if qtd_urls == 75:
            # se processou 75 urls, pausa por 5 segundos para evitar bloqueio
            time.sleep(5)
            qtd_urls = 0

        f_link = link.rstrip("\n")
        try:
            result_list.append(scrap_info(f_link, source))
            print(f'A URL: "{f_link}" da fonte {source} foi analizada.')
            qtd_urls += 1
        except Exception:
            print(f'ERRO ao atualizar a URL: "{f_link}" da fonte {source}.')

    # converte a lista de resultados em um DataFrame
    df = pd.DataFrame(result_list).fillna('NaN')

    # Persistindo dados nos foramtos CVS, JSON e XLSX
    df.to_csv(path_or_buf='uol_news_data.csv', sep=csv_sep, encoding='utf-8-sig')

    with pd.ExcelWriter('uol_news_data.xlsx') as writer:
        df.to_excel(excel_writer=writer, sheet_name=f'{source} News')

    df.to_json('uol_news_data.json')
