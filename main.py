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

# nltk.download('all')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')

url = 'https://noticias.uol.com.br/'
csv_sep = ';'
stop_words = set(stopwords.words('portuguese'))


def get_html(news_url):
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
    return BeautifulSoup(content, 'html.parser')


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


def get_dates(html):
    dates = html.find(attrs={'class': 'headline-footer'}).find_all('time')

    if len(dates) < 2:
        return dates[0].get('datetime'), None
    return dates[0].get('datetime'), dates[1].get('datetime')


def update_and_get_old_url(html):
    news_links = get_news_links_from_html(html)

    try:
        url_data = open('URL_DATA.txt', 'r+')
    except FileNotFoundError:
        url_data = open('URL_DATA.txt', 'w+')

    old_urls = url_data.readlines()

    for link in news_links:
        f_link = f'{link}\n'
        if f_link not in old_urls:
            old_urls.append(f_link)
            url_data.writelines(f_link)

    url_data.close()
    return old_urls


def scrap_info(news_url, source):
    html = get_html(news_url)
    title = html.find('h1')
    autors = html.find_all(attrs={'class': 'solar-author-name'})
    date, update_date = get_dates(html)

    # Processamento do texto
    content = '\n'.join(map(str, (paragraph.get_text() for paragraph in html.find_all(attrs={'class': 'bullet'}))))
    content_without_pontuation = ''.join([char for char in content if char not in string.punctuation])
    tokens = word_tokenize(content_without_pontuation)
    tokens_normalized = [word.lower() for word in tokens]
    tokens_without_stopword = [word for word in tokens_normalized if word not in stop_words]
    stemmer = RSLPStemmer()  # alternativas: PorterStemmer, LancasterStemmer, SnowballStemmer
    tokens_stemmed = [[stemmer.stem(word) for word in token_list] for token_list in tokens_without_stopword]
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = [[lemmatizer.lemmatize(word) for word in token_list] for token_list in tokens_without_stopword]

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

    for link in links:
        if qtd_urls == 75:
            time.sleep(5)
            qtd_urls = 0

        f_link = link.rstrip("\n")
        try:
            result_list.append(scrap_info(f_link, source))
            print(f'A URL: "{f_link}" da fonte {source} foi analizada.')
            qtd_urls += 1
        except Exception:
            print(f'ERRO ao atualizar a URL: "{f_link}" da fonte {source}.')

    df = pd.DataFrame(result_list).fillna('NaN')

    # Persistindo dados nos foramtos CVS, JSON e XLSX
    df.to_csv(path_or_buf='uol_news_data.csv', sep=csv_sep, encoding='utf-8-sig')

    with pd.ExcelWriter('uol_news_data.xlsx') as writer:
        df.to_excel(excel_writer=writer, sheet_name=f'{source} News')

    df.to_json('uol_news_data.json')
