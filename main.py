import string

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
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
    response = requests.get(news_url)
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
    tokens = word_tokenize(content)
    tokens_normalizados = [word.lower() for word in tokens]
    # TODO: Validar pontuação concatenada a tokens
    tokens_sem_pontuacao = [word for word in tokens_normalizados if word not in string.punctuation]
    tokens_sem_stopword = [word for word in tokens_sem_pontuacao if word not in stop_words]

    return {
        'title': title.get_text(),
        'content': content,
        'tokens': tokens,
        'tokens_normalizados': tokens_normalizados,
        'tokens_sem_pontuacao': tokens_sem_pontuacao,
        'tokens_sem_stopword': tokens_sem_stopword,
        'autors': '\n'.join(map(str, (autor.get_text() for autor in autors))),
        'date': date,
        'update_date': update_date,
        'url': news_url,
        'source': source
    }


if __name__ == '__main__':
    source = 'UOL'
    html = get_html(url)

    result_list = []
    links = update_and_get_old_url(html)

    for link in links:
        f_link = link.rstrip("\n")
        try:
            result_list.append(scrap_info(f_link, source))
            print(f'A URL: "{f_link}" da fonte {source} foi analizada.')
        except Exception:
            print(f'ERRO ao atualizar a URL: "{f_link}" da fonte {source}.')

    df = pd.DataFrame(result_list)

    # Persistindo dados nos foramtos CVS, JSON e XLSX
    df.to_csv(path_or_buf='uol_news_data.csv', sep=csv_sep, encoding='utf-8-sig')

    with pd.ExcelWriter('uol_news_data.xlsx') as writer:
        df.to_excel(excel_writer=writer, sheet_name=f'{source} News')

    df.to_json('uol_news_data.json')
