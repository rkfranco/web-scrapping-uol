import pandas as pd
import requests
from bs4 import BeautifulSoup

# URL da página a ser raspada
url = "https://www.informeblumenau.com/candidaturas-com-identidade-religiosa-crescem-225-em-24-anos/"

# Faça a requisição HTTP para obter o conteúdo da página
response = requests.get(url)
content = response.content

# Crie o objeto BeautifulSoup
soup = BeautifulSoup(content, 'html.parser')

# Extraia todas as tags de títulos e parágrafos
titles = soup.find_all(['h1'])
data = soup.find(attrs={'class': 'entry-meta-date'})
autor = soup.find(attrs={'class': 'entry-meta-author'})
classe = soup.find(attrs={'class': 'entry-content clearfix'})
paragraphs = classe.find_all('p')

# Exiba os resultados
print("Títulos:")
for title in titles:
    print(title.get_text())

print("autor:")
print(autor.get_text())

print("data:")
print(data.get_text())

print("Parágrafos:")
for paragraph in paragraphs:
    print(paragraph.get_text())

data = {
    "Títulos": '\n'.join(map(str, (title.get_text() for title in titles))),
    "Parágrafos": '\n'.join(map(str, (paragraph.get_text() for paragraph in paragraphs))),
    "autor": [autor.get_text()],
    "data": [data.get_text()],
}

# load data into a DataFrame object:
df = pd.DataFrame(data)
df.to_json('news_data.json')
df.to_csv(
    path_or_buf='news_data.csv',
    sep=',',  # não funciona direito com o texto, trocar por uma solução melhor
    encoding='utf-8',
)
