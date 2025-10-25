import requests
from bs4 import BeautifulSoup

def capturar_texto(url):
    resp = requests.get(url)
    resp.raise_for_status()  # para levantar erro se status != 200
    soup = BeautifulSoup(resp.text, "html.parser")
    # extrair todo o texto visível
    texto = soup.get_text(separator="\n")
    return texto

if __name__ == "__main__":
    url = "https://apps.univesp.br/manual-do-aluno/"
    conteudo = capturar_texto(url)
    print(conteudo)
