import requests
from bs4 import BeautifulSoup
from crawler import YAHOO_FINANCE_URL
from cache.in_memory_cache import cache

@cache
def __get_summary(ticker_symbol):
    html = requests.get(YAHOO_FINANCE_URL + ticker_symbol).text
    soup = BeautifulSoup(html, "lxml")

    if soup.find("div", {"id": "quote-summary"}) is None:
        return {}

    rows = soup.find("div", {"id": "quote-summary"}).find_all("tr")

    result = {}
    for row in rows:
        name = row.contents[0].find("span").text
        value = row.contents[1].find("span").text if row.contents[1].find("span") is not None else row.contents[1].text

        result[name] = value

    return result


def get_pe_ratio(ticker_symbol):
    return __get_summary(ticker_symbol)["PE Ratio (TTM)"]
