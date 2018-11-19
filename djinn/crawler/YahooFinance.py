import requests
from bs4 import BeautifulSoup

YAHOO_FINANCE_URL = "https://finance.yahoo.com/quote/"


def get_yahoo_report(stock):
    html = requests.get(YAHOO_FINANCE_URL + stock).text
    soup = BeautifulSoup(html, "lxml")

    rows = soup.find("div", {"id": "quote-summary"}).find_all("tr")

    result = {}
    for row in rows:
        name = row.contents[0].find("span").text
        value = row.contents[1].find("span").text if row.contents[1].find("span") is not None else row.contents[1].text

        result[name] = value

    return result



def main():
    for k, v in get_yahoo_report("AAPL").items():
        print(k + ":\t" + v + "\n")


if __name__ == "__main__":
    main()
