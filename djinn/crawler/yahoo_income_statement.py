import requests
import locale
from bs4 import BeautifulSoup
<<<<<<< HEAD
from crawler import YAHOO_FINANCIAL_REPORT_URL
||||||| merged common ancestors

YAHOO_FINANCE_URL = "https://finance.yahoo.com/quote/{}/financials"
=======

YAHOO_FINANCE_URL = "https://finance.yahoo.com/quote/{}/financials"
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
>>>>>>> pull setlocale out of function body #3


def get_income_statement(stock):
    html = requests.get(YAHOO_FINANCIAL_REPORT_URL.format(stock.lower())).text
    soup = BeautifulSoup(html, "lxml")

    matched_doms = soup.select("#mrt-node-Col1-3-Financials table tbody")

    if matched_doms is None:
        return {}
    else:
        rows = matched_doms[0].find_all("tr")
        result = {}

        for i in range(1, 4):
            row = rows[i]
            name = __get_cell_value(row, 0)
            value = locale.atoi(__get_cell_value(row, 1))

            result[name] = value

        return result


def __get_cell_value(row, col):
    return row.contents[col].find("span").text if row.contents[col].find("span") is not None else row.contents[col].text

