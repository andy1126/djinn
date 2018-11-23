#!/usr/bin/env python 


import crawler.yahoo_finance

if __name__ == "__main__":
    print("P/E ration: %s" % crawler.yahoo_finance.get_pe_ratio("AAPL"))
