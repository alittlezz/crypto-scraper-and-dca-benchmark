import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from decimal import *
import pandas as pd
import time
import math
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pprint
import statistics
import matplotlib.pyplot as plt
import copy
import numpy as np

getcontext().prec = 20
EPSILON = 1e-1
MARKET_CAP_MIN = Decimal(10_000)
symbols = []
prices = []
market_caps = []
dates = []
START_DATE = datetime(2014, 1, 1)
FINISH_DATE = datetime(2022, 1, 1)


def scrape_for_date(driver, year, month):
    URL = f"https://coinmarketcap.com/historical/{year}{month}01/"
    print(f"Processing month {month} and year {year}")

    # driver = webdriver.Chrome('chromedriver')
    driver.get(URL)

    # driver.find_element_by_xpath("//td[@aria-label='Market Cap: activate to sort column descending']").click()
    # WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH,"//td[text()='Market Cap']"))).click()
    for i in range(120):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        if (i + 1) % 20 == 0:
            try:
                driver.find_element(
                    By.XPATH, "//button[text() = 'Load More']").click()
            except:
                print(f"Found last load more button at iteration {i + 1}!")
                break
    i = 0
    data = []
    cr_date = f"01/{month}/{year}"
    for row in driver.find_elements(By.TAG_NAME, "tr"):
        i += 1
        if i > 3:
            try:
                attrs = row.text.split("\n")
                rank = int(attrs[0])
                assert rank == i - 3
                symbol = attrs[2]
                market_cap = Decimal(attrs[3][1:].replace(",", ""))
                price = Decimal(attrs[4][1:].replace(",", ""))
                if market_cap < MARKET_CAP_MIN:
                    continue
                data.append((symbol, market_cap, price, cr_date))
                # print(rank, symbol, market_cap, price)
            except:
                continue
                # print('ERROR at', attrs)

    data = sorted(data, key=lambda x: x[1], reverse=True)
    global symbols, market_caps, prices, dates
    symbols = symbols + [d[0] for d in data]
    market_caps = market_caps + [d[1] for d in data]
    prices = prices + [d[2] for d in data]
    dates = dates + [d[3] for d in data]

    # soup = BeautifulSoup(page.content, "html.parser")
    # for row in soup.find_all('tr', class_='cmc-table-row')[-3:]:
    #     print('Row:', row.prettify())


def is_stablecoin(df, ticker):
    prices = df[df["Ticker"] == ticker]["Price"]
    avg = prices.mean()
    mn = prices.min()
    mx = prices.max()
    return math.fabs(avg - 1) < EPSILON and mn > 0.8 and mx < 1.2


def filter_data(df):
    iss = []
    dic = {}
    for ticker in df["Ticker"].unique():
        dic[ticker] = is_stablecoin(df, ticker)
    for ticker in df["Ticker"]:
        iss.append(dic[ticker])
    df["Stablecoin"] = iss

    df_final = pd.DataFrame()
    for year in range(2014, 2023):
        if year == 2022:
            month_end = 4
        else:
            month_end = 13
        for month in range(1, month_end):
            smonth = str(month)
            if month < 10:
                smonth = "0" + smonth
            cr_date = "01/" + smonth + "/" + str(year)
            entries = df[(df["Date"] == cr_date) & (df["Stablecoin"] == False)]
            if len(entries) == 0:
                print(cr_date)
            df_final = pd.concat([df_final, entries])
    return df_final


def scrape_data():
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1200")

    driver = webdriver.Chrome("chromedriver", options=options)
    for year in range(2014, 2023):
        if year == 2022:
            month_end = 4
        else:
            month_end = 13
        for month in range(1, month_end):
            smonth = str(month)
            if month < 10:
                smonth = "0" + smonth
            scrape_for_date(driver, str(year), smonth)
    df = pd.DataFrame()
    df["Ticker"] = symbols
    df["Price"] = prices
    df["Market Cap"] = market_caps
    df["Date"] = dates


def get_day_weights(df, topk, skip_first=0):
    df = df[skip_first: topk + skip_first]
    dic = {}
    sm = Decimal(df["Market Cap"].sum())
    for ticker, mc in zip(df["Ticker"], df["Market Cap"]):
        dic[ticker] = Decimal(mc) / sm
    return dic


def get_day_weights_bottom(df, topk):
    df = df[-topk:]
    dic = {}
    sm = Decimal(df["Market Cap"].sum())
    for ticker, mc in zip(df["Ticker"], df["Market Cap"]):
        dic[ticker] = Decimal(mc) / sm
    return dic


def get_day_prices(df):
    dic = {}
    for ticker, price in zip(df["Ticker"], df["Price"]):
        dic[ticker] = Decimal(price)
    return dic


def get_current_nominal(portfolio, prices):
    sm = 0
    for ticker in portfolio.keys():
        sm += prices.get(ticker, 0) * portfolio[ticker]
    return sm


def reajust_portfolio(old_portfolio, nominal, weights, prices, fee):
    ideal_portfolio = initialize_portfolio(nominal, weights, prices, 0)
    portfolio = {}
    buys = []
    remaining_nominal = 0
    sm = 0
    for ticker in set(old_portfolio.keys()).union(set(ideal_portfolio.keys())):
        count_old = old_portfolio.get(ticker, 0)
        count = ideal_portfolio.get(ticker, 0)
        if count_old > count:
            remaining_nominal += (count_old - count) * \
                prices.get(ticker, 0) * (1 - fee)
            if count > 0:
                portfolio[ticker] = count
        elif count_old < count:
            buys.append(ticker)
            sm += weights[ticker]
        else:
            portfolio[ticker] = count_old
    for ticker in buys:
        weight = weights[ticker] / sm
        amount_usd = weight * remaining_nominal * (1 - fee)
        portfolio[ticker] = old_portfolio.get(
            ticker, 0) + amount_usd / prices[ticker]

    return portfolio


def initialize_portfolio(nominal, weights, prices, fee):
    dic = {}
    for ticker in weights.keys():
        amount_usd = weights[ticker] * nominal * (1 - fee)
        dic[ticker] = amount_usd / prices[ticker]
    return dic


def run_simulation(
    df,
    nominal,
    reccuring,
    start_date,
    finish_date,
    heuristic,
    apply_log=False,
    fee=Decimal(0.004),
):
    old_portfolio = {}
    nominals = []
    hodl = []
    dates = []
    i = 0
    while start_date <= finish_date:
        # Set up
        dates.append(start_date)
        cr_date = start_date.strftime("%d/%m/%Y")
        today = df[df["Date"] == cr_date]
        weights = heuristic(today)
        prices = get_day_prices(today)

        # Compute new portofolio
        if i == 0:
            hodl.append(nominal)
            portfolio = initialize_portfolio(nominal, weights, prices, fee)
        else:
            hodl.append(hodl[-1] + reccuring)
            nominal = get_current_nominal(old_portfolio, prices)
            portfolio = reajust_portfolio(
                old_portfolio, nominal + reccuring, weights, prices, fee
            )
        nominals.append(nominal + reccuring)

        # Set up next
        old_portfolio = copy.deepcopy(portfolio)
        start_date = start_date + relativedelta(months=1)
        i += 1

    returns = [get_percentage(x, y) for (x, y) in zip(hodl, nominals)]
    if apply_log:
        returns = [x.log10() for x in returns]
    return dates, returns


def get_percentage(base, current):
    if current > base:
        return (current / base - 1) * 100
    return -(1 - current / base) * 100


def compare_heuristics(df, nominal, recurring, heuristics):
    start_date = START_DATE
    results = {}
    i = 0
    while start_date <= FINISH_DATE - relativedelta(months=6):
        i += 1
        for heuristic in heuristics:
            if heuristic.__name__ not in results:
                results[heuristic.__name__] = []
            _, returns = run_simulation(
                df, nominal, recurring, start_date, FINISH_DATE, heuristic
            )
            for x in returns[6:]:
                results[heuristic.__name__].append(x)
        start_date += relativedelta(months=1)
        if i % 12 == 0:
            print("Finished year", start_date, "...")
    print("-" * 50)
    for heuristic in heuristics:
        returns = results[heuristic.__name__]
        print("For heuristic", heuristic.__name__,
              "we got the following results")
        print("Median:", np.median(returns), "%")
        print("Mean:", np.mean(returns), "%")
        print("STD:", np.std(returns), "%")
        print("Best return:", np.max(returns), "%")
        print("Worst return:", np.min(returns), "%")
        print("-" * 50)
    fig, axs = plt.subplots(3, 3)
    for i, date_range in enumerate(
        [
            (datetime(2014, 1, 1), datetime(2022, 1, 1)),
            (datetime(2017, 1, 1), datetime(2022, 1, 1)),
            (datetime(2018, 1, 1), datetime(2022, 1, 1)),
            (datetime(2014, 1, 1), datetime(2020, 1, 1)),
            (datetime(2017, 1, 1), datetime(2020, 1, 1)),
            (datetime(2018, 1, 1), datetime(2020, 1, 1)),
            (datetime(2014, 1, 1), datetime(2018, 1, 1)),
            (datetime(2017, 1, 1), datetime(2018, 1, 1)),
            (datetime(2014, 1, 1), datetime(2017, 1, 1)),
        ]
    ):
        x = i // 3
        y = i % 3
        for heuristic in heuristics:
            dates, returns = run_simulation(
                df,
                nominal,
                recurring,
                date_range[0],
                date_range[1],
                heuristic,
            )
            if i == 0:
                axs[x, y].plot(dates, returns, label=heuristic.__name__)
            else:
                axs[x, y].plot(dates, returns)
    fig.legend(loc="upper left")
    plt.show()


def drop_invalid_tickers(df):
    bad_tickers = []
    for ticker in df["Ticker"].unique():
        prices = df[df["Ticker"] == ticker]["Price"]
        bad = False
        for prev_price, price in zip(prices[:-1], prices[1:]):
            diff = math.fabs(prev_price - price)
            if diff / prev_price > 500:
                bad = True
                break
        if bad:
            bad_tickers.append(ticker)
    return df[~df["Ticker"].isin(bad_tickers)]


def heuristic_top_1(df):
    return get_day_weights(df, 1)


def heuristic_top_10(df):
    return get_day_weights(df, 10)


def heuristic_top_10_skip_10(df):
    return get_day_weights(df, 10, skip_first=10)


def heuristic_top_10_no_btc(df):
    return get_day_weights(df, 10, skip_first=1)


def heuristic_top_20_skip_10(df):
    return get_day_weights(df, 20, skip_first=10)


def heuristic_top_20_no_btc(df):
    return get_day_weights(df, 20, skip_first=1)


def heuristic_top_15_skip_10(df):
    return get_day_weights(df, 15, skip_first=10)


df = pd.read_csv("crypto_data_filtered.csv")
df = drop_invalid_tickers(df)

nominal = Decimal(10000)
recurring = Decimal(0)
# Compare all of the heuristics defined above
compare_heuristics(df, nominal, recurring, [
    heuristic_top_1,
    heuristic_top_10,
    heuristic_top_10_skip_10,
    heuristic_top_10_no_btc,
    heuristic_top_20_skip_10,
    heuristic_top_20_no_btc,
    heuristic_top_15_skip_10,
])
