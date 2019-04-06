
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm


def daily_data_wrapper(func):
    def intra_day_volatility_wrapper(dta, Dates, start_date):
        new_dta = func(dta, Dates, start_date)
        close_p = new_dta["Close"].values
        return np.sqrt(252*8*60)*np.std(close_p[1:]/close_p[:-1] - 1)
    return intra_day_volatility_wrapper


@daily_data_wrapper
def intra_day_volatility(dta, Dates, start_date):
    return dta[np.logical_and(Dates.month == start_date.month, Dates.day == start_date.day)]


def duration_data_selector(dta, duration):

    if duration == "1M":
        new_dta = dta[dta["1M Maturity"]>=0]
        return new_dta.index, new_dta["Close"].values, new_dta["1M Maturity"].values, new_dta["V1M"].values
    elif duration == "3M":
        new_dta = dta[dta["3M Maturity"]>=0]
        return new_dta.index, new_dta["Close"].values, new_dta["3M Maturity"].values, new_dta["V3M"].values
    elif duration == "1W":
        new_dta = dta[dta["1W Maturity"]>=0]
        return new_dta.index, new_dta["Close"].values, new_dta["1W Maturity"].values, new_dta["V1W"].values


def option_price(s, tau, sigma, k, r, q):
    d1 = (np.log(s/k) + (r - q + sigma**2/2)*tau)/(sigma*tau)
    d2 = d1 - sigma*np.sqrt(tau)
    return s*np.exp(-q*tau)*norm.cdf(d1) - k * np.exp(-r*tau)*norm.cdf(d2)


def delta(s, tau, sigma, k, r, q):
    d1 = (np.log(s/k) + (r - q + sigma**2/2)*tau)/(sigma*tau)
    return np.exp(-q*tau) * norm.cdf(d1)


def vega(s, tau, sigma, k, r, q):
    d1 = (np.log(s/k) + (r - q + sigma**2/2)*tau)/(sigma*tau)
    return s*np.exp(-q*tau)*norm.pdf(d1)*np.sqrt(tau)


def hedging_pnl(s, tau, sigma, k, r, q):

    optionPV = option_price(s[:-1],tau[:-1], sigma[:-1],k, r,q)
    repl_port = np.zeros((len(tau), 1))
    total_port = np.zeros((len(tau), 1))
    bankacc = np.zeros((len(tau), 1))
    delta_vec = delta(s[:-1], tau[:-1], sigma[:-1], k, r, q)

    stock_pos = delta_vec[1:] * s[1:-1]
    stock_pos_diff = stock_pos - delta_vec[:-1] * s[1:-1]
    bankacc[0] = optionPV[0] - delta_vec[0]*s[0]
    repl_port[0] = optionPV[0]
    dt = np.diff(tau)

    for t in range(len(dt)-1):
        bankacc[t+1] = bankacc[t]*(1 + dt[t]*r) - stock_pos_diff[t] *(1-dt[t]*q)
        repl_port[t+1] = stock_pos[t] + bankacc[t+1]
        total_port[t+1] = optionPV[t+1] - repl_port[t+1] - bankacc[0]*(1 + r*dt[0])**tau[t+1]

    optionPV = np.append(optionPV, np.max(s[-1] - k, 0))
    bankacc[-1] = bankacc[-2]*(1+dt[-1]*r)
    repl_port[-1] = delta_vec[-1]*s[-1] + bankacc[-1]
    total_port[-1] = optionPV[-1] - repl_port[-1] - bankacc[0]* (1+ r*dt[0])**tau[-1]

    return optionPV, repl_port, total_port


def hedging_pnl_delta_band(s, tau, sigma, k, r, q, bound, func ):
    optionPV = option_price(s[:-1],tau[:-1],sigma[:-1], k, r,q)
    delta_vec = func(s[:-1], tau[:-1], sigma[:-1], k, r, q)

    repl_port = np.array([])
    total_port = np.array([])
    bankacc = np.array([])

    repl_port = np.append(repl_port, optionPV[0])
    total_port = np.append(total_port, 0)
    bankacc = np.append(bankacc, optionPV[0] - s[0]*delta_vec[0])
    delta_diff = np.diff(delta_vec)
    cumulate_delta = 0
    time_stamp = 0
    stock_index = 0
    dt = tau[1] - tau[0]

    for i, t in enumerate(tau[1:-1]):
        cumulate_delta += delta_diff[i]
        if np.abs(cumulate_delta) >= bound:
            stock_pos = (s[i+1] - s[stock_index]) * cumulate_delta
            bankacc = np.append(bankacc, bankacc[-1]*(1 + dt*r)**(t - time_stamp) - stock_pos*(1-dt*q)**(t - time_stamp))
            repl_port = np.append(repl_port, bankacc[-1] + delta_vec[i+1] * s[i+1])
            total_port = np.append(total_port,  optionPV[i+1] - repl_port[-1] - bankacc[0] * (1 + dt*r)**t)
            stock_index = i+1
            cumulate_delta = 0
            time_stamp = t

    stock_pos = s[-1] - s[stock_index] * cumulate_delta
    optionPV = np.append(optionPV, np.max(s[-1] - k, 0))
    bankacc = np.append(bankacc, bankacc[-1]*(1 + dt*r)**(t - time_stamp) - stock_pos*(1-dt*q)**(t - time_stamp))
    repl_port = np.append(repl_port, bankacc[-1] + delta_vec[-1] * s[-1])
    total_port = np.append(total_port, optionPV[-1] - repl_port[-1] - bankacc[0] * (1 + dt*r)**tau[-1])

    return optionPV, repl_port, total_port


if __name__ == "__main__":
    df = pd.read_excel("USDBRL_PriceHist.xlsx", index_col= 0)
    dates, s, tau, sigma = duration_data_selector(df, "1M")
    opPV, repPV, totalPV = hedging_pnl_delta_band(s, tau, sigma, s[0], 0.065, 0.02, 0.000001, delta)
