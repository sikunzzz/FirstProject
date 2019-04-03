
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm

def daily_data_selector(dta, dates, start_date):
    return dta[np.logical_and(dates.month == start_date.month, dates.day == start_date.day)]


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


def hedging_pnl(s, tau, sigma,k, r, q):

    optionPV = option_price(s[:-1],tau[:-1],k,sigma[:-1],r,q)
    repl_port = np.zeros((len(tau), 1))
    total_port = np.zeros((len(tau), 1))
    bankacc = np.zeros((len(tau), 1))
    delta_vec = delta(s[:-1], tau[:-1], k, sigma[:-1], r, q)

    stock_pos = delta_vec[1:] * s[1:-1]
    stock_pos_diff = stock_pos - delta_vec[:-1] * s[1:-1]
    bankacc[0] = optionPV[0] - delta_vec[0]*s[0]
    repl_port[0] = optionPV[0]
    dt = np.diff(tau)

    for t in range(len(dt)-1):
        bankacc[t+1] = bankacc[t]*(1 + dt[t]*r) - stock_pos_diff[t] *(1-dt[t]*q)
        repl_port[t+1] = stock_pos[t] + bankacc[t+1]
        total_port[t+1] = optionPV[t+1] - repl_port[t+1] - bankacc[0]*(1 + r*dt[0])**tau[t+1]

    np.append(optionPV, np.max(s[-1] - k, 0))
    bankacc[-1] = bankacc[-2]*(1+dt[-1]*r)
    repl_port[-1] = delta_vec[-1]*s[-1] + bankacc[-1]
    total_port[-1] = optionPV[-1] = repl_port[-1] - bankacc[0]* (1+ r*dt[0])**tau[-1]

    return optionPV, repl_port, total_port



def hedging_pnl_delta_band( s, tau, sigma, k, r, q, bound ):

    return


def hedging_pnl_vega_band:
    return


