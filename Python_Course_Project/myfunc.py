
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def dataselector():
    return


def option_price(s, tau, k, sigma, r, q):
    d1 = (np.log(s/k) + (r - q + sigma**2/2)*tau)/(sigma*tau)
    d2 = d1 - sigma*np.sqrt(tau)
    return s*np.exp(-q*tau)*norm.cdf(d1) - k * np.exp(-r*tau)*norm.cdf(d2)


def delta(s, tau, k, sigma, r, q):
    d1 = (np.log(s/k) + (r - q + sigma**2/2)*tau)/(sigma*tau)
    return np.exp(-q*tau) * norm.cdf(d1)


def hedging_pnl(s, tau, k, sigma, r, q):

    optionPV = option_price(s[:-1],tau[:-1],k,sigma,r,q)
    repl_port = np.zeros(len(tau), 1)
    total_port = np.zeros(len(tau), 1)
    bankacc = np.zeros(len(tau), 1)
    delta_vec = delta(s[:-1], tau[:-1], k, sigma[:-1], r, q)

    stock_pos = delta_vec[1:] * s[1:-1]
    stock_pos_diff = stock_pos - delta_vec[:-1] * s[1:-1]
    bankacc[0] = optionPV[0] - delta_vec[0]*s[0]
    repl_port[0] = optionPV[0]
    dt = np.diff(tau)

    for t in len(dt)-1:
        bankacc[t+1] = bankacc[t]*(1 + dt[t]*r) - stock_pos_diff[t] *(1-dt[t]*q)
        repl_port[t+1] = stock_pos[t] + bankacc[t+1]
        total_port[t+1] = optionPV[t+1] - repl_port[t+1] - bankacc[0]*(1 + r*dt[0])**tau[t+1]

    optionPV.append(np.maximum(s[-1] - k, 0))
    bankacc[-1] = bankacc[-2]*(1+dt[-1]*r)
    repl_port[-1] = delta_vec[-1]*s[-1] + bankacc[-1]
    total_port[-1] = optionPV[-1] = repl_port[-1] - bankacc[0]* (1+ r*dt[0])**tau[-1]

    return optionPV, repl_port, total_port


def hedging_pnl_delta_band:
    return


def hedging_pnl_vega_band:
    return


