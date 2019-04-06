
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

from functions_def import intra_day_volatility


class OptionsHedgingPnl:

    def __init__(self, dta, r, q):
        self.dta = dta
        self.r = r
        self.q = q

    def option_price(self):
        d1 = (np.log(self.s[:-1]/self.k) + (self.r - self.q + self.sigma[:-1]**2/2)*self.tau[:-1])/(self.sigma[:-1]*self.tau[:-1])
        d2 = d1 - self.sigma[:-1]*np.sqrt(self.tau[:-1])
        return self.s[:-1]*np.exp(-self.q*self.tau[:-1])*norm.cdf(d1) - self.k * np.exp(-self.r*self.tau[:-1])*norm.cdf(d2)

    def delta(self):
        d1 = (np.log(self.s[:-1]/self.k) + (self.r - self.q + self.sigma[:-1]**2/2)*self.tau[:-1])/(self.sigma[:-1]*self.tau[:-1])
        return np.exp(-self.q*self.tau[:-1]) * norm.cdf(d1)

    def vega(self):
        d1 = (np.log(self.s[:-1]/self.k) + (self.r - self.q + self.sigma[:-1]**2/2)*self.tau[:-1])/(self.sigma[:-1]*self.tau[:-1])
        return self.s[:-1]*np.exp(-self.q*self.tau[:-1])*norm.pdf(d1)*np.sqrt(self.tau[:-1])

    def duration_data_selector(self):
        if self.duration == "1M":
            new_dta = self.dta[self.dta["1M Maturity"]>=0]
            return new_dta, new_dta.index, new_dta["Close"].values, new_dta["1M Maturity"].values, new_dta["V1M"].values
        elif self.duration == "3M":
            new_dta = self.dta[self.dta["3M Maturity"]>=0]
            return new_dta, new_dta.index, new_dta["Close"].values, new_dta["3M Maturity"].values, new_dta["V3M"].values
        elif self.duration == "1W":
            new_dta = self.dta[self.dta["1W Maturity"]>=0]
            return new_dta, new_dta.index, new_dta["Close"].values, new_dta["1W Maturity"].values, new_dta["V1W"].values

    def hedging_pnl(self):
        optionPV = self.option_price()
        repl_port = np.zeros((len(self.tau), 1))
        total_port = np.zeros((len(self.tau), 1))
        bankacc = np.zeros((len(self.tau), 1))
        delta_vec = self.delta()

        stock_pos = delta_vec[1:] * self.s[1:-1]
        stock_pos_diff = stock_pos - delta_vec[:-1] * self.s[1:-1]
        bankacc[0] = optionPV[0] - delta_vec[0]*self.s[0]
        repl_port[0] = optionPV[0]
        dt = np.diff(self.tau)

        for t in range(len(dt)-1):
            bankacc[t+1] = bankacc[t]*(1 + dt[t]*self.r) - stock_pos_diff[t] *(1-dt[t]*self.q)
            repl_port[t+1] = stock_pos[t] + bankacc[t+1]
            total_port[t+1] = optionPV[t+1] - repl_port[t+1] - bankacc[0]*(1 + self.r*dt[0])**self.tau[t+1]

        optionPV =np.append(optionPV, np.max(self.s[-1] - self.k, 0))
        bankacc[-1] = bankacc[-2]*(1+dt[-1]*self.r)
        repl_port[-1] = delta_vec[-1]*self.s[-1] + bankacc[-1]
        total_port[-1] = optionPV[-1] - repl_port[-1] - bankacc[0]* (1+ self.r*dt[0])**self.tau[-1]

        return optionPV, repl_port, total_port


class OptionHedgingPnl1W(OptionsHedgingPnl):

    def __init__(self, dta, r, q):
        super().__init__(dta, r, q)
        self.duration = "1W"
        self.dta, self.date, self.s, self.tau, self.sigma = self.duration_data_selector()
        self.k = self.s[0]


class OptionHedgingPnl1M(OptionsHedgingPnl):

    def __init__(self, dta, r, q):
        super().__init__(dta, r, q)
        self.duration = "1M"
        self.dta, self.date, self.s, self.tau, self.sigma = self.duration_data_selector()
        self.k = self.s[0]


class OptionHedgingPnl3M(OptionsHedgingPnl):

    def __init__(self, dta, r, q):
        super().__init__(dta, r, q)
        self.duration = "3M"
        self.dta, self.date, self.s, self.tau, self.sigma = self.duration_data_selector()
        self.k = self.s[0]


if __name__ == "__main__":
    df = pd.read_excel("USDBRL_PriceHist.xlsx", index_col=0)
    option3m_analysis = OptionHedgingPnl3M(df, 0.065, 0.02)
    # op_pv, repl_pv, total_pv = option1m_analysis.hedging_pnl()
    # plt.plot(option1m_analysis.date, op_pv)
    date_range = option3m_analysis.date.normalize().unique()
    intra_vol_vec = [intra_day_volatility(option3m_analysis.dta, option3m_analysis.date, x) for x in date_range]
    plt.plot(intra_vol_vec)
    plt.show()


