

from scipy.stats import norm
import numpy as np

class OptionsHedgingPnl:

    def __init__(self, dta, r, q):
        self.dta = dta
        self.r = r
        self.q = q

    def option_price(self):
        d1 = (np.log(self.s/self.k) + (self.r - self.q + self.sigma**2/2)*self.tau)/(self.sigma*self.tau)
        d2 = d1 - self.sigma*np.sqrt(self.tau)
        return self.s*np.exp(-self.q*self.tau)*norm.cdf(d1) - self.k * np.exp(-self.r*self.tau)*norm.cdf(d2)


    def delta(self):
        d1 = (np.log(self.s/self.k) + (self.r - self.q + self.sigma**2/2)*self.tau)/(self.sigma*self.tau)
        return np.exp(-self.q*self.tau) * norm.cdf(d1)


    def vega(self):
        d1 = (np.log(self.s/self.k) + (self.r - self.q + self.sigma**2/2)*self.tau)/(self.sigma*self.tau)
        return self.s*np.exp(-self.q*self.tau)*norm.pdf(d1)*np.sqrt(self.tau)


    def duration_data_selector(self):

        if self.duration == "1M":
            new_dta = self.dta[self.dta["1M Maturity"]>=0]
            return new_dta.index, new_dta["Close"].values, new_dta["1M Maturity"].values, new_dta["V1M"].values
        elif self.duration == "3M":
            new_dta = self.dta[self.dta["3M Maturity"]>=0]
            return new_dta.index, new_dta["Close"].values, new_dta["3M Maturity"].values, new_dta["V3M"].values
        elif self.duration == "1W":
            new_dta = self.dta[self.dta["1W Maturity"]>=0]
            return new_dta.index, new_dta["Close"].values, new_dta["1W Maturity"].values, new_dta["V1W"].values


class OptionHedgingPnl_1W(OptionsHedgingPnl):

    def __init__(self, dta, r, q):
        super().__init__(dta, r, q)
        self.duration = "1W"


class OptionHedgingPnl_1M(OptionsHedgingPnl):

    def __init__(self, dta, r, q):
        super().__init__(dta, r, q)
        self.duration = "1M"


class OptionHedgingPnl_3M(OptionsHedgingPnl):

    def __init__(self, dta, r, q):
        super().__init__(dta, r, q)
        self.duration = "3M"
