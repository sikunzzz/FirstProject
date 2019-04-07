from functions_def import *
import matplotlib.pyplot as plt


# Use factory pattern design to analyse hedging pnl of option

class OptionsHedgingData:
    """ Base class to for selection data """

    def __init__(self, r, q):
        """ Initialise member of foreign and base currency interest rate"""
        self.r = r
        self.q = q


class OptionHedgingData1W(OptionsHedgingData):
    """ Subclass for selecting 1W data"""

    def __init__(self, dta, r, q):
        """ Initialise using the base class, and further initialisatin"""
        super().__init__(r, q)
        self.duration = "1W"
        self.dta = dta
        self.date, self.s, self.tau, self.sigma = dta.index, dta["Close"].values, dta["1W Maturity"].values, dta["V1W"].values
        self.k = self.s[0]


class OptionHedgingData1M(OptionsHedgingData):
    """ Subclass for selecting 1M data"""

    def __init__(self, dta, r, q):
        """ Initialise using the base class, and further initialisatin"""
        super().__init__(r, q)
        self.duration = "1M"
        self.dta = dta
        self.date, self.s, self.tau, self.sigma = dta.index, dta["Close"].values, dta["1M Maturity"].values, dta["V1M"].values
        self.k = self.s[0]


class OptionHedgingData3M(OptionsHedgingData):
    """ Subclass for selecting 3M data"""

    def __init__(self, dta, r, q):
        """ Initialise using the base class, and further initialisatin """
        super().__init__(r, q)
        self.duration = "3M"
        self.dta = dta
        self.date, self.s, self.tau, self.sigma = dta.index, dta["Close"].values, dta["3M Maturity"].values, dta["V3M"].values
        self.k = self.s[0]


class Analysis:
    """
    Analysis class contains functions to plot for results using different dynamic hedging methods
    """

    def hedging_pnl_plot(self, option, pv_option):
        """

        :param option: one of subclasses of OptionHedgingData
        :param pv_option: integer, range from 0 to 2, representing
        option price path, replicating portfolio value path, total portfolio value path respectively
        :return: graphs for dynamic hedging pnl
        """
        results = hedging_pnl(option.s, option.tau, option.sigma, option.k, option.r, option.q)
        plt.plot(option.date, results[pv_option])
        plt.xlabel("Date")
        plt.ylabel("Path")
        plt.title("Value Path for plain dynamic hedging")
        plt.show()

    def hedging_pnl_band_plot(self, option, pv_option, func):
        """

        :param option: one of subclasses of OptionHedgingData
        :param pv_option: integer, range from 0 to 2, representing
        option price path, replicating portfolio value path, total portfolio value path respectively
        :param func: delta/vega functions
        :return: graphs for dynamic hedging pnl using delta/vega bound
        """
        if pv_option == 0:
            return self.hedging_pnl_plot(option, pv_option)
        else:
            del_vec = func(option.s[:-1], option.tau[:-1], option.sigma[:-1], option.k, option.r, option.q)
            diff_vec = np.diff(del_vec)
            min_del, max_del = np.abs(np.min(diff_vec)), np.max(np.cumsum(diff_vec))
            bound = np.linspace(min_del, max_del, num=1000)
            final_pnl = [
                hedging_pnl_delta_band(option.s, option.tau, option.sigma, option.k, option.r, option.q, x, func)[
                    pv_option][-1] for x in bound]
            plt.plot(bound, final_pnl)
            plt.xlabel("Bound")
            plt.ylabel("Pnl at Maturity")
            plt.title("Pnl at maturity for various hedging bound")
            plt.show()

    def intraday_vol_plot(self, option):
        """

        :param option: one of subclasses of OptionHedgingData
        :return: graph for intra day volatility
        """
        date_range = option.date.normalize().unique()
        intra_vol_vec = [intra_day_volatility(option.dta, option.date, x) for x in date_range]
        plt.plot(date_range, intra_vol_vec)
        plt.xlabel("Date")
        plt.ylabel("Intra-day Volatility")
        plt.title("Intra-day Volatility for the duration of the option")
        plt.show()


class AnalysisFactory:
    """ Factory of Analysis class to select which method to use"""

    def get_pnl_method(self, option, pv_option, method):
        """
        :param option: one of subclasses of OptionHedgingData
        :param pv_option: integer, range from 0 to 2, representing
        :param method: 'plain', 'delta', 'vega', 'day_vol'
        :return: function in Analysis class
        """
        analysis = Analysis()
        if method == "plain":
            return analysis.hedging_pnl_plot(option, pv_option)
        elif method == "delta" or method == "vega":
            return analysis.hedging_pnl_band_plot(option, pv_option, eval(method))
        elif method == "day_vol":
            return analysis.intraday_vol_plot(option)
        else:
            ValueError(method)


class OptionFactory:
    """ Factory of OptionHedging subclass to select which duration to use"""

    def get_option(self, dta, duration, r, q):
        """

        :param dta: pandas data frame
        :param duration: '1W'. '1M', '3M'
        :param r: interest rate for foreign currency
        :param q: interest rate for base currency
        :return: subclass of OptionHedgingData
        """
        if duration == "1W":
            new_dta = dta[dta["1W Maturity"] >= 0]
            return OptionHedgingData1W(new_dta, r, q)
        elif duration == "1M":
            new_dta = dta[dta["1M Maturity"] >= 0]
            return OptionHedgingData1M(new_dta, r, q)
        elif duration == "3M":
            new_dta = dta[dta["3M Maturity"] >= 0]
            return OptionHedgingData3M(new_dta, r, q)
        else:
            ValueError(duration)


class OptionInitializeAndAnalysis:
    """ The final control that select from OptionFactory and AnalysisFactory"""

    def initialise(self, dta, duration, r, q, pv_option, method="plain"):
        """
        :param dta: pandas data frame
        :param duration: '1W'. '1M', '3M'
        :param r: interest rate for foreign currency
        :param q: interest rate for base currency
        :param pv_option: integer, range from 0 to 2, representing
        :param method: 'plain', 'delta', 'vega', 'day_vol'
        :return: function in Analysis
        """
        optionfac = OptionFactory()
        option = optionfac.get_option(dta, duration, r, q)
        analysisfac = AnalysisFactory()
        return analysisfac.get_pnl_method(option, pv_option, method)


if __name__ == "__main__":
    df = pd.read_excel("USDBRL_PriceHist.xlsx", index_col=0)
    trial_analysis = OptionInitializeAndAnalysis()
    trial_analysis.initialise(df, "1M", 0.065, 0.02, 2, "delta")
    trial_analysis.initialise(df, "1M", 0.065, 0.02, 2, "day_vol")