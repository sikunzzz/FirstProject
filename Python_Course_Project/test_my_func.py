
import pandas as pd
dta = pd.read_excel("USDBRL_PriceHist.xlsx", index_col=0)
new_dta = dta[dta["1W Maturity"] >= 0]



def test_option_len():
    assert len(new_dta.index) == 535


def test_time_zero():
    assert new_dta["1W Maturity"].any != 0