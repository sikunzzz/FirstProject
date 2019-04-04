

def get_data_decorater(func):

    def func_wrapper(dta, duration):
        return func(dta, duration)

    return func_wrapper


def get_pnl_dcorater(func1):

    def func_wrapper(s, tau, sigma, k,r,q, func2)
        reutrn func1(s, tau, sigma, k,r,q, func2)

    return func_wrapper

class my_decorater:

    def __init__(self, f, g):

