
The project used USD/BRL tick data from 2018/05/10 to 2018/08/10 to analyse dynamic hedging pnl for 1W, 1M, 3M options.

It consists of two files: functions_def.py and class_def.py.

(1) functions_def.py defines functions to calculate option price, delta, vega, dynamic hedging pnl for every time step and dynamic hedging
pnl using delta and vega bound. It also defines functions to select daily data and calculate intra-day volatility

(2) functions defined functions_def.py are then used in class_def.py. 

(3) Several classes are defined in class_def.py.  OptionInitializeAndAnalysis is at the top of factory pattern design, it uses the two factories
OptionFactory and AnalysisFactory. OptionFactory determines which options (1W, 1M, 3M) to initialize, while AnalysisFactory determines which
delta-hedging method to use. Then different subclasses of OptionHedgingData are defined and different functions in Analysis are defined.

(4) Finally, by running two examples in class_def.py, it will produce two graphs.

(5) Some class structure in class_def.py may be redundant and improvement can be made further in the future.
