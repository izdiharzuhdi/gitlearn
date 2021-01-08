import investpy
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from pandas import ExcelWriter
import openpyxl
import xlsxwriter
import talib
import numpy as np
from os.path import dirname
from trendline import get_support_resistance, get_extrema
import matplotlib.pyplot as plt

class stock():
    def __init__(self, ticker, country):
        self.ticker = ticker
        self.country = country
        self.stockdata = None
        self.acceptable = True

    def get_gradient(self, param):
        '''
        Gradient is the difference for values between 2 consecutive days
        '''
        return self.stockdata[param].diff()

    def get_indicators(self):
        data = self.stockdata
        # Get MACD
        data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data['Close'], fastperiod=3, slowperiod=18, signalperiod=4)
        # data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(data['Close'])
        # data["macd_d/dt"] = np.gradient(data['macd']) / data['Close']  * 100 
        # data["macd_d2/dt2"] = np.gradient(data['macd_d/dt'])
        data["macd_d/dt"] = self.get_gradient('macd_hist')
        data["macd_d2/dt2"] = self.get_gradient('macd_d/dt')
        
        # Get MA10 and MA30
        data["ma7"] = talib.MA(data["Close"], timeperiod=7)
        data["ma26"] = talib.MA(data["Close"], timeperiod=26)
    
        # Get Bollinger Bands
        data["upperband"], data["middleband"], data["lowerband"] = talib.BBANDS(data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        
        # Get RSI
        data["rsi"] = talib.RSI(data["Close"])
    
        # Get STOCHASTIC RSI
        # data["fastk"], data["fastd"]  = talib.STOCHRSI(data["Close"])
    
        # Get STOCHASTIC
        data["slowk"], data["slowd"]  = talib.STOCH(data["High"], data["Low"], data["Close"], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        # data["slowk_d/dt"] = self.get_gradient('slowk')
        # data["slowk_d2/dt2"] = self.get_gradient('slowk_d/dt')
        data["stoch"] = data["slowk"] - data["slowd"]
        data["stoch-v"] = self.get_gradient('stoch')

        # Get Chaikin Volume Oscillator
        '''
        A Chaikin Oscillator reading above zero indicates net buying pressure, while one below zero registers net selling pressure. 
        Divergence between the indicator and pure price moves are the most common signals from the indicator, and often flag market 
        turning points.
        '''
        data["ck_AD"] = talib.AD(data["High"], data["Low"], data["Close"], data["Volume"])
        '''
        When OBV is rising, it shows that buyers are willing to step in and push the price higher. 
        When OBV is falling, the selling volume is outpacing buying volume, which indicates lower prices. 
        In this way, it acts like a trend confirmation tool. If price and OBV are rising, that helps indicate a continuation of the trend.
        '''
        data["OBV"] = talib.OBV(data["Close"], data["Volume"])

        '''
        Positive ZC marks bullish
        '''
        data["ZC"] = (data["ma7"] - data["ma26"]) / data["Close"]
        data["ZC_d/dt"] = self.get_gradient('ZC') * 10
        # data["ZC_d2/dt2"] = self.get_gradient('ZC_d/dt')
        data["ZC_d2/dt2"] = talib.MA(data["ZC_d/dt"], timeperiod=5)
    
        self.stockdata = data


    def screener_macd(self, data, days, monotonicdays=4):
        '''
        1) MACD crosses zero
        2) Signal and MACD needs to be on the positive region, signal in negative region means the bullish is short living (not necessarily)
        3) Avoid long candle
        4) Avoid flat candle
        5) Avoid blip MACD cross
        return last_index - days <= zero_crossings[-1] and data['macd_hist'][-monotonicdays:].is_monotonic
        '''
        np_macd = np.array(data['macd_hist'])
        zero_crossings = np.where(np.diff(np.sign(np_macd)))[0]
        last_index = len(data.index) - 1
        '''
        last_index - days <= zero_crossings[-1] is a logic to detect whether there has been any sign change
        on the watched parameter for the last nth days, with n starts from the last date in the data
        For example, if days=1, this expression will give true if there has been sign change from yesterday to today.
        '''
        # Return the dataframe if MACD crosses the x-axis with positive gradient
        return last_index - days <= zero_crossings[-1] and data['macd_hist'][-monotonicdays:].is_monotonic
        # return data['macd_d/dt'][-monotonicdays:].is_monotonic
    
    def screener_zc_v(self, data, days=4, monotonicdays=2):
        '''
        ZC deflection
        '''
        np_stoch = np.array(data['ZC_d/dt'])
        zero_crossings = np.where(np.diff(np.sign(np_stoch)))[0]
        last_index = len(data.index) - 1
        # return last_index - days <= zero_crossings[-1] and data['slowk'][-monotonicdays:].is_monotonic
        return last_index - days <= zero_crossings[-1] and data['ZC'][-monotonicdays:].is_monotonic
    
    def screener_stochastic(self, data):
        '''
        Common triggers occur when the %K line drops below 20—the stock is considered oversold, 
        and it is a buying signal.
        If the %K peaks just below 100 and heads downward, the stock should be sold before that 
        value drops below 80.
        Generally, if the %K value rises above the %D, then a buy signal is indicated by this 
        crossover, provided the values are under 80. If they are above this value, the security 
        is considered overbought.
        \nreturn data['slowk'][-1] > data['slowd'][-1] and data['slowk'][-1] < 30
        '''
        # if data['slowk'][-1] < 20:
        #     return True
        # elif data['slowk'][-1] > data['slowd'][-1] and data['slowk'][-1] < 80:
        #     return True
        # else:
        #     return False
        return data['slowk'][-1] > data['slowd'][-1] and data['slowk'][-1] < 40
    
    def screener_stochastic_gradient(self, data, days, monotonicdays=4):
        '''
        return stoch-v crosses 0
        return last_index - days <= zero_crossings[-1]
        '''
        np_stoch = np.array(data['stoch-v'])
        zero_crossings = np.where(np.diff(np.sign(np_stoch)))[0]
        last_index = len(data.index) - 1
        # return last_index - days <= zero_crossings[-1] and data['slowk'][-monotonicdays:].is_monotonic
        return last_index - days <= zero_crossings[-1]


    def get_rawdata(self, from_date, to_date):
        '''
        Method to query stock price data from investing.com
        '''
        self.stockdata = investpy.stocks.get_stock_historical_data(self.ticker, self.country, from_date, to_date, as_json=False, order='ascending', interval='Daily')
    

    def run_filter(self):
        '''
        Filter idea:
        slowk > slowd, below 20
        ZC positive
        If ZC negative but near zero, check ZC-v
        '''
        stockdata = self.stockdata
        # self.get_indicators()
        # stockdata = self.get_support_resistance_trendlines(stockdata)
        # self.acceptable &= self.screener_izd(stockdata, 1, 2) # Norway
        # self.acceptable &= self.screener_izd_volxmacd(stockdata)
        # self.acceptable &= self.screener_macd(stockdata, 1, monotonicdays=3) # Malaysia
        self.acceptable &= self.screener_stochastic_gradient(stockdata, 5, monotonicdays=3)
        # self.acceptable &= self.screener_zc(stockdata, 2)
        # self.acceptable &= self.screener_zc_v(stockdata, 1, 2)
        self.acceptable &= self.screener_stochastic(stockdata)

    def slicedata(self, data):
        stockdata = self.stockdata.iloc[-days:]
        stockdata.reset_index(inplace=True)
        self.stockdata = stockdata

    def dataoutput(self):
        return [self.acceptable, self.stockdata]


two_yr_ago = datetime.today()-relativedelta(years=2)
days = 180
baselocation = '/'.join([dirname(__file__), 'Server', 'daily'])
# baselocation = '/'.join([dirname(__file__), 'Server', 'backtest'])

fro = int(two_yr_ago.timestamp())
to_date_str = datetime.today().strftime("%d%m%Y")
from_date = datetime.fromtimestamp(fro).strftime("%d/%m/%Y")
to_date = datetime.today().strftime("%d/%m/%Y")
# to_date = '09/09/2020'
country = 'Norway'
# country = 'Malaysia'
# # Get list of all ticker for country
counterlist = investpy.stocks.get_stocks(country)
excelfilename = '_'.join([country, str(to_date_str) + '.xlsx'])

# Get list for all holdings
# keywords = [
#     'Solstad', 
#     'Interoil',
#     'Aker Solutions OL',
#     'Sapura',
#     # 'Adevinta',
#     ]
# search_results = list()
# for keyword in keywords:
#     search_results.append(investpy.stocks.search_stocks('full_name', keyword))
# counterlist = pd.concat(search_results).reset_index()
# excelfilename = '_'.join(['Holding', str(to_date_str) + '.xlsx'])

# For testing 
# keywords = ['Marine Harvest', 'Solstad']
# search_results = list()
# for keyword in keywords:
#     search_results.append(investpy.stocks.search_stocks('full_name', keyword))
# counterlist = pd.concat(search_results).reset_index()
# to_date = '30/01/2020'
# to_date_str = ''.join(to_date.split('/'))
# excelfilename = '_'.join(['Test', str(to_date_str) + '.xlsx'])

# counterlist = counterlist[:200]
# print(counterlist)

excelfilename = '/'.join([baselocation, excelfilename])
droplist = []
# # Get raw stock data
with ExcelWriter(excelfilename, engine='xlsxwriter') as writer:
    for i, ticker in enumerate(tqdm(counterlist['symbol'])):
        try:
            __stock = stock(ticker, counterlist.iloc[i]['country'])
            __stock.get_rawdata(from_date, to_date)
            # __stock.get_support_resistance()
            __stock.get_indicators()
            # __stock.get_candle_pattern()
            # __stock.get_extrema()
            __stock.run_filter()
            __stock.slicedata(days)
            processedData = __stock.dataoutput()
            # Exclude the ticker which doesnt fullfil filter criteria from stored dat in Excel
            if processedData[0] == True:
                processedData[1].to_excel(writer, sheet_name=ticker)
            if processedData[0] == False:
                droplist.append(i)
        except Exception as e:
            droplist.append(i)
            print('Error for {a}\n{b}'.format(a=ticker, b=e))
    # Exclude the ticker which doesnt fullfil filter criteria from summary list
    filtered_counter = counterlist.drop(droplist)
    filtered_counter.to_excel(writer, sheet_name='list')
    tqdm.write('{0} created'.format(excelfilename))
    writer.save()
print(filtered_counter)




