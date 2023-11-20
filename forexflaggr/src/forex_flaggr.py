"""
------------------------------------------------------------------------------------------------
forex_flagger.py

ForexFlagger class.

    - fetch data from yfinance
    - construct dataframe
    - compute (weighted) moving averages
    - generate plotly graph

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""
from .dependencies import (dt, yf, go, np)

class ForexFlaggr:
    def __init__(self) -> None:
        self.data = None

    @staticmethod
    def moving_average(signal, period=20):
        return signal.rolling(window=period).mean()
    
    @staticmethod
    def weighted_moving_average(signal, period=14):
        weights = np.linspace(0,1,period)
        sum_weights = np.sum(weights)
        return signal\
            .rolling(window=period)\
            .apply(lambda x: np.sum(weights*x)/sum_weights)
    
    def filter_by_date(self, start_date, end_date):
        return self.data[start_date:end_date]
    
    
    def fetch_data(self, stock="USDZAR=X", sample_interval='1h', n_days=500):
        self.title  = stock
        start_date  = dt.datetime.today() - dt.timedelta(n_days) 
        end_date    = dt.datetime.today()
        self.data   = yf.download(stock, start_date, end_date, interval=sample_interval)
        return self

    def join_df(self, ma, wma):
        self.ma         = ma
        self.wma        = wma
        self.wma.name   = 'wma'
        self.ma.name    = 'ma'
        self.df_all     = self.data.join(ma).join(wma)
    
    def plot_signal(self, MA_periods=None, title='USD/ZAR'):
        if self.data is None:
            return self

        if MA_periods is None: MA_periods = 24*14 # two weeks
        ma  = ForexFlaggr.moving_average(self.data.Close, MA_periods)
        wma = ForexFlaggr.weighted_moving_average(self.data.Close, MA_periods)

        fig = go.Figure()
        fig.add_trace(go.Line(x=self.data.index, y=self.data.Close, name=self.title))
        fig.add_trace(go.Line(x=self.data.index, y=ma,              name='Moving avg.'))
        fig.add_trace(go.Line(x=self.data.index, y=wma,             name='Weighted ma.'))
        fig.update_layout(template='none', title=self.title)
        self.join_df(ma, wma)
        self.fig = fig
        
        return self
