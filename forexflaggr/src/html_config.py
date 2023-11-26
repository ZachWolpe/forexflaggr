"""
------------------------------------------------------------------------------------------------
html_config.py

A personilised script to build an html documentation for general analysis.

: 26.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

from pygam import LinearGAM, s, f, te
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import forexflaggr as fxr
import pandas as pd
import numpy as np
import itertools
import warnings
import logging
import json
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class Build_HTML_Config:
    """
    Build HTML Config.

    A pre-configured script to build an html documentation for general analysis.
    It is likely that this script will be modified for each project, however it forms a good starting point as a general config. 

    Arguments:
        - _n_days (str):        number of days to fetch data for. Default is 500.
        - _MA_periods (int):    number of periods to use for moving average. Default is 12*14 (2 weeks).
        - _store_path (str):    path to store html files. Default is 'output'.
        - ticker_1 (str):       ticker 1. Default is 'USDZAR=X'.
        - ticker_2 (str):       ticker 2. Default is '^IRX'.
        - ticker_3 (str):       ticker 3. Default is '^GSPC'.
        - ticker_label_1 (str): ticker label 1. Default is 'USD/ZAR'.
        - ticker_label_2 (str): ticker label 2. Default is 'US-TBills'.
        - ticker_label_3 (str): ticker label 3. Default is 'S&P500'.   
    """

    def __init__(self,
                 _n_days:int=500, _MA_periods:int=None, _store_path:int='output',
                 ticker_1:str='USDZAR=X', ticker_2:str='^IRX', ticker_3:str='^GSPC',
                 ticker_label_1:str='USD/ZAR', ticker_label_2:str='US-TBills', ticker_label_3:str='S&P500'
                 ) -> None:
       
        """
        ------------------------------------------------------------------------------------------------
        1. SET GLOBAL VARIABLES
        ------------------------------------------------------------------------------------------------
        """

        def _default_setter(param, default):
            if param is None:
                return default
            return param
        
        self._n_days            = _default_setter(_n_days,      500)
        self._MA_periods        = _default_setter(_MA_periods,  int(12*14)) # 2 weeks (1 sample/hour * 14 days)
        self._store_path        = _default_setter(_store_path,  'output')
        self.ticker_1           = _default_setter(ticker_1,     'USDZAR=X')
        self.ticker_2           = _default_setter(ticker_2,     '^IRX')
        self.ticker_3           = _default_setter(ticker_3,     '^GSPC')
        self.ticker_label_1     = _default_setter(ticker_label_1, 'USD/ZAR')
        self.ticker_label_2     = _default_setter(ticker_label_2, 'US-TBills')
        self.ticker_label_3     = _default_setter(ticker_label_3, 'S&P500')

        logging.info('Build HTML Config initialised.')


        """
        ------------------------------------------------------------------------------------------------
        2. RUN FOREXFLAGGR

        Build all ForexFlaggr modules and produce plots. Save the plots to html.
        ------------------------------------------------------------------------------------------------
        """
        self.run()


        """
        ------------------------------------------------------------------------------------------------
        3. PRODUCE HTML DOCUMENTATION
        ------------------------------------------------------------------------------------------------
        """
        self.build_html_document()





    def run(self):
        """Run Build HTML Config."""
        self\
            .fetch_financial_data()\
            .extract_latest_price()\
            .build_pie_chart_recommendation()\
            .fit_loess_model()\
            .fit_gam_model()\
            .build_multidimensional_model()
        return self


    def fetch_financial_data(self):
        """Fetch financial data."""
        # default : USDZAR
        ff = fxr.ForexFlaggr()
        ff\
            .fetch_data(stock=self.ticker_1, n_days=self._n_days)\
            .plot_signal(MA_periods=self._MA_periods)

        # default : USTreasury Bills
        fi = fxr.ForexFlaggr()
        fi\
            .fetch_data(stock=self.ticker_2, n_days=self._n_days)\
            .plot_signal(MA_periods=self._MA_periods)

        # default : S&P500
        fs = fxr.ForexFlaggr()
        fs\
            .fetch_data(stock=self.ticker_3, n_days=self._n_days)\
            .plot_signal(MA_periods=self._MA_periods)

        # safety check
        assert ff.data.shape[0] > 0, 'No data found for USDZAR, try to clear the  yfinance cache. Consider running: \n      > `pip install yfinance --upgrade --no-cache-dir`.'

        # save plots as html
        ff.fig.write_html(f"{self._store_path}/usdzar.html")
        fi.fig.write_html(f"{self._store_path}/UStreasury.html")
        fs.fig.write_html(f"{self._store_path}/sp500.html")

        self.ff = ff
        self.fi = fi
        self.fs = fs

        logging.info('First set of plots saved to html.')     
        return self
    
    
    def extract_latest_price(self):
        """Extract latest price."""
        # Price data
        _print_data = fxr.ForexFlaggr.get_price(self.ff)
        # save as json
        with open(f"{self._store_path}/price_data.json", 'w') as f:
            json.dump(_print_data, f)
        _datetime, _timezone, _close, _open, _high, _low = _print_data
        # print('Price '_datetime, _timezone, _close, _open, _high, _low)

        self._datetime  = _datetime
        self._timezone  = _timezone
        self._close     = _close
        self._open      = _open
        self._high      = _high
        self._low       = _low

        logging.info('Price data saved to txt.')
        return self


    def build_pie_chart_recommendation(self, _n_samples=252):
        """Build pie chart recommendation."""
        # Price momentum
        pcr_fig = fxr.pie_chart_recommendation.plot_pie_recommendation(self.ff.df_all, n_samples=_n_samples, fig=go.Figure())
        pcr_fig.write_html(f"{self._store_path}/pie_chart_recommendation.html")

        logging.info('Pie momentum chart recommendation saved to html.')
        return self
    
    def fit_loess_model(self):
        """Fit LOESS model."""
        # GAMs + LOESS
        # extract data
        model_data  = self.ff.data.reset_index()
        X, y        = np.array(model_data.index), np.array(model_data.Close)

        # build LOESS model
        loess_model = fxr.LOESS_Model(X, y)
        loess_model\
            .build()\
            .plot_prediction()
        loess_model.fig.write_html(f"{self._store_path}/loess.html")

        logging.info('LOESS model saved to html.')
        return self


    def fit_gam_model(self):
        """Fit GAM model."""
        # GAMs + LOESS
        # extract data
        model_data  = self.ff.data.reset_index()
        X, y        = list(model_data.index), model_data.Close
        X, y        = np.reshape(X, (-1, 1)), np.reshape(y.values, (-1, 1))

        # build GAM model
        gam = fxr.GAM_Model(X, y)
        gam.build().plot_prediction()
        gam.fig.write_html(f"{self._store_path}/gam.html")

        logging.info('GAM model saved to html.')
        return self


    def build_multidimensional_model(self):
        """Build multidimensional model."""
        # Complex Model
        # 1. build dataframe
        def transform_df(ff, col_name='USDZAR'):
            df = ff.df_all.copy()
            df.index = df.index.strftime('%Y-%m-%d').copy()
            df = df.groupby(df.index).mean()
            df = df[['Close']]
            df.columns = [col_name]
            return df

        df = transform_df(self.ff, self.ticker_label_1).join(transform_df(self.fi, self.ticker_label_2)).join(transform_df(self.fs, self.ticker_label_3)).reset_index()


        # 2. plot 3d scatter plot
        fig = px.scatter_3d(df, x='Datetime', y=self.ticker_label_2, z=self.ticker_label_1, color=self.ticker_label_3)
        fig.update_layout(template='plotly_dark', title=f'{self.ticker_label_1} ~ (Datetime, {self.ticker_label_2}, {self.ticker_label_3})')
        fig.write_html(f"{self._store_path}/3d_scatter.html")

        logging.info('3D scatter plot saved to html.')

        # 3. fit gam: note use index in place of date as a numeric
        df = df.dropna()
        gam = LinearGAM(s(0) + s(1) + s(2)).fit(df.reset_index()[['index', self.ticker_label_2, self.ticker_label_3]], df[self.ticker_label_1])
        gam.summary()

        # 4. plot gam
        fig, axs = plt.subplots(1,3)

        titles = [self.ticker_label_1, self.ticker_label_2, self.ticker_label_3]
        for i, ax in enumerate(axs):
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, width=.95)
            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi, c='r', ls='--')
            ax.set_title(titles[i])

        # save
        fig.savefig(f"{self._store_path}/gam.png")
  
        # 5. Make Prediction
        def gam_prediction(gam=gam, df=df):
            df          = df.copy()
            yhat        = gam.predict(df.reset_index()[['index', self.ticker_label_2, self.ticker_label_3]])
            df['yhat']  = yhat
            return df
        

        df2 = gam_prediction()
        # 6. Plot prediction vs actual: create df
        df_plot             = df.copy()
        df_plot['source']   = 'actual'
        df_plot2            = df2.copy()
        df_plot2['source']  = 'predicted'
        df_plot2['USDZAR']  = df_plot2['yhat']
        df_plot2            = df_plot2.drop(columns=['yhat']) 
        df_plot             = pd.concat([df_plot, df_plot2], axis=0)

        # 6. Plot prediction vs actual: build plot
        fig = px.scatter_3d(df_plot, x='Datetime', y=self.ticker_label_2, z=self.ticker_label_1, color=self.ticker_label_3, symbol='source', color_continuous_scale='turbo')
        fig.update_layout(template=None, title=f'{self.ticker_label_1} ~ (Datetime, {self.ticker_label_2}, {self.ticker_label_3})', showlegend=False)
        fig.write_html(f"{self._store_path}/3d_scatter_prediction.html")

        logging.info('3D scatter plot prediction saved to html.')

        # 7. Plot over a larger prediction plane to examine the model: build prediction space
        Z = pd.DataFrame(list(itertools.product(
                df.reset_index()['index'],
                df[self.ticker_label_2].unique(),
                df[self.ticker_label_3].unique()
                )), columns=['index', self.ticker_label_2, self.ticker_label_3])
        

        # 7. Plot over a larger prediction plane to examine the model: 
        #   take every 100 sample, to downsample the data
        _Z = Z.iloc[::100, :]

        # fit
        yhat        = gam.predict(_Z)
        _Z          = _Z.set_index('index').join(df['Datetime'])
        _Z['yhat']  = yhat


        # transform to grid
        # _Z.set_index('Datetime', inplace=True)
        _Z = _Z.reset_index().pivot_table(index='Datetime', columns=self.ticker_label_2, values='yhat')

        # 8. plot
        fig = go.Figure(data=[go.Surface(z=_Z.values, x=_Z.index, y=_Z.columns)])
        fig = go.Figure(data=[
            go.Surface(z=_Z.values, x=_Z.index, y=_Z.columns),
            go.Scatter3d(x=df['Datetime'], y=df[self.ticker_label_2], z=df[self.ticker_label_1], mode='markers',
            marker=dict(size=5, color='lightblue',line=dict(color='darkblue', width=1.5)))
            ])

        # change legend header
        fig.update_layout(scene = dict(
                            xaxis_title='',
                            yaxis_title=self.ticker_label_2,
                            zaxis_title=self.ticker_label_1,
            ),
            legend=dict(title='S&P500', yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=1200, color="white")))
        fig.update_layout(template='plotly_dark', title=f'{self.ticker_label_1} ~ (Datetime, {self.ticker_label_2}, {self.ticker_label_3})')

        # save
        fig.write_html(f"{self._store_path}/3d_hyperplane_scatter.html")

        logging.info('3D hyperplane scatter plot saved to html.')
        return self



    def build_html_document(self):
        # set params 
        PRICE           = round(self._close, 3)
        TIME            = self._datetime
        TIMEZONE        = self._timezone
        OUTPUT_PATH     = self._store_path
        _OUTPUT_PATH    = '' # relative to html file.
        FIGURE_WIDTH    = 600

        html_file_text_a  = """
            <!DOCTYPE html>
            <html lang="en">
            <style>
        """

        html_file_text_b = """
            p       {margin: 0; padding: 0;}
            p.ex1   {font-size: 30px;}
            p.ex2   {font-size: 180px;}
        """

        
        html_file_text_c = f"""
            </style>
            <title>Examples</title>
            <p class="ex2">{PRICE} <span style="font-size: 80px;">USD/ZAR</span></p>
            <p class="ex1">{TIME} ({TIMEZONE})</p>
            <br>
            <br>
            <br>

            <!-- <div w3-include-html="sp500.html"></div> -->
            <embed type="text/html" src=".{_OUTPUT_PATH}/gam.html"                           width="{FIGURE_WIDTH*2}"   height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/3d_hyperplane_scatter.html"         width="{FIGURE_WIDTH}"     height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/3d_scatter_prediction.html"         width="{FIGURE_WIDTH}"     height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/3d_scatter.html"                    width="{FIGURE_WIDTH}"     height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}//pie_chart_recommendation.html"      width="{FIGURE_WIDTH}"    height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/loess.html"                         width="{FIGURE_WIDTH}"     height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/sp500.html"                         width="{FIGURE_WIDTH}"     height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/usdzar.html"                        width="{FIGURE_WIDTH}"     height="500">
            <embed type="text/html" src=".{_OUTPUT_PATH}/UStreasury.html"                    width="{FIGURE_WIDTH}"     height="500">


            <textarea id="myTextarea" rows="20" cols="150">
                ______________________________________________________________________________________________________________________________________________
                ______________________________________________________________________________________________________________________________________________

                        /$$          /$$$$$$$$                                            /$$$$$$$$ /$$                                        
                      /$$$$$$       | $$_____/                                           | $$_____/| $$                                        
                     /$$__  $$      | $$     /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$      | $$      | $$  /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$ 
                    | $$  \__/      | $$$$$ /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/      | $$$$$   | $$ |____  $$ /$$__  $$ /$$__  $$ /$$__  $$
                    |  $$$$$$       | $$__/| $$  \ $$| $$  \__/| $$$$$$$$ \  $$$$/       | $$__/   | $$  /$$$$$$$| $$  \ $$| $$  \ $$| $$  \__/
                     \____  $$      | $$   | $$  | $$| $$      | $$_____/   $$  $$       | $$      | $$ /$$__  $$| $$  | $$| $$  | $$| $$      
                     /$$  \ $$      | $$   |  $$$$$$/| $$      |  $$$$$$$ /$$/\  $$      | $$      | $$|  $$$$$$$|  $$$$$$$|  $$$$$$$| $$      
                    |  $$$$$$/      |__/    \______/ |__/       \_______/|__/  \__/      |__/      |__/ \_______/ \____  $$ \____  $$|__/      
                     \_  $$_/                                                                                     /$$  \ $$ /$$  \ $$          
                       \__/                                                                                      |  $$$$$$/|  $$$$$$/          
                                                                                                                  \______/  \______/                   
                ______________________________________________________________________________________________________________________________________________
                ______________________________________________________________________________________________________________________________________________                                                 

            </textarea>
            <br>
            <br>
            For more info, visit the project <a href="https://github.com/ZachWolpe/forexflaggr">repository.</a>
            <br>
            <br>

            </html>
        """

        # save html_file_text as html file
        html_file_text = html_file_text_a + html_file_text_b + html_file_text_c
        with open(f'{OUTPUT_PATH}/forexflaggr.html', 'w') as f:
            f.write(html_file_text)
        
        logging.info('HTML document saved to html.')
        return self


