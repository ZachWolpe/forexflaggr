"""
------------------------------------------------------------------------------------------------
forexflaggr_runtime.py

Run all the code from forexflagger_runtime.ipynb to produce the plots

: 24.11.23
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

# Parameters ----------------------------------------------------------->>
_n_days     = 500
_MA_periods = int(12*14) # 2 weeks (1 sample/hour * 14 days)
_store_path = 'output'
# Parameters ----------------------------------------------------------->>

# USDZAR
ff = fxr.ForexFlaggr()
ff\
    .fetch_data(stock='USDZAR=X', n_days=_n_days)\
    .plot_signal(MA_periods=_MA_periods)

# USTreasury Bills
fi = fxr.ForexFlaggr()
fi\
    .fetch_data(stock='^IRX', n_days=_n_days)\
    .plot_signal(MA_periods=_MA_periods)

# S&P500
fs = fxr.ForexFlaggr()
fs\
    .fetch_data(stock='^GSPC', n_days=_n_days)\
    .plot_signal(MA_periods=_MA_periods)

# safety check
assert ff.data.shape[0] > 0, 'No data found for USDZAR, try to clear the  yfinance cache.'

# save plots as html
ff.fig.write_html(f"{_store_path}/usdzar.html")
fi.fig.write_html(f"{_store_path}/UStreasury.html")
fs.fig.write_html(f"{_store_path}/sp500.html")

logging.info('First set of plots saved to html.')



# Price data
_print_data = fxr.ForexFlaggr.get_price(ff)
# save as json
with open(f"{_store_path}/price_data.json", 'w') as f:
    json.dump(_print_data, f)
_datetime, _timezone, _close, _open, _high, _low = _print_data
print(_datetime, _timezone, _close, _open, _high, _low)

logging.info('Price data saved to txt.')

# Price momentum
_n_samples = 252
pcr_fig = fxr.pie_chart_recommendation.plot_pie_recommendation(ff.df_all, n_samples=_n_samples, fig=go.Figure())
pcr_fig.write_html(f"{_store_path}/pie_chart_recommendation.html")


logging.info('Pie momentum chart recommendation saved to html.')

# GAMs + LOESS
# extract data
model_data  = ff.data.reset_index()
X, y        = np.array(model_data.index), np.array(model_data.Close)

# build LOESS model
loess_model = fxr.LOESS_Model(X, y)
loess_model\
    .build()\
    .plot_prediction()
loess_model.fig.write_html(f"{_store_path}/loess.html")

logging.info('LOESS model saved to html.')

# build GAM model
model_data  = ff.data.reset_index()
X,y         = list(model_data.index), model_data.Close
X,y         = np.reshape(X, (-1, 1)), np.reshape(y.values, (-1, 1))


# build GAM model
gam = fxr.GAM_Model(X, y)
gam.build().plot_prediction()
gam.fig.write_html(f"{_store_path}/gam.html")

logging.info('GAM model saved to html.')


# Complex Model
# 1. build dataframe
def transform_df(ff, col_name='USDZAR'):
    df = ff.df_all.copy()
    df.index = df.index.strftime('%Y-%m-%d').copy()
    df = df.groupby(df.index).mean()
    df = df[['Close']]
    df.columns = [col_name]
    return df

df = transform_df(ff, 'USDZAR').join(transform_df(fi, 'USTBills')).join(transform_df(fs, 'S&P500')).reset_index()


# 2. plot 3d scatter plot
fig = px.scatter_3d(df, x='Datetime', y='USTBills', z='USDZAR', color='S&P500')
fig.update_layout(template='plotly_dark', title='USDZAR ~ (USTreasury, S&P500)')
fig.write_html(f"{_store_path}/3d_scatter.html")

logging.info('3D scatter plot saved to html.')

# 3. fit gam: note use index in place of date as a numeric
df = df.dropna()
gam = LinearGAM(s(0) + s(1) + s(2)).fit(df.reset_index()[['index', 'USTBills', 'S&P500']], df['USDZAR'])
gam.summary()

# 4. plot gam
fig, axs = plt.subplots(1,3)
titles = ['USDZAR', 'USTBills', 'S&P500']
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi, c='r', ls='--')
    ax.set_title(titles[i])

# save
try:
    fig.savefig(f"{_store_path}/gam.png")
except Exception:
    pass

# 5. Make Prediction
def gam_prediction(gam=gam, df=df):
    df          = df.copy()
    yhat        = gam.predict(df.reset_index()[['index', 'USTBills', 'S&P500']])
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
fig = px.scatter_3d(df_plot, x='Datetime', y='USTBills', z='USDZAR', color='S&P500', symbol='source', color_continuous_scale='turbo')
fig.update_layout(template=None, title='USDZAR ~ (USTreasury, S&P500)')
fig.write_html(f"{_store_path}/3d_scatter_prediction.html")

logging.info('3D scatter plot prediction saved to html.')

# 7. Plot over a larger prediction plane to examine the model: build prediction space
Z = pd.DataFrame(list(itertools.product(
    df.reset_index()['index'],
    df['USTBills'].unique(),
    df['S&P500'].unique()
    )), columns=['index', 'USTBills', 'S&P500'])


# 7. Plot over a larger prediction plane to examine the model: 
#   take every 100 sample, to downsample the data
_Z = Z.iloc[::100, :]

# fit
yhat        = gam.predict(_Z)
_Z          = _Z.set_index('index').join(df['Datetime'])
_Z['yhat']  = yhat


# transform to grid
# _Z.set_index('Datetime', inplace=True)
_Z = _Z.pivot_table(index='Datetime', columns='USTBills', values='yhat')

# 8. plot
fig = go.Figure(data=[go.Surface(z=_Z.values, x=_Z.index, y=_Z.columns)])
fig = go.Figure(data=[
    go.Surface(z=_Z.values, x=_Z.index, y=_Z.columns),
    go.Scatter3d(x=df['Datetime'], y=df['USTBills'], z=df['USDZAR'], mode='markers',
    marker=dict(size=5, color='lightblue',line=dict(color='darkblue', width=1.5)))
    ])

# change legend header
fig.update_layout(scene = dict(
                    xaxis_title='',
                    yaxis_title='USTBills Yield',
                    zaxis_title='USDZAR',
    ),
    legend=dict(title='S&P500', yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=1200, color="white")))
fig.update_layout(title='USDZAR ~ (Datetime, US-TBills, S&P500)', template='plotly_dark')


fig.write_html(f"{_store_path}/3d_hyperplane_scatter.html")

logging.info('3D hyperplane scatter plot saved to html.')