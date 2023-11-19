"""
------------------------------------------------------------------------------------------------
forexflaggr_runtime.py

Runtime example of forexflaggr package.

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

import forexflaggr as fxr

ff = fxr.ForexFlaggr()
ff.fetch_data("USDZAR=X").plot_signal()
# plotly write plot to jpeg
print(ff.df_all)
import datetime as dt
start_date  = dt.datetime.today() - dt.timedelta(500) 
end_date    = dt.datetime.today()
print(start_date, end_date)
ff.fig.write_image('./forexflaggr_runtime.jpeg')


# # fetch 10 days of USD/ZAR data
# _n_days     = 10
# _MA_periods = int(12*14) # 2 monyhs
# ff = fxr.ForexFlaggr()
# ff\
#     .fetch_data(stock='USDZAR=X', n_days=_n_days)\
#     .plot_signal(MA_periods=_MA_periods)
# ff.fig.show()


# # fetch 10 day interest yield data: 13 WEEK TREASURY BILL (^IRX)

# fi = fxr.ForexFlaggr()
# fi\
#     .fetch_data(stock='^IRX', n_days=_n_days)\
#     .plot_signal(MA_periods=_MA_periods)
# fi.fig.show()
