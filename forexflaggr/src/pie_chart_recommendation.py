"""
------------------------------------------------------------------------------------------------
pie_chart_recommendation.py

Pie chart recommendation class.

    - compute pie percentage
        (percentage above MA over a given period)
    - return buy recommendation
    - generate pie chart

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

from services.dependencies import *

# pie chart above MA

class pie_chart_recommendation:
    n_samples = 200

    @staticmethod
    def compute_pie_perc(df_all, n_samples:int):    
        x       = df_all.iloc[-n_samples:,]
        perc    = np.mean(x.Close >= x.wma) * 100
        return perc, perc + 180

    @staticmethod
    def buy_recommendation(perc):
        _recommendation = {
            -1: 'DO NOT BUY.',
            0:  'MONITOR.',
            1:  'BUY!'
        }
        _perc       = perc/100.
        _thresholds = [0.7, 0.5]
        if _perc >= _thresholds[0]:
            return _recommendation[1]
        if _perc >= _thresholds[1]:
            return _recommendation[0]
        return _recommendation[-1]


    @staticmethod
    def plot_pie_recommendation(df_all, fig, n_samples=None, *args, **kwargs):
        if n_samples is None:
            n_samples = pie_chart_recommendation.n_samples
        perc, _     = pie_chart_recommendation.compute_pie_perc(df_all, n_samples)
        perc        = round(perc, 2)
        _rec        = pie_chart_recommendation.buy_recommendation(perc)
        _colours    = ['green', 'rgba(0,0,0,0)']
        _colours    = ['green', '#f3b77a']
        # _colours    = ['navy', '#f3b77a']



        fig.add_trace(go.Pie(labels=['Buy',' '], values=[perc, 100-perc], hole=0.5, sort=False, showlegend=False), *args, **kwargs)
        fig.update_traces(textinfo='none', marker=dict(colors=_colours, line=dict(color='#000000', width=1.5)))
        fig.update_layout(
            # title_text="Rate Above WMA",
            annotations=[dict(text=f'{perc}%',  x=0.5, y=0.5, font_size=25, showarrow=False),
                        dict(text=f'{_rec}',    x=0.5, y=-0.2, font_size=30, showarrow=False)
                        ])
        return fig


