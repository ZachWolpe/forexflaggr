"""
------------------------------------------------------------------------------------------------
gam_model.py

A GAM model class, building of the stochastic model interface.

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

from .dependencies import (LinearGAM, go, np)
from .stochastic_model_interface import Stochastic_Model_Interface




class GAM_Model(Stochastic_Model_Interface):
    """
    Generalized Additive Model
        documentation: https://pygam.readthedocs.io/en/latest/api/linearGAM.html
    """
    n_samples = 500

    def __init__(self, X, y) -> None:
        super().__init__(X, y, LinearGAM, 'GAM')
        self._n_splines = 25

    @property
    def n_splines(self):
        return self._n_splines
    
    @n_splines.setter
    def n_splines(self, n):
        self._n_splines = n
        self.model      = LinearGAM(n_splines=n).gridsearch(self.X, self.y)

    @Stochastic_Model_Interface.internal_tracing
    def fit(self, X=None, y=None):
        X,y         = self.check_Xy(X,y)
        self.model  = LinearGAM(n_splines=self.n_splines).gridsearch(X, y)
        self.XX     = self.model.generate_X_grid(term=0, n=GAM_Model.n_samples)
        return self
    
    @Stochastic_Model_Interface.internal_tracing
    def transform(self):
        self.yhat, self.CI = self.predict(self.XX)
        self.CI_lower, self.CI_upper = self.CI[:,0], self.CI[:,1]
        return self

    def fit_transform(self, X=None, y=None):
        self.fit(X,y)
        self.transform()
        return self
    
    def predict(self, x_pred, CI_width=.95):
        """returns the mean prediction & confidence intervals"""
        return self.model.predict(x_pred), self.model.prediction_intervals(x_pred, width=CI_width)
    
    @Stochastic_Model_Interface.internal_tracing
    def extrapolate(self, CI_width=0.95, n_steps=500):
        """Extrapolate forward n_steps."""
        # self.internal_logger('Extrapolating {n_steps} forward...')
        # m   = self.X.min()
        M   = self.X.max()
        self.Xforward  = np.linspace(M, M + n_steps, n_steps)
        self.yhatforward, self.CIforward = self.predict(self.Xforward, CI_width)
        self.CIf_lower, self.CIf_upper   = self.CIforward[:,0], self.CIforward[:,1]
        return self
    
    @Stochastic_Model_Interface.internal_tracing
    def build(self, X=None, y=None, n_steps=500):
        self.check_Xy(X,y)
        self.fit(X=X, y=y)\
            .transform()\
            .extrapolate(n_steps=n_steps)\
            .plot_prediction()
        return self

    @Stochastic_Model_Interface.internal_tracing
    def plot_prediction(self, c1='orange', c2='darkblue', c3='lightblue', *args, **kwargs):
        """Plot the prediction & confidence intervals."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.X.flatten(),  y=self.y.flatten(),   name='raw data', line=dict(color=c1)))
        fig.add_trace(go.Scatter(x=self.XX.flatten(), y=self.yhat,          name='prediction', line=dict(color=c2)))
        fig.add_trace(go.Scatter(x=self.XX.flatten(), y=self.CI_lower,      name='confidence', line=dict(color=c3)))
        fig.add_trace(go.Scatter(x=self.XX.flatten(), y=self.CI_upper,      name=self.model_name, line=dict(color=c3), showlegend=False))
        fig.update_layout(template='none', title=f'{self.model_name} ~ (USD/ZAR)', yaxis_title='Exchange Rate', xaxis_title='Date')
        
        try:
            # if extrapolated
            # make lines dashed
            fig.add_trace(go.Scatter(x=self.Xforward, y=self.yhatforward, name='GAM', line=dict(color='black', dash='dash'), showlegend=False))
            fig.add_trace(go.Scatter(x=self.Xforward, y=self.CIf_lower, name='GAM', line=dict(color='black', dash='dash'), showlegend=False))
            fig.add_trace(go.Scatter(x=self.Xforward, y=self.CIf_upper, name='GAM', line=dict(color='black', dash='dash'), showlegend=False))
        except Exception:
            pass
        self.fig = fig
        return self
   

