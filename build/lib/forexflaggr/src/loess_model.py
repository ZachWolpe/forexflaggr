"""
------------------------------------------------------------------------------------------------
loess_model.py

A loess model class, building of the stochastic model interface.

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

from .dependencies import (lowess, go)
from .stochastic_model_interface import Stochastic_Model_Interface





class LOESS_Model(Stochastic_Model_Interface):
    """LOESS Model"""
    
    def __init__(self, X, y, model=lowess.Lowess(), model_name='LOESS'):
        super().__init__(X, y, model, model_name)
    
    @Stochastic_Model_Interface.internal_tracing
    def fit(self, X=None, y=None):
        X,y = self.check_Xy(X,y)
        self.model.fit(X, y)
        self.internal_logger
        return self
    
    @Stochastic_Model_Interface.internal_tracing
    def transform(self):
        self.yhat = self.model.predict(self.X)
        return self
    
    def fit_transform(self, X=None, y=None):
        self.fit(X, y)
        self.transform()
        return self
    
    def predict(self, x_pred):
        return self.model.predict(x_pred)
    
    def extrapolate(self, x_pred, n_steps=500):
        raise NotImplementedError
    
    @Stochastic_Model_Interface.internal_tracing
    def build(self, X=None, y=None, n_steps=500):
        """Implement fit & extrapolate."""
        self.fit_transform(X, y)
        return self

    @Stochastic_Model_Interface.internal_tracing
    def plot_prediction(self, c1='orange', c2='black', *args, **kwargs):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.X, y=self.y, name='USD/ZAR',            line=dict(color=c1)))
        fig.add_trace(go.Scatter(x=self.X, y=self.yhat, name=self.model_name,   line=dict(color=c2)))
        fig.update_layout(template='none', title='USD/ZAR', yaxis_title='Exchange Rate', xaxis_title='Date', *args, **kwargs)
        self.fig = fig
        return self