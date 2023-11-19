"""
------------------------------------------------------------------------------------------------
stochastic_model_interface.py

A base class that provides an interface for stochastic models.

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

from .dependencies import (dt)

class Stochastic_Model_Interface:
    """Interface for stochastic models"""
    
    def __init__(self, X, y, model, model_name):
        self.X = X
        self.y = y
        self.model      = model
        self.model_name = model_name
    
    @staticmethod
    def internal_logger(msg):
        print(f'[{dt.datetime.now()}] {msg}')

    @staticmethod
    def internal_tracing(func):
        """Decorator for internal tracing (timing & logging)."""
        def wrapper(*args, **kwargs):
            # Stochastic_Model_Interface.internal_logger(f'Running {func.__name__}.')
            start   = dt.datetime.now()
            result  = func(*args, **kwargs)
            end     = dt.datetime.now()
            s = ''
            l = max(0,15-len(func.__name__))
            Stochastic_Model_Interface.internal_logger(f'<{func.__name__}> finished in {s:<{l}}{end-start}.')
            return result
        return wrapper
        
    def check_Xy(self,X,y):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        self.X, self.y = X,y
        return X,y
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError
    
    def fit_transform(self, X, y):
        raise NotImplementedError
    
    def predict(self, x_pred):
        raise NotImplementedError
    
    def extrapolate(self, x_pred, n_steps=500):
        raise NotImplementedError
    
    def build(self, X=None, y=None, n_steps=500):
        """Implement fit & extrapolate."""
        raise NotImplementedError
    
    def plot_prediction(self, X=None, y=None, n_steps=500):
        """Implement plot."""
        raise NotImplementedError
    