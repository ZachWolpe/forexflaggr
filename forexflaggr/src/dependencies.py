"""
------------------------------------------------------------------------------------------------
dependencies.py

Centralized package dependencies.

: 18.11.23
: Zach Wolpe
: zach.wolpe@mlxgo.com
------------------------------------------------------------------------------------------------
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from pygam import LinearGAM
from moepy import lowess
import datetime as dt
import yfinance as yf
import numpy as np

warnings.filterwarnings('ignore')