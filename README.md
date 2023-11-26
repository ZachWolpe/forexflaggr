# forexflaggr

A minimal package to pull and analyze financial (forex) data.

# Getting started


## Installation


Install the latest version of forexflaggr with pip:

```bash
pip install forexflaggr
```

or directly from the source code:

```bash
pip install git+https://github.com/ZachWolpe/forexflaggr.git
```

If the API fails to return data unexpectedly, it may be due to the _yahoo finance_ cache. Empty the cache with:

```bash
pip install yfinance --upgrade --no-cache-dir 
```

# Usage

Import the package:
    
```python
import forexflaggr as fxr
```
Some key functions:

- `fxr.ForexFlaggr().fetch_data()`  : fetches data from _yahoo finance_.`
- `fxr.ForexFlaggr().plot_data()`   : plots data`
- `fxr.ForexFlaggr.get_price(ff)`   : returns the price of a given currency pair`.
- `fxr.pie_chart_recommendation.plot_pie_recommendation()` : plot a recommendation pie chart.
- `fxr.LOESS_Model(X, y)`           : returns a LOESS model.
- `fxr.GAM_Model(X, y)`             : returns a GAM model.


See a complete example usage in `./examples/forexflaggr_runtime.ipynb`

# Off-the-shelf

For an off-the-shelf `USD/ZAR` analysis solution in the form of an `html` doc, run:
    
```python
import forexflaggr as fxr
fxr.build_html()
```
