# Reassessing Volatility for Financial Forecasting: An Empirical Study of ATR vs. Standard Deviation in Predictive Trading with RNNs
 ## Abstract
Over the past two decades, portfolio analysis and financial risk management have become increasingly data-driven fields of study. Particularly, forecasting volatility has gained traction recently due to its implications in portfolio optimization and risk assessment, as high volatility indicates investor uncertainty, often associated with bearish sentiments. Common volatility proxies in financial research have included the standard deviation of prices and absolute daily returns. As such, various methods have been implemented to forecast such quantities, including GARCH (Generalized AutoRegressive Conditional Heteroskedasticity), DNNs (Deep Neural Networks), and RNNs (Recurrent Neural Networks), among others. However, recent studies have criticized standard deviation for failing to capture real-time spikes due to its lagging nature. Moreover, it is reliant on restricted distribution assumptions, such as normality, and is sensitive to outliers. In this study, we investigate the implementation of Average True Range (ATR), a model-free volatility proxy that averages the daily price fluctuations across an interval. We test its performance in next-day forecasting and trading strategy using two types of Recurrent Neural Network (RNN) models across nine sector-based Exchange-Trade Funds (ETFs) representing diverse price and volatility profiles. We benchmark against the standard deviation of the squared returns as well as the standard deviation of the raw prices, two heavily explored metrics. We find that forecasting ATR consistently achieves higher accuracy than the standard deviation of squared returns and the standard deviation of prices on all S&P 500 Sector ETFs, with two notable exceptions. Additionally, we find that ATR performs better (around 3-4% higher returns) in ETFs that tend to exhibit large jumps or drops. We propose ATR as a better volatility proxy for predicting daily price ranges in markets characterized by asymmetric and heavy-tailed return distributions.

## Definitions
### ATR

$ATR = \frac{\sum_{i=1}^{N}TR_i}{N}$

#### TR

$Max[{(H-L)}, Abs{(H-C_p)}, Abs{(L-C_p)}]$

### S.D. of log returns

$SD\ of\ log\ returns = {\frac{(\sum_{t=1}^N{log(\frac{C_t}{(C_{t-1})})}-\mu_u)^2}N}$

#### S.D. of prices

$SD\ of\ prices = \frac{\sum_{t=1}^N(C_t-\mu_c)^2}{N}$



## Installation
Download and install version 1.85.1 of [Visual Studio Code](https://code.visualstudio.com/download). Then, in the terminal, create an env using the environment.txt file and [conda](https://www.anaconda.com/download).

``` bash
conda create -n <your env name> --file environment.txt -y
```

If this does not work, use pip to manually install imported libraries. 

```bash
pip install -r requirements.txt
```

## Usage
To reproduce the results in the `./Output` folder, Run all cells in the `full_sims_test.ipynb` and `final_analysis.ipynb` notebooks. A new folder should be created in `./output` named `./runtest_{i}` that holds all the evaluation metrics, including graphs and dataframes that signify various accuracy metrics. 

## License
[MIT](https://choosealicense.com/licenses/mit/)
