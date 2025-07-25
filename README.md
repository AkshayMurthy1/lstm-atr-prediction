# Investigating Methods to Accurately Model Volatility through Various Deep Learning Approaches
The aim of this project is to use compare the effectiveness of forecasting ATR versus traditional metrics such as Squared returns or Standard Deviation using various Deep Learning approaches. Specifically, we look at the accuracy metrics of LSTM and RNN used to predict ATR, Squared Returns, and Rolling Standard Deviation across 8 ETFs on daily stock data. Historically, using ATR in volatility forecasting as been under-theorized, and we hope to gain new insights on different metrics on modelling volatility and investigate their effects on portfolio returns when incorporated into a simple backtest trading strategy.

## Definitions
### ATR

$ATR = \frac{\sum_{i=1}^{n}TR_i}{n}$

#### TR

$Max[{(H-L)}, Abs{(H-C_p)}, Abs{(L-C_p)}]$

### STD of log returns

$STD\ of\ log\ returns = {\frac{\sum{log(\frac{C_t}{(C_{t-1})})}}N}$


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
To reproduce the results in the `./Output` folder, Run all cells in the `full_sims_test.ipynb` notebook. A new folder should be created in `./Output` named `./runtest_{i}` that holds all the evaluation metrics, including graphs and dataframes that signify various accuracy metrics.

## License
[MIT](https://choosealicense.com/licenses/mit/)
