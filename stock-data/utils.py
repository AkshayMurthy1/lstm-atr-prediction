import torch
from torch.nn import *
import torch.nn.functional as F
from pandas_datareader import data as web
import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore")

T=30 #period of 30 days

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        #yf.pdr_override()
        df = yf.download(ticker, start=start_date, end=end_date,auto_adjust=False)
        #print("DF: ",df)
# can use this as well        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.day_name()
        df['Week_Number'] = df['Date'].dt.strftime('%U')
        df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                    'Week_Number', 'Year_Week', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        num_lines = len(df)
        df = df[col_list]
        print('read ', num_lines, ' lines of data for ticker: ' , ticker)
        return df
    except Exception as error:
        print(error)
        return None
    
def tt_split(df_n,vol_metric,scaler:StandardScaler):
    train = df_n.loc[[i<=len(df_n)*4/5 for i in range(len(df_n))]]
    X_train = train[["index","Open","Close","High","Low"]].to_numpy()
    y_train = train[vol_metric].to_numpy()
    y_train = scaler.fit_transform(y_train.reshape(-1,1))
    #long_atr = y_train.mean()
    X_train = np.concatenate((X_train,y_train),axis=1)

    test = df_n.loc[[i>len(df_n)*4/5 for i in range(len(df_n))]]
    X_test = test[["index","Open","Close","High","Low", vol_metric]].to_numpy()
    y_test = test[vol_metric].to_numpy()
    
    y_test = scaler.transform(y_test.reshape(-1,1))
    X_test = np.concatenate((X_test,y_test),axis=1)
    return X_train,y_train,X_test,y_test
    
def make_seq(X_train,y_train,X_test,y_test):
    #T=30
    #T = 30  # sequence length (window size)
    X_seq = []
    y_seq = []
    X_seq_test = []
    y_seq_test = []

    for i in range(len(X_train) - T):
        X_seq.append(X_train[i:i+T])  # shape: [T, 6] <- what is wanted in lstm
        y_seq.append(y_train[i+T])    # predict next ATR value
    for i in range(len(X_test)-T):    
        X_seq_test.append(X_test[i:i+T])
        y_seq_test.append(y_test[i+T])


    X_seq = torch.tensor(X_seq, dtype=torch.float32)
    y_seq = torch.tensor(y_seq, dtype=torch.float32)#.unsqueeze(1)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
    y_seq_test = torch.tensor(y_seq_test, dtype=torch.float32)#.unsqueeze(1)
    
    return X_seq,y_seq,X_seq_test,y_seq_test

def create_loaders(X_seq,y_seq,X_seq_test,y_seq_test,batch_size=64):
    dataset = TensorDataset(X_seq, y_seq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset_test = TensorDataset(X_seq_test,y_seq_test)
    loader_test = DataLoader(dataset_test,batch_size=batch_size,shuffle=True)
    return loader,loader_test

class NN_LSTM(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = LSTM(input_size=input_size,hidden_size=30)
        self.fc = Linear(30,output_size)
    def activation(self,X):
        return F.relu(X)
    def forward(self,input):
        input,_ = self.lstm(input)
        input = self.fc(input[-1,:,:])
        #print(input.shape)
        return input #return the last prediction
#lstm_layer = LSTM(input_size=4,hidden_size=30)

def get_cleaned_df(ticker,start,end):
    df = get_stock(ticker,start_date=start,end_date=end,s_window=14,l_window=50)
    df_n= df.xs(ticker,axis=1,level=1)
    
        #define ATR columns
    high = df_n["High"]
    low = df_n["Low"]
    close = df_n["Close"]

    prev_close = close.shift(1)


    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    df_n["ATR"] = tr.rolling(7).mean()
    
    log_diff = np.log(df_n["Close"]/df_n["Close"].shift(1))
    df_n["SD_Log_Close"] = log_diff.rolling(7).std()
    df_n["ATR_normalized"] = (df_n["ATR"] - df_n["ATR"].mean())/df_n["ATR"].std()
    df_n["SD_normalized"] = (df_n["SD_Log_Close"] - df_n["SD_Log_Close"].mean())/df_n["SD_Log_Close"].std()

    df_n = df_n.dropna()
    df_n = df_n.reset_index().reset_index()
    df_n["index"] = df_n.index%7
    return df_n

def get_trained_model(df,scaler,metric="ATR"):
    X_train,y_train,X_test,y_test = tt_split(df, metric,scaler)


    X_seq,y_seq,X_seq_test,y_seq_test = make_seq(X_train,y_train,X_test,y_test)

    loader,loader_test = create_loaders(X_seq,y_seq,X_seq_test,y_seq_test)

    #training loop
    
    model = NN_LSTM(input_size=6,output_size=1)
    epochs = 20
    optim = torch.optim.Adam(params = model.parameters())
    crit = MSELoss()
    pde_crit = MSELoss()
    losses = []

    for i in range(epochs):
        running_loss = 0
        for x_window,y_atr in loader:
            #print("Running")
            input = x_window.permute(1,0,2) #shape = [seq_length,batch_length,4]
            #t = input[-1][:,0]
            
            out = model(input)
            #print(y_atr.shape)
            #print(out,y_atr)
            #break
            #print("T shape: ",t.shape)
            loss = crit(out,y_atr)# + pde_crit(PDE_loss(t,out,long_atr))
            running_loss+=loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
                # could try loss += (i+1)/period/sum(j/period for j in range(period))crit(out,y_train[i]); adds a coeff to give more weigt to recent ones
        #break
        running_loss/=(len(loader))
        #print(f"Epoch {i+1} | Training Loss: {running_loss}")
        losses.append(running_loss)

    # sns.lineplot(x=[i for i in range(len(losses))],y=losses)
    # plt.title(f"Training Loss of LSTM ({metric}) across {epochs} epochs")
    # plt.show()
    return model
    

def backtest_strategy(data: pd.DataFrame,
                          lstm_model,
                          scaler:StandardScaler,
                          vol_metric:str,
                          ini_cash=10000,
                          R: float = 1000.0,
                          upper_scale = 1.5,
                          lower_scale = 1.5):
    """
    Backtest a simple ATR‑based mean‑reversion strategy:
      - Predict next ATR with an LSTM
      - If predicted ATR is unusually high → go flat (sell)
      - If unusually low → go long
    Assumes 'data' has columns ['index','Open','High','Low','Close',vol_metric].
    'scaler' is a fitted StandardScaler on the train-period ATR.
    """
    cash = ini_cash
    shares = 0.0

    buys = []
    sells = []
    preds = []
    passive_shares = ini_cash/data.loc[T,"Open"]
    # Precompute rolling IQR thresholds on ATR over T bars
    # (we’ll compute quantiles on‑the‑fly inside the loop)
    for i in range(T, len(data)-1):
        window_atr = data[vol_metric].iloc[i-T:i]
        q1 = window_atr.quantile(0.25)
        q3 = window_atr.quantile(0.75)
        med = window_atr.quantile(0.50)
        iqr = q3 - q1
        lower = med - upper_scale * iqr
        upper = med + lower_scale * iqr

        # Prepare model input: last T bars of OHLC + normalized ATR
        X = data[['index','Open','Close','High','Low']].iloc[i-T:i].copy()
        X['ATR_norm'] = scaler.transform(data[[vol_metric]].iloc[i-T:i])  # shape (T,1)
        # reshape to (1, T, features)
        #print(X)
        model_in = torch.tensor(X.values.reshape(T,1, X.shape[1])).to(torch.float32)
        # Predict next normalized ATR, then denormalize
        atr_next_norm = lstm_model(model_in)
        #print(atr_next_norm.shape)
        atr_next_norm=atr_next_norm.item()
        atr_next = scaler.inverse_transform([[atr_next_norm]])[0,0]
        #if atr_next<0:
        #    print(atr_next)
        preds.append(atr_next)

        # next bar’s prices
        open_next  = data['Open'].iloc[i+1]
        close_next = data['Close'].iloc[i+1]

        # entry/exit signals
        if atr_next > upper and shares > 0:
            sells.append(i)
            # sell all
            cash += shares * close_next
            print(f"On the {i}th day, sold {shares} shares for ${shares*close_next}")
            shares = 0.0
            
        elif atr_next < lower:
            # buy: risk R = shares * ATR_next -> shares = R / ATR_next
            target_shares = R / atr_next
            # print("Lower: ",lower)
            # print("ATR Next: ",atr_next)
            # print("Target shares: ", target_shares)
            # adjust cash & position
            delta = max(target_shares - shares,0)
            # print("Delta: ",delta)
            # print("Total for Delta: $", delta*open_next)
            if cash<delta*open_next:
                for j in range(int(delta)):
                    if cash>j*open_next:
                        delta = j
            cash -= delta * open_next
            shares+=delta
            print(f"On the {i}th day, Bought {delta} shares for ${delta*open_next}")
            buys.append(i)

    # At end, mark-to-market at last close
    final_value = cash + shares * data['Close'].iloc[-1]
    passive_value = passive_shares*data['Close'].iloc[-1]
    return final_value, cash, shares,passive_value,buys,sells,preds
