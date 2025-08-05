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
import seaborn as sns
warnings.filterwarnings("ignore")

T=30 #period of 30 days
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE: ",device)

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
    
def tt_split(df_n,vol_metric,scaler:StandardScaler,scaler_x:StandardScaler,feats = ['Open','Close','High','Low','Volume']):
    train = df_n.loc[[i<=len(df_n)*4/5 for i in range(len(df_n))]]
    X_train = train[feats].to_numpy()
    X_train = scaler_x.fit_transform(X_train)
    y_train = train[vol_metric].to_numpy()
    y_train = scaler.fit_transform(y_train.reshape(-1,1))
    #long_atr = y_train.mean()
    X_train = np.concatenate((X_train,y_train),axis=1)

    test = df_n.loc[[i>len(df_n)*4/5 for i in range(len(df_n))]]
    X_test = test[feats].to_numpy()
    y_test = test[vol_metric].to_numpy()
    X_test =scaler_x.transform(X_test)
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
        self.first = LSTM(input_size=input_size,hidden_size=50)
        self.second = LSTM(input_size=50,hidden_size=70)
        #self.first = RNN(input_size=input_size,hidden_size=50)
        self.fc = Linear(70,1)
        
    def activation(self,X):
        return F.relu(X)
    def forward(self,input):
        input,_ = self.first(input)
        input,_ = self.second(input)
        input = self.fc(input[-1,:,:])
        #print(input.shape)
        return input #return the last prediction

class NN_RNN(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        #self.first = LSTM(input_size=input_size,hidden_size=50)
        self.first = RNN(input_size=input_size,hidden_size=50)
        self.fc = Linear(50,1)
        
    def activation(self,X):
        return F.relu(X)
    def forward(self,input):
        input,_ = self.first(input)
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
    df_n["TR"] = tr
    df_n["ATR"] = tr.rolling(7).mean()
    
    log_diff = np.log(df_n["Close"]/df_n["Close"].shift(1))
    df_n["SD_Log_Close"] = log_diff.rolling(7).std()
    #df_n["ATR_normalized"] = (df_n["ATR"] - df_n["ATR"].mean())/df_n["ATR"].std()
    #df_n["SD_normalized"] = (df_n["SD_Log_Close"] - df_n["SD_Log_Close"].mean())/df_n["SD_Log_Close"].std()

    df_n["SR"] = np.log((close/prev_close)**2)
    df_n["SD_Squared_Returns"] = df_n["SR"].rolling(7).std()
    df_n["SD_Prices"] = close.rolling(7).std()
    q25 = [np.quantile(roll,.25) for roll in df_n["SR"].rolling(7)]
    q75 = [np.quantile(roll,.75) for roll in df_n["SR"].rolling(7)]
    #print("QUANTILE25: ",q25)
    #print("QUANTILE75: ",q75)
    df_n["IQR"] = np.array(q75)-np.array(q25)
    df_n = df_n.dropna()
    df_n = df_n.reset_index().reset_index()
    df_n["index"] = df_n.index%7
    return df_n

def get_trained_model(df,scaler,scaler_x,metric="ATR",model_type = "RNN"):
    X_train,y_train,X_test,y_test = tt_split(df, metric,scaler,scaler_x)
    in_size = X_train.shape[1]

    X_seq,y_seq,X_seq_test,y_seq_test = make_seq(X_train,y_train,X_test,y_test)

    loader,loader_test = create_loaders(X_seq,y_seq,X_seq_test,y_seq_test)

    #training loop
    
    if model_type == "LSTM":
        model = NN_LSTM(input_size=in_size,output_size=1)
    else:
        model = NN_RNN(input_size=in_size,output_size=1)
    model = model.to(device)
    epochs = 50
    optim = torch.optim.Adam(params = model.parameters())
    crit = MSELoss()
    pde_crit = MSELoss()
    losses = []
    losses_test = []

    for i in range(epochs):
        running_loss = 0
        for x_window,y_atr in loader:
            #print("Running")
            input = x_window.permute(1,0,2) #shape = [seq_length,batch_length,4]
            #t = input[-1][:,0]
            input = input.to(device)
            y_atr = y_atr.to(device)
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

        with torch.no_grad():
            testing_loss = 0
            for x_window_test,y_atr_test in loader_test:
                x_window_test = x_window_test.to(device)
                out_test = model(x_window_test.permute(1,0,2))
                #print(y_atr_test.shape)
                y_atr_test = y_atr_test.to(device)
                loss = crit(out_test,y_atr_test)
                testing_loss+=loss.item()
            losses_test.append(testing_loss/(len(loader_test)))

            
    fig_tr,ax_tr = plt.subplots()
    sns.lineplot(x=[i for i in range(len(losses))],y=losses,ax=ax_tr)
    ax_tr.set_title(f"Training Loss of {model_type} ({metric}) across {epochs} epochs")

    fig_te,ax_te = plt.subplots()
    sns.lineplot(x=[i for i in range(len(losses_test))],y=losses_test,ax=ax_te)
    ax_te.set_title(f"Testing Loss of {model_type} ({metric}) across {epochs} epochs")
    return model,fig_tr,fig_te
    



# Assumes model is a torch.nn.Module, and scaler/scaler_x are fitted StandardScaler instances

import numpy as np
import pandas as pd
import torch

# Assumes model is a torch.nn.Module, and scaler/scaler_x are fitted StandardScaler instances

def backtest_strategy_mr(data: pd.DataFrame,
                          lstm_model:NN_LSTM,
                          scaler:StandardScaler,
                          scaler_x:StandardScaler,
                          vol_metric:str,
                          ini_cash=1000,
                          R: float = 100.0,
                          buy_scale = 1.5,
                          sell_scale = 1.5,lr=.001,
                          feats = ['Open','Close','High','Low','Volume'],
                          buy_max=500) -> tuple:
    cash = ini_cash
    shares = 0.0

    buys = []
    sells = []
    preds = []
    preds_norm = []
    
    passive_shares = ini_cash/data.loc[T,"Open"]
    # Precompute rolling IQR thresholds on ATR over T bars
    # (we’ll compute quantiles on‑the‑fly inside the loop)
    
    crit = MSELoss()
    optim = torch.optim.Adam(lstm_model.parameters(),lr=lr) # have higher learning rate
    t_money = []
    p_money=[]
    atr_past_a = 0
    bias=0
    uppers = []
    lowers = []
    vols = []

    for i in range(T, len(data)-1):
        # if i>T:
        #     true_atr_norm = scaler.transform(np.reshape(data[vol_metric].iloc[i],(-1,1)))
        #     #print("P size: ",len(preds_norm))
        #     loss = crit(torch.tensor(torch.tensor([true_atr_norm],dtype=torch.float32)),preds_norm[-1])
        #     loss.backward()
        #     optim.step()

        ## Bias
        # if (i-T)%(10) == 0:
        #     if i!=T:
        #         past_ten_preds = preds[i-10-T:i-T]
        #         #print("SAHEP: ",past_ten_preds,i)
        #         bias = np.mean(data[vol_metric].iloc[i-10:i]-past_ten_preds)
        # Prepare model input: last T bars of OHLC + normalized ATR

        #define upper and lower
        #past_mets = data[vol_metric].iloc[i-T:i]
        #print("Len:",len(preds))
        #print("Our ind:",-min(len(preds),T))

        past_mets = np.array(preds[-min(len(preds),T):])
        if len(past_mets)<T:
            to_add = np.array(data[vol_metric].iloc[i-T+1:i-(len(past_mets))+1]) #making this make sense with current day as predcitor
            past_mets = np.append(to_add,past_mets)
        upper = past_mets.mean() + sell_scale*past_mets.std()
        lower = past_mets.mean()-buy_scale*past_mets.std()
        lowers.append(lower)
        uppers.append(upper)

        X = data[feats].iloc[i-T+1:i+1].copy() #predict the next day and buy/sell on that day's close
        X[feats] = scaler_x.transform(X[feats])    
        X['ATR_norm'] = scaler.transform(data[[vol_metric]].iloc[i-T:i])  # shape (T,1)
        # reshape to (1, T, features)
        #print(X)
        model_in = torch.tensor(X.values.reshape(T,1, X.shape[1])).to(torch.float32).to(device)
        # Predict next normalized ATR, then denormalize
        met_next_norm = lstm_model(model_in)
        #print(met_next_norm.shape)
        preds_norm.append(met_next_norm)
        met_next_norm=met_next_norm.item()
        
        met_next = scaler.inverse_transform([[met_next_norm]])[0,0] + bias
        vols.append(met_next)
        #if met_next<0:
        #    print(met_next)
        preds.append(met_next)

        #define open and close
        open = data['Open'].iloc[i]
        close = data['Close'].iloc[i]
        # next bar’s prices
        open_next  = data['Open'].iloc[i+1]
        close_next = data['Close'].iloc[i+1]

        atr_past_a += (data['ATR'] - met_next)


        # entry/exit signals
        if met_next > upper and shares > 0.0:
            sells.append(i)
            # sell all
            cash += shares * close
            print(f"On the {i}th day, sold {shares} shares for ${shares*close}")
            shares = 0.0
            
        elif met_next < lower:
            # buy: risk R = shares * met_next -> shares = R / met_next
            target_shares = (R / met_next)
            # print("Lower: ",lower)
            # print("ATR Next: ",met_next)
            # print("Target shares: ", target_shares)
            # adjust cash & position
            delta = max(target_shares - shares,0) #keep it positive (sanity check)
            delta = min(delta,buy_max/close) #don't let it be above a certain threshold of shares to buy
            # print("Delta: ",delta)
            # print("Total for Delta: $", delta*open_next)
            if cash<delta*close:
                for j in range(0,int(delta)):
                    if cash>j*close:
                        delta = j
            if cash>delta*close:
                cash -= delta * close
                shares+=delta
                print(f"On the {i}th day, Bought {delta} shares for ${delta*close}")
                buys.append(i)
        t_money.append(cash+shares*data['Close'].iloc[i])
        p_money.append(passive_shares*data['Close'].iloc[i])
    # At end, mark-to-market at last close
    final_value = cash + shares * data['Close'].iloc[-1]
    passive_value = passive_shares*data['Close'].iloc[-1]

    return final_value, cash, shares,passive_value,buys,sells,preds, t_money,p_money, lowers, uppers, vols


def backtest_strategy(data: pd.DataFrame,


                          lstm_model:NN_LSTM,
                          scaler:StandardScaler,
                          scaler_x:StandardScaler,
                          vol_metric:str,
                          ini_cash=1000,
                          R: float = 1000.0,
                          buy_scale = 1.5,
                          sell_scale = 1.5,lr=.001,
                          feats = ['Open','Close','High','Low','Volume']) -> tuple:

    cash = ini_cash
    shares = 0.0

    buys = []
    sells = []
    preds = []
    preds_norm = []
    
    passive_shares = ini_cash/data.loc[T,"Open"]
    # Precompute rolling IQR thresholds on ATR over T bars
    # (we’ll compute quantiles on‑the‑fly inside the loop)

    crit = MSELoss()
    optim = torch.optim.Adam(lstm_model.parameters(),lr=lr) # have higher learning rate
    t_money = []
    p_money=[]
    atr_past_a = 0
    bias=0
    uppers = []
    lowers = []
    vols = []
    for i in range(T, len(data)-1):
        # if i>T:
        #     true_atr_norm = scaler.transform(np.reshape(data[vol_metric].iloc[i],(-1,1)))
        #     #print("P size: ",len(preds_norm))
        #     loss = crit(torch.tensor(torch.tensor([true_atr_norm],dtype=torch.float32)),preds_norm[-1])
        #     loss.backward()
        #     optim.step()

        ## Bias
        # if (i-T)%(10) == 0:
        #     if i!=T:
        #         past_ten_preds = preds[i-10-T:i-T]
        #         #print("SAHEP: ",past_ten_preds,i)
        #         bias = np.mean(data[vol_metric].iloc[i-10:i]-past_ten_preds)

        if vol_metric == "SD_Squared_Returns" or "SD_Prices":
            arr = data['Close'].iloc[i-T:i]
            mu = arr.mean()
            sigma = arr.std()
            lower = mu - buy_scale * sigma
            upper = mu + sell_scale * sigma
            
        else:
            # ATR-based bands
            # --- Compute rolling bands on price using ATR ---
            price_window = data['Close'].iloc[i-T:i]
            sma = price_window.mean()
            atr_band = data['TR'].iloc[i-T:i].mean()
            lower = sma - buy_scale * atr_band
            upper = sma + sell_scale * atr_band

        lowers.append(lower)
        uppers.append(upper)
        # Prepare model input: last T bars of OHLC + normalized ATR
        X = data[feats].iloc[i-T:i].copy()
        X[feats] = scaler_x.transform(X[feats])    
        X['ATR_norm'] = scaler.transform(data[[vol_metric]].iloc[i-T:i])  # shape (T,1)
        # reshape to (1, T, features)
        #print(X)
        model_in = torch.tensor(X.values.reshape(T,1, X.shape[1])).to(torch.float32).to(device)
        # Predict next normalized ATR, then denormalize
        met_next_norm = lstm_model(model_in)
        #print(met_next_norm.shape)
        preds_norm.append(met_next_norm)
        met_next_norm=met_next_norm.item()
        
        met_next = scaler.inverse_transform([[met_next_norm]])[0,0] + bias
        #if met_next<0:
        #    print(met_next)
        preds.append(met_next)

        # next bar’s prices
        open_next  = data['Open'].iloc[i+1]
        close_next = data['Close'].iloc[i+1]

        atr_past_a += (data['ATR'] - met_next)


        # entry/exit signals
        if data.loc[i, 'Open'] > upper and shares > 0.0:
            sells.append(i)
            # sell all
            cash += shares * close_next
            print(f"On the {i}th day, sold {shares} shares for ${shares*close_next}")
            shares = 0.0
            
        elif data.loc[i, 'Open'] < lower:
            # buy: risk R = shares * met_next -> shares = R / met_next
            target_shares = (R / met_next)
            # print("Lower: ",lower)
            # print("ATR Next: ",met_next)
            # print("Target shares: ", target_shares)
            # adjust cash & position
            delta = max(target_shares - shares,0)
            # print("Delta: ",delta)
            # print("Total for Delta: $", delta*open_next)
            if cash<delta*open_next:
                for j in np.linspace(0,delta,int(delta)):
                    if cash>j*open_next:
                        delta = j
            delta /= 2
            
            cash -= delta * open_next
            shares+=delta
            print(f"On the {i}th day, Bought {delta} shares for ${delta*open_next}")
            buys.append(i)
        t_money.append(cash+shares*data['Close'].iloc[i])
        p_money.append(passive_shares*data['Close'].iloc[i])
    # At end, mark-to-market at last close
    final_value = cash + shares * data['Close'].iloc[-1]
    passive_value = passive_shares*data['Close'].iloc[-1]

    return final_value, cash, shares,passive_value,buys,sells,preds, t_money,p_money, lowers, uppers, vols



def customize_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    #ax.grid(True)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    if ax.get_legend():
        ax.legend(fontsize=10, frameon=True)
