import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data():
    df = pd.read_csv('data/NFLX.csv')
    df.sort_values('Date', inplace=True)

    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    # date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
    # df = df.reindex(date_range)
    # df.ffill(inplace=True)
    
    df['Mid'] = (df['Open'] + df['Close']) / 2
    
    return df

def scale_data(df, train_df, test_df):
    scaler = MinMaxScaler()
    
    window_size = 1000
    train_scaled = np.zeros_like(train_df[['Mid']])

    for i in range(0, train_df.shape[0], window_size):
        window = train_df[['Mid']][i: i + window_size]
        train_scaled[i: i + window_size] = scaler.fit_transform(window)
        
    train_df['Scaled'] = train_scaled
    test_df['Scaled'] = scaler.transform(test_df[['Mid']])
    df['Scaled'] = pd.concat([train_df['Scaled'], test_df['Scaled']])
    
    return df, train_df, test_df

def smooth_data(train_df):
    train_df['Smoothed'] = train_df['Scaled'].ewm(com=0.5).mean()
    return train_df



df = load_data()
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
df, train_df, test_df = scale_data(df, train_df, test_df)
train_df = smooth_data(train_df)