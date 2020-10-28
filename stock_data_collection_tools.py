# -*- coding: utf-8 -*-
"""
===============================================================================
Stock Data Collection Tools (SDCT) for retrieving stock closing prices for the
german DAX index.
===============================================================================

DESCRIPTION
-----------

This python script contains the necessary functions to retrieve closing prices
for the german DAX index and visualize their cross-correlations based on open
source data.

The following functions have been adopted almost 1 to 1 from Kinsley's Youtube
Playlist 'Python Programming for Finance' at his channel sentdex. The code can
be easily altered to collect and manipulate stock data for any index.

The following functions are building up on top of each other in a linear
fashion:
    save_dax_tickers:
        retrieve the necessary ticker symbols from the respective wikipedia
        page
    get_dax_from_yahoo:
        based on the collection of ticker symbols, send request for
        collecting historical OHLC stock data to Yahoo!Finance
    compile_dax:
        compile the single stock time series into one data table containing
        the adjusted closing prices
    visualize_data:
        display the correlations between each pair of stocks closing prices
        as a heatmap in matrix form

"""

import bs4 as bs #webscraper
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web #get historical OHLC data
import pickle #python object serialisation
import requests #sending http requests
from tqdm import tqdm #progress bar
import os


def save_dax_tickers():
    '''Collect ticker symbols listed in the german DAX from the wikipedia page.

       Returns
       -------
       .pkl object
           a list with the respective ticker symbols 
    
    '''
    
    resp=requests.get('https://de.wikipedia.org/wiki/DAX')
    
    soup=bs.BeautifulSoup(resp.text,'lxml') 
    table=soup.find('table',{'class': 'wikitable sortable'})
    
    tickers=[]
    
    for row in table.findAll('tr')[1:]:
        ticker=row.findAll('td')[1].text.split('>')[0]
        tickers.append(ticker)
    
    with open('daxtickers.pickle','wb') as f:
        pickle.dump(tickers,f)
        
    return(tickers)


def get_dax_from_yahoo():
    '''Create a directory and save historical OHLC stock data for each ticker
       symbol from the daxtickers.pickle object for the last 10 years.
       
       Returns
       -------
       .csv files
           containing OHLC prices for each ticker
        
    '''
    
    #load .pkl file
    with open('daxtickers.pickle','rb') as f:
        tickers=pickle.load(f)
        
    #create directory for csv files        
    if not os.path.exists('dax_stock_dfs'):
        os.makedirs('dax_stock_dfs')
        
    #static settings (change as you like) 
    start=dt.datetime(2010,1,1)   
    end=dt.datetime.now()
    
    #go through each ticker and retrieve data
    for ticker in tqdm(tickers):
        if not os.path.exists('dax_stock_dfs/{}_DE.csv'.format(ticker)):
            try:
                #NOTE: adding .DE to the ticker symbol to look for german
                #stocks!
                df=web.DataReader(ticker+'.DE','yahoo',start, end)
                df.to_csv('dax_stock_dfs/{}.csv'.format(ticker+'_DE'))                
            except:
                continue
        else:
            print('Already have {}'.format(ticker))
            
        
def compile_dax():
    '''For each ticker symbol in daxtickers.pickle, open the respective csv
       file with the OHLC prices and conbine their adjusted closing prices into
       one single dataframe.
       
       Returns
       -------
       
       dataframe
           the joined closing prices
    '''
    
    #load ticker .pkl file
    with open('daxtickers.pickle','rb') as f:
        tickers=pickle.load(f)
    
    #initialize main dataframe   
    main_df=pd.DataFrame()
    
    for ticker in tqdm(tickers):
        try:
            #load csv file for each ticker
            df=pd.read_csv('dax_stock_dfs/{}_DE.csv'.format(ticker))
            df.set_index('Date',inplace=True)
            
            #rename the column with adjusted closing prices into the ticker
            #and drop all other columns
            df.rename(columns={'Adj Close': ticker},inplace=True)
            df.drop(['Open','High','Low','Close','Volume'],axis=1,inplace=True)
            
            #add the remaining column to the main dataframe 
            if main_df.empty:
                main_df=df
            else:
                main_df=main_df.join(df,how='outer')
        except:
            continue
    #save dataframe as csv file    
    main_df.to_csv('dax_joined_closes.csv')     
    

def visualize_data(data='dax_joined_closes.csv'):
    '''Visualization of cross correlation matrix for the adjusted closing
       prices of the german DAX index as a heatmap plot.
        
       The color scale RdYlGn has been used to indicate positive (green),
       negative (red) and no correlation (yellow). 
       
    '''
    
    #load the dataframe of joined closing prices
    df=pd.read_csv(data)
    #compute corellation matrix
    df_corr=df.corr()
    data=df_corr.values
    
    #the following codes deals with methods for proper visualization of the 
    #above correlation matrix    
    fig=plt.figure(figsize=(15,13))
    ax=fig.add_subplot()
    
    heatmap=ax.pcolor(data,cmap=plt.cm.get_cmap('RdYlGn'))
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1])+0.5,minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5,minor=False)
    #change axes orientation for better overview
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    #set names for each row/column of the matrix
    column_labels=df_corr.columns
    row_labels=df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    #set proper limits on the colorbar (correlation from -1 to 1) 
    heatmap.set_clim(-1,1)
    #beautification: apply tight layout
    plt.tight_layout()    
    
    plt.show()      