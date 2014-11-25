
# -*- coding: utf-8 -*-
CONST_BBANDS_OUT_TRESHOLD=0
CONST_BBANDS_PERIOD=20
CONST_RSI_PERIOD=14
CONST_AROON_PERIOD=14
"""
IMPORT FILE TO PANDA
"""

import pandas as pd
FILENAME = 'dax30_m1_testdata.csv'

df=pd.read_csv(FILENAME,nrows=500,header=None,names=['tag','minute','open','high','low','close','num_ticks'],
               verbose=True,parse_dates=False)
 

#Test for nulls
pd.isnull(df)
df.dtypes
df.groupby('tag').minute.count()
 

#change data type
df.columns
df.dtypes

df['zeit']=df['tag']  + " " +  df['minute']
df['zeit']=pd.to_datetime(df['zeit'])
df.zeit
df.head()
df['tag']=df['zeit']
df=df.drop('minute',axis=1)
df=df.drop('zeit',axis=1)
df


"""
CONVERT TO ARRAY AND CALCULATE INDICATORS
"""
import numpy as np
import talib as ta
ta.get_functions()
 
close_price=np.array(df['close'])

df['bbands_upper'], df['bbands_middle'], df['bbands_lower']= ta.BBANDS(
        close_price, 
        timeperiod=CONST_BBANDS_PERIOD,
         nbdevup=2,
         nbdevdn=2,
         matype=0)
#df['rsi']= ta.RSI(close_price,CONST_BBANDS_PERIOD)
#df['aroon_up'],df['aroon_down']=ta.AROON(np.array(df['high']),np.array(df['low']), timeperiod=CONST_AROON_PERIOD)



##Filter array
df=df[df['tag'] <'2014-10-31']
df.count()
#-> 369 rows

# if high > bbands_upper  then bbands_out_up
# if low < bbands_lower  then bbands_out_down
df['bb_out']=False 
df['bb_out_up']=False 
df['bb_out_down']=False 

df.loc[df['high']>(df['bbands_upper'] + CONST_BBANDS_OUT_TRESHOLD),'bb_out'] = True
df.loc[df['low']<(df['bbands_lower'] - CONST_BBANDS_OUT_TRESHOLD),'bb_out'] = True

df.loc[df['high']>(df['bbands_upper'] + CONST_BBANDS_OUT_TRESHOLD),'bb_out_up'] = True
df.loc[df['low']<(df['bbands_lower'] - CONST_BBANDS_OUT_TRESHOLD),'bb_out_down'] = True
 
df.groupby('bb_out').count()    #285 = false, 84 True
df.groupby('bb_out_up').count() #314 = false, 55 True

 

"""
CALCULATE START TRADE
""" 
#The trade starts after the first candle back in bbands (Open_price)
df['sl_kurs']=0.0
df['vp_kurs']=0.0

def set_sl(i): # sets stop loss in panda
    print "step into  function set_sl"
    j=i+1  #start trade on next candl
    kurs_diff= (df.bbands_upper[j]-df.bbands_middle[j]) * 0.95
    print kurs_diff
    if i < df.index.max() and df.bb_out_up[i-1] == True:
        print "break out upper band"
        df.signal[j]='start trade'
        df.target_direction[j]='down'
        df.sl_kurs[j]=df.open[j] + kurs_diff
        df.vp_kurs[j]=df.open[j] - kurs_diff
    elif i < df.index.max() and df.bb_out_down[i-1] == True:
        print "break out lower band"
        df.signal[j]='start trade'
        df.target_direction[j]='up'
        df.sl_kurs[j]=df.open[j] - kurs_diff
        df.vp_kurs[j]=df.open[j] + kurs_diff

 
#determine first back again after break out
df['signal']=''
df['target_direction']=''
df['trade_nr']=0
count=1

for i in df.index:
    if df.bb_out[i] == False and i > 0 and df.bb_out[i-1] == True:
        #first candle back in bbands
       df.signal[i]='prepare trade'         
       df.trade_nr[i]=count
       start_trade= True
       
       #calculate sl_kurs and vp_kurs on basis of open price of next (!) candle       
       set_sl(i)        
       print i  , count
       count+=1

#0:90 -> 5 Trades: 26, 48, 58, 72, 89 (prepare Trade)
 
df.to_csv('dax30_m1_testdata_result.csv')
df[df['signal']=='start trade'].to_csv('dax30_m1_testdata_result_start_trade.csv')

df[df['signal']=='start trade']

 
"""
CHECK IF TRADE WOULD BE WON OR LOSS
"""
df[49:88]
df['won_loss']=''
df['start_trade']=0
#i_trade_ended_at_index=0 #used to have only one trade at a time

def set_won_loss(i):
    for j in df.index[i:]:
        # down
        if df.target_direction[i]=='down':
            if df.high[j] > df.sl_kurs[i]:
                df.won_loss[j]='loss'
                df.start_trade[j]=i
                break
            elif df.low[j] < df.vp_kurs[i]:
                df.won_loss[j]='won'
                df.start_trade[j]=i                
                break
        #up    
        if df.target_direction[i]=='up':
            if df.high[j] > df.vp_kurs[i]:
                df.won_loss[j]='won'
                df.start_trade[j]=i                
                break                
            elif df.low[j] < df.sl_kurs[i]:
                df.won_loss[j]='loss'
                df.start_trade[j]=i                
                break
    ##    i_trade_ended_at_index=j
                
                
    
for i in df.index:
    if df.signal[i]=='start trade':
  #      if i > i_trade_ended_at_index:
       print "for index ", i , " set won or loss"
       set_won_loss(i)
        

df[df['signal']=='start trade'].count()
df.groupby('won_loss').count('tag')

df[ df['won_loss']!='' ] 


### FLAG Parallel trades 
df['parallel_trade']=False
for i in df.index:
    for j in df.index[i:]:
        if 
 
    
########  TESTING ############
########  TESTING ############
########  TESTING ############
##nachfolger
df['prev']=df.tag.shift(-2)
df[40:60]

next(df.iterrows())[1]

for count, row in df.iterrows():
 print count 



"""
##chart
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc

from matplotlib.dates import DateFormatter, WeekdayLocator,\
     DayLocator, MONDAY

mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12


df['tag']=df['tag'].apply(lambda tag: mdates.date2num(tag))

fig,ax=plt.subplots()  
#fig.subplots_adjust(bottom=0.2)
#ax.xaxis.set_major_locator(mondays)
#ax.xaxis.set_minor_locator(alldays)
#ax.xaxis.set_major_formatter(weekFormatter)

csticks = candlestick_ohlc(ax,df[['tag', 'open', 'high', 'low', 'close']].values, width=0.0005)  

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show() 
  
 ##example1
import matplotlib.pyplot as plt 
plt.plot([1, 2, 3, 4],[4, 7, 8, 12])
plt.show() 


##example1
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick, candlestick2_ohlc
import matplotlib.dates as mdates
from pandas.io.data import DataReader 
# get daily stock price data from yahoo finance for S&P500
SP = DataReader("^GSPC", "yahoo")
SP.reset_index(inplace=True)
print(SP.columns)
SP['Date2'] = SP['Date'].apply(lambda date: mdates.date2num(date.to_pydatetime()))
fig, ax = plt.subplots()
SP.dtypes
csticks = candlestick(ax, SP[['Date2', 'Open', 'Close', 'High', 'Low']].values)
plt.show() 
 
#### use inbuild csv
#### 
import csv  
line_list=list(csv.reader(open(FILENAME)))
line_list[0:5]
 