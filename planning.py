import streamlit as st
import pandas as pd
import numpy as np
import datetime 
import base64
from io import BytesIO
import os

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from scipy.stats import boxcox, truncnorm
from scipy.special import inv_boxcox

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose
from col_functions import read_col_files, filtered_data_merged, data_filter, store_code, regression_table
from prophet_model import fit_model, predict_model

import random

# Read Dataset
@st.cache
def read_file(store_code):
    try:
        df = pd.read_csv('./data/' + store_code + '.csv', parse_dates=['datetime'])
        df = df.drop(['Unnamed: 0'], axis=1)
        df = df.set_index('datetime')
    except:
        print(store_code + 'file is not available')
        pass
    return df


# Format_func - Retrieve store code from select_box
def get_store_info(option):
    store_name = store_info[store_info['Store Code'] == option]
    return list(store_name.full_name)

def remove_closing_hours(df):
    ''' Remove closing hours from 10pm - 9:59am'''
    return df[(df.index.hour > 9) & (df.index.hour < 22)] 

# Aggregate dataset of all channels to 30mins
@st.cache
def resample_tc(resample, df):
    ''' Generate df of aggregated 30mins with tc and remove closing hours '''
    # Process dataset for TC
    all_channels_tc_df = df.resample(resample).count().bill_size
    all_channels_tc_df.fillna(0,inplace=True)
    # all_channels_tc_df = remove_closing_hours(all_channels_tc_df)
    return all_channels_tc_df

@st.cache
def resample_sales(resample, df):
    ''' Resample sales and removing closing hours '''
    # Process dataset for total sales
    all_channels_df = df.resample(resample).sum().bill_size
    all_channels_df.fillna(0,inplace=True)
    #all_channels_df = remove_closing_hours(all_channels_df)
    return all_channels_df

@st.cache
def resample_ta(resample, sales, tc):
    ''' Reample ta and removing closing hours'''
    # Process dataset for TA
    all_channels_ta_df = sales.div(tc)
    all_channels_ta_df.fillna(0,inplace=True)
    #all_channels_ta_df = remove_closing_hours(all_channels_ta_df)
    return all_channels_ta_df

# Retreive store info
store_info = store_code()

# Select box - choosing store to retrieve dataset
selected_store = st.sidebar.selectbox('Select store',store_info['Store Code'], format_func=get_store_info)
selected_store_df = read_file(selected_store)

# Create a dictionary for resample dataset by channels
channel_list = selected_store_df.channel.unique()
channels_df = {}
for channel in channel_list:
    channels_df['{}'.format(channel)] = selected_store_df[selected_store_df.channel == channel]

# Set up to split and resample df by channels
channels_split_df = {}
select_resample = 'H'
#select_resample = st.sidebar.selectbox('Select Resample Period', ['D','W','M','H','30min'])

# Split pickup by TC, sales and TA by resample option
channels_split_df['Dinein - TC'] = resample_tc(select_resample,channels_df['Dinein'])
channels_split_df['Dinein - Sales'] = resample_sales(select_resample,channels_df['Dinein'])
channels_split_df['Dinein - TA'] = resample_ta(select_resample,channels_split_df['Dinein - Sales'],channels_split_df['Dinein - TC'])

# Split pickup by TC, sales and TA by resample option
channels_split_df['Pickup - TC'] = resample_tc(select_resample,channels_df['Pickup'])
channels_split_df['Pickup - Sales'] = resample_sales(select_resample,channels_df['Pickup'])
channels_split_df['Pickup - TA'] = resample_ta(select_resample,channels_split_df['Pickup - Sales'],channels_split_df['Pickup - TC'])

# Split delivery by TC, sales and TA by resample option
channels_split_df['Delivery - TC'] = resample_tc(select_resample, channels_df['Delivery'])
channels_split_df['Delivery - Sales'] = resample_sales(select_resample, channels_df['Delivery'])
channels_split_df['Delivery - TA'] = resample_ta(select_resample, channels_split_df['Delivery - Sales'], channels_split_df['Delivery - TC'])

#df_display = st.sidebar.selectbox('Select dataset to display', list(channels_split_df))
display_details = st.sidebar.checkbox('display forecast details')

# Set up date-picker
forecast_date = st.sidebar.date_input('Select Forecast Range',min_value=channels_split_df['Dinein - Sales'].index.max())

# Setting up sales Forecast by channel, by TC, by TA
df = {}
df_full = pd.DataFrame()
for df_display in list(channels_split_df):

    # Preprocess and transform dataframe
    df_fit = channels_split_df[df_display].reset_index()
    df_fit = df_fit.rename(columns={'datetime':'ds','bill_size':'y_original'})
    df_fit = df_fit[df_fit['y_original'] > 0]
    df_fit = df_fit.set_index('ds')
    df_fit['y'], transform_lambda = boxcox(df_fit['y_original'])
    
    # Get dataframe ready to fit Prophet
    df_fit = df_fit.reset_index()
    df_transform = df_fit[['ds','y']]

    # Fit and predict using FB Prophet Model
    fit_m = fit_model(df_transform, select_resample, 'multiplicative')
    pred_m = predict_model(fit_m, forecast_date , forecast_date + datetime.timedelta(days=1), select_resample)

    pred_m = pred_m.set_index('ds')
    pred_m[['yhat_lower_f','yhat_upper_f','y_final']] = inv_boxcox(pred_m[['yhat_lower','yhat_upper','yhat']], transform_lambda)

    # Setup empty dataframe for future merge
    df_full['ds'] = pred_m.index
    df_full = df_full.set_index('ds')

    df['{}'.format(df_display)] = pred_m[['yhat_lower_f','yhat_upper_f','y_final']]
    df['{}'.format(df_display)] = df['{}'.format(df_display)].rename(columns={'yhat_lower_f':'{}_y_lower'.format(df_display), 'yhat_upper_f':'{}_y_upper'.format(df_display), 'y_final':'{}_y_final'.format(df_display)})
    df_full = pd.concat([df_full, df['{}'.format(df_display)]],axis=1)

sales_col_opt = df_full.filter(like='Sales_y_final')
sales_col_opt.columns = ['Dinein','Pickup','Delivery']
sales_col_opt['Total Sales'] = sales_col_opt.sum(axis=1)
plt_sales = px.line(sales_col_opt,x=sales_col_opt.index, y='Total Sales')
st.plotly_chart(plt_sales,use_container_width=True)
forecast_daily_sales = int(sales_col_opt['Total Sales'].sum())
st.write('Total Sales of the day is: ', forecast_daily_sales)

## Assign random minutes for each row
# Preprocessing

df['Dinein - TC'] = df['Dinein - TC'].round(0).astype('int')

# Setup to generate random TC using normal distribution
mu_tc  = df['Dinein - TC'][['Dinein - TC_y_final']].to_numpy()[:,0]
sigma_tc = df['Dinein - TC']['Dinein - TC_y_final'].std()
df['Dinein - TC']['simulate_TC'] = np.random.randint(df['Dinein - TC']['Dinein - TC_y_lower'],df['Dinein - TC']['Dinein - TC_y_upper'])

# Duplicate rows based on projected TC / hour
repeat_array = df['Dinein - TC'].reset_index()[['simulate_TC']].to_numpy()[:,0]
dinein_df = df['Dinein - TC'].loc[df['Dinein - TC'].index.repeat(repeat_array)]

# Assign random minutes for each ticket and sort according to time
dinein_df['minutes'] = np.random.randint(0, 59, dinein_df.shape[0])
dinein_df.index= dinein_df.index + pd.to_timedelta(dinein_df[['minutes']].to_numpy()[:,0], unit='m')
dinein_df = dinein_df.sort_index()

# Duplicate same number of row in ta dataframe as tc dataframe
dinein_ta_df = df['Dinein - TA'].loc[df['Dinein - TA'].index.repeat(repeat_array)]
dinein_ta_df.index = dinein_df.index

# Set up to generate random TA using normal Distribution
mu_ta = dinein_ta_df['Dinein - TA_y_final']
sigma_ta = dinein_ta_df['Dinein - TA_y_final'].std()
dinein_df['simulate_TA'] = np.random.normal(mu_ta,sigma_ta).clip(dinein_ta_df['Dinein - TA_y_lower'],None).round(0).astype(int)

st.write(dinein_df)
st.write('simulate dine-in sales: ', dinein_df['simulate_TA'].sum())
st.write('simulated dine-in tc: ', dinein_df['simulate_TA'].count())
st.write('simulated dine-in ta: ', dinein_df['simulate_TA'].sum()/dinein_df['simulate_TA'].count())


if display_details:

    # Plot normal chart without seasonal decompose
    channels_plot = px.line(channels_split_df[df_display])
    channels_plot.update_layout(height=300)
    st.plotly_chart(channels_plot)
    
    st.write(df_fit)
    
    # Display data distribution before and after transformation
    dist_plot = make_subplots(rows=2, cols=1)
    dist_plot.add_trace(go.Histogram(x=df_fit['y']),row=1,col=1)
    dist_plot.add_trace(go.Histogram(x=df_fit['y_original']), row=2,col=1)
    st.plotly_chart(dist_plot, use_container_width=True)

    # Display fitting, tends, seasonalities
    fig1 = fit_m.plot(pred_m)
    st.write(fig1)
    fig2 = fit_m.plot_components(pred_m)
    st.write(fig2)
    st.write(pred_m)

    def mape(actual, forecast):
        mape = abs((actual-forecast)/forecast) * 100
        return mape
    def mae(actual, forecast):
        mae = abs(actual-forecast)
        return mae
    pred_m = pred_m.reset_index()

    mape = mape(pred_m['y_final'].iloc[:len(df_fit)],df_fit['y_original']).dropna()
    mae = mae(pred_m['y_final'].iloc[:len(df_fit)],df_fit['y_original']).dropna()
    diff_df = pd.DataFrame(data=[df_fit['y_original'],pred_m['y_final']] ).T
    diff_df['mape'] = diff_df['y_final'].sub(diff_df['y_original']).div(diff_df['y_original']).mul(100)
    st.write(diff_df)
    st.write(mape.describe())
    st.write(mae.describe())


# Obtain COL data from COL_functions

col, trans = read_col_files()
filtered_data, filtered_trans = data_filter(col, trans, [selected_store])
df = filtered_data_merged(filtered_data, filtered_trans, store_info).sort_values(by='Date',ascending=False).set_index('Date')
df = df.iloc[-60:].reset_index()
df['dis_Date'] = df['Date'].apply(lambda x: x.strftime("%d %b, %Y"))
spmh_store_plt = px.scatter(df, x='Actual sales',y='Actual SPMH',color='Store Name',trendline='ols', hover_data=['dis_Date'])
st.plotly_chart(spmh_store_plt)
regression_table = regression_table(spmh_store_plt, 'Store Name')
st.write(regression_table)
forecast_spmh = regression_table.Gradient * forecast_daily_sales + regression_table['y-intercept']
st.write('Minimum SPMH from regression is: ', int(forecast_spmh))
manhour_allowed = forecast_daily_sales / forecast_spmh
st.write('Maximum manhour allowance from regression is:', int(manhour_allowed))