import streamlit as st
import pandas as pd
import numpy as np
import datetime 
import base64
from io import BytesIO
import os

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose


# Define holidays and events
tet_holiday_2021 = pd.DataFrame({
    'holiday': 'tet_holiday_2021',
    'ds': pd.date_range('2021-02-10',periods=7,freq='D'),
    'lower_window': -5,
    'upper_window':4,
    'prior_scale': 15
})
tet_holiday = pd.DataFrame({
    'holiday': 'tet_holiday',
    'ds': pd.date_range('2020-01-23',periods=6,freq='D'),
    'lower_window': -5,
    'upper_window':4,
    'prior_scale': 15
})
women_day = pd.DataFrame({
    'holiday':'women_day',
    'ds': pd.to_datetime(['2018-10-20','2019-10-20','2020-10-20','2021-10-21']),
    'lower_window':0,
    'upper_window':0,
    'prior_scale': 5
})
v_day = pd.DataFrame({
    'holiday':'v_day',
    'ds': pd.to_datetime(['2018-02-14','2019-02-14','2020-02-14','2021-02-14']),
    'lower_window':-2,
    'upper_window':0,
    'prior_scale': 5
})
teachers_day = pd.DataFrame({
    'holiday': 'teachers_day',
    'ds': pd.to_datetime(['2018-11-20','2019-11-20','2020-11-20','2021-11-21']),
    'lower_window':-1,
    'upper_window': 1,
    'prior_scale': 5
})
national_holidays = pd.DataFrame({
    'holiday': 'national_holiday',
    'ds': pd.to_datetime(['2019-01-01','2019-04-02','2019-04-30','2019-05-01','2019-09-02','2020-01-01','2020-04-02','2020-04-30','2020-05-01','2020-09-02','2021-01-01','2021-04-02','2021-04-30','2021-05-01','2021-09-02']),
    'lower_window': 0,
    'upper_window': 0,
})
observance_2020 = pd.DataFrame({
    'holiday': 'observance_2020',
    'ds': pd.to_datetime(['2020-03-08','2020-05-09','2020-05-31','2020-06-01','2020-06-20','2020-06-28','2020-10-01','2020-10-20','2020-10-31','2020-12-24','2020-12-25','2020-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
observance_2021 = pd.DataFrame({
    'holiday': 'observance_2021',
    'ds': pd.to_datetime(['2021-03-08','2021-05-09','2021-05-31','2021-06-01','2021-06-20','2021-06-28','2021-10-01','2021-10-20','2021-10-31','2021-12-24','2021-12-25','2021-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
observance_2019 = pd.DataFrame({
    'holiday': 'observance_2019',
    'ds': pd.to_datetime(['2019-03-08','2019-05-09','2019-05-31','2019-06-01','2019-06-20','2019-06-28','2019-10-01','2019-10-20','2019-10-31','2019-12-24','2019-12-25','2019-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
observance_2018 = pd.DataFrame({
    'holiday': 'observance_2018',
    'ds': pd.to_datetime(['2018-03-08','2018-05-09','2018-05-31','2018-06-01','2018-06-20','2018-06-28','2018-10-01','2018-10-20','2018-10-31','2018-12-24','2018-12-25','2018-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
covid_lockdown = pd.DataFrame({
    'holiday': 'covid_lockdown',
    'ds': pd.date_range('2020-04-01',periods=23,freq='D'),
    'lower_window': 0,
    'upper_window': 2,
})
school_holiday = pd.DataFrame({
    'holiday': 'school_breaks',
    'ds': pd.date_range('2020-07-11',periods=19,freq='D'),
    'lower_window': 0,
    'upper_window': 0,    
})

holidays = pd.concat((tet_holiday,national_holidays,observance_2020, observance_2019, observance_2018,women_day,v_day, covid_lockdown, school_holiday, teachers_day))

# Prophet model fitting function
@st.cache
def fit_model(df, store_code, holidays, channel):
    # Initial model set up
    m = Prophet(
        growth='linear',
        seasonality_mode='multiplicative',
        holidays=holidays,
        uncertainty_samples=100,
        daily_seasonality=False
    )
    m.add_country_holidays(country_name='VN')
    m.train_holiday_names
    m.add_seasonality('sundays', period=1, prior_scale=10, fourier_order=10, mode='multiplicative', condition_name='is_sunday')
    m.add_seasonality('mondays', period=1, prior_scale=1.5, fourier_order=10, mode='multiplicative', condition_name='is_monday')
    m.add_seasonality('tuesdays', period=1, prior_scale=1.5, fourier_order=10, mode='multiplicative', condition_name='is_tuesday')
    m.add_seasonality('wednesdays', period=1, prior_scale=1.5, fourier_order=10, mode='multiplicative', condition_name='is_wednesday')
    m.add_seasonality('thursdays', period=1, prior_scale=1.5, fourier_order=10, mode='multiplicative', condition_name='is_thursday')
    m.add_seasonality('fridays', period=1, prior_scale=1.5, fourier_order=10, mode='multiplicative', condition_name='is_friday')
    m.add_seasonality('saturday', period=1, prior_scale=5, fourier_order=10, mode='multiplicative', condition_name='is_saturday')

    # Preprocess data
    if channel != 'all':
        df_test = df[(df.store_code == store_code) & (df.channel == channel)][['datetime','bill_size']]
    else:
        df_test = df[(df.store_code == store_code)][['datetime','bill_size']]

    df_test.columns = ['ds','y']
    df_test['ds'] = pd.to_datetime(df_test['ds'])
    df_test.set_index('ds',inplace=True)
    df_hr = df_test.resample('H').sum().reset_index()
    df_hr_o = df_hr[(df_hr['ds'].dt.hour > 9) & (df_hr['ds'].dt.hour < 22)]
    df_hr_o = df_hr[df_hr['y'] > 0]
    
    # define seasonality
    def is_sunday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 6
    def is_monday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 0
    def is_tuesday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 1
    def is_wednesday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 2
    def is_thursday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 3
    def is_friday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 4
    def is_saturday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 5
    df_hr_o['is_sunday'] = df_hr_o['ds'].apply(is_sunday)
    df_hr_o['is_monday'] = df_hr_o['ds'].apply(is_monday)
    df_hr_o['is_tuesday'] = df_hr_o['ds'].apply(is_tuesday)
    df_hr_o['is_wednesday'] = df_hr_o['ds'].apply(is_wednesday)
    df_hr_o['is_thursday'] = df_hr_o['ds'].apply(is_thursday)
    df_hr_o['is_friday'] = df_hr_o['ds'].apply(is_friday)
    df_hr_o['is_saturday'] = df_hr_o['ds'].apply(is_saturday)
    
    # Fit model
    m.fit(df_hr_o)
    return m

@st.cache
def store_code():
    df = pd.read_excel('./data/Tracking store by year.xls', nrows=100,usecols=['Store Code','Store', 'AC', 'Region','Province','Concept','Opening Date'],parse_dates=['Opening Date'])

    df['full_name'] = df['Store Code'] + '-' + df['Store']
    return df

#@st.cache
#def store_code():
#    store_codes = [os.path.splitext(f)[0] for f in os.listdir('./data/')]
#    return store_codes

# Prophet predict function that. Prerequisite: Fitted model
@st.cache
def predict_model(m, start,end, freq):
    future = pd.DataFrame({'ds': pd.date_range(start=start, end=end, freq=freq)})
    future_o = future[(future['ds'].dt.hour > 9) & (future['ds'].dt.hour < 22)]

    # define seasonality
    def is_sunday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 6
    def is_monday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 0
    def is_tuesday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 1
    def is_wednesday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 2
    def is_thursday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 3
    def is_friday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 4
    def is_saturday(ds):
        date = pd.to_datetime(ds)
        return ds.weekday() == 5

    future_o['is_sunday'] = future_o['ds'].apply(is_sunday)
    future_o['is_monday'] = future_o['ds'].apply(is_monday)
    future_o['is_tuesday'] = future_o['ds'].apply(is_tuesday)
    future_o['is_wednesday'] = future_o['ds'].apply(is_wednesday)
    future_o['is_thursday'] = future_o['ds'].apply(is_thursday)
    future_o['is_friday'] = future_o['ds'].apply(is_friday)
    future_o['is_saturday'] = future_o['ds'].apply(is_saturday)

    forecast = m.predict(future_o)
    return forecast

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
select_resample = st.sidebar.selectbox('Select Resample Period', ['D','W','M','30min'])

# Split pickup by TC, sales and TA by resample option
channels_split_df['Pickup - TC'] = resample_tc(select_resample,channels_df['Pickup'])
channels_split_df['Pickup - Sales'] = resample_sales(select_resample,channels_df['Pickup'])
channels_split_df['Pickup - TA'] = resample_ta(select_resample,channels_split_df['Pickup - Sales'],channels_split_df['Pickup - TC'])

# Split delivery by TC, sales and TA by resample option
channels_split_df['Delivery - TC'] = resample_tc(select_resample, channels_df['Delivery'])
channels_split_df['Delivery - Sales'] = resample_sales(select_resample, channels_df['Delivery'])
channels_split_df['Delivery - TA'] = resample_ta(select_resample, channels_split_df['Delivery - Sales'], channels_split_df['Delivery - TC'])

df_display = st.sidebar.selectbox('Select dataset to display', list(channels_split_df))
select_decompose = st.sidebar.checkbox('Split into Time Series Components')

if select_decompose:
    # Plot seasonal decompose result
    decompose_result = seasonal_decompose(channels_split_df[df_display],model='additive', extrapolate_trend='freq')
    decompose_fig = make_subplots(rows=4, cols=1)
    decompose_fig.add_trace(go.Scatter(x=decompose_result.observed.index,y=decompose_result.observed,name="Observed Data"),row=1, col=1)
    decompose_fig.add_trace(go.Scatter(x=decompose_result.trend.index,y=decompose_result.trend,name="Trend"),row=2, col=1)
    decompose_fig.add_trace(go.Scatter(x=decompose_result.seasonal.index,y=decompose_result.seasonal,name="Seasonality"),row=3,col=1)
    decompose_fig.add_trace(go.Bar(x=decompose_result.resid.index,y=decompose_result.resid,name="Residual"),row=4,col=1)
    decompose_fig.update_layout(height=600)
    st.plotly_chart(decompose_fig, use_container_width=True)
else:
    channels_plot = px.line(channels_split_df[df_display])
    channels_plot.update_layout(height=300)
    st.plotly_chart(channels_plot)