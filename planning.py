import streamlit as st
import pandas as pd
import numpy as np
import datetime 
import base64
from io import BytesIO
import os

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.seasonal import seasonal_decompose
from col_functions import read_col_files, filtered_data_merged, data_filter, store_code, regression_table

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
covid_wave_1 = pd.DataFrame({
    'holiday': 'covid_lockdown',
    'ds': pd.date_range('2020-04-01',periods=22,freq='D'),
    'lower_window': 0,
    'upper_window': 2,
})
covid_wave_2 = pd.DataFrame({
    'holiday': 'covid_lockdown',
    'ds': pd.date_range('2020-07-24',periods=30,freq='D'),
    'lower_window': 0,
    'upper_window': 7,
})
covid_wave_3 = pd.DataFrame({
    'holiday': 'covid_lockdown',
    'ds': pd.date_range('2020-11-29',periods=26,freq='D'),
    'lower_window': 0,
    'upper_window': 7,
})
covid_wave_4 = pd.DataFrame({
    'holiday': 'covid_lockdown',
    'ds': pd.date_range('2021-01-28',periods=58,freq='D'),
    'lower_window': 0,
    'upper_window': 7,
})
school_holiday = pd.DataFrame({
    'holiday': 'school_breaks',
    'ds': pd.date_range('2020-07-11',periods=19,freq='D'),
    'lower_window': 0,
    'upper_window': 0,    
})

holidays = pd.concat((tet_holiday,national_holidays,observance_2020, observance_2019, observance_2018,women_day,v_day, covid_wave_1, covid_wave_2, covid_wave_3, covid_wave_4,school_holiday, teachers_day))

# Prophet model fitting function
@st.cache
def fit_model(df, holidays,resample, mode):
    # Initial model set up
    m = Prophet(
        growth='linear',
        seasonality_mode=mode,
        holidays=holidays,
        uncertainty_samples=100,
        daily_seasonality=False
    )
    m.add_country_holidays(country_name='VN')
    if resample == 'H':
        m.add_seasonality('sundays', period=1, prior_scale=10, fourier_order=15, mode=mode, condition_name='is_sunday')
        m.add_seasonality('mondays', period=1, prior_scale=1.5, fourier_order=15, mode=mode, condition_name='is_monday')
        m.add_seasonality('tuesdays', period=1, prior_scale=1.5, fourier_order=15, mode=mode, condition_name='is_tuesday')
        m.add_seasonality('wednesdays', period=1, prior_scale=1.5, fourier_order=15, mode=mode, condition_name='is_wednesday')
        m.add_seasonality('thursdays', period=1, prior_scale=1.5, fourier_order=15, mode=mode, condition_name='is_thursday')
        m.add_seasonality('fridays', period=1, prior_scale=1.5, fourier_order=15, mode=mode, condition_name='is_friday')
        m.add_seasonality('saturday', period=1, prior_scale=5, fourier_order=15, mode=mode, condition_name='is_saturday')

    df.columns = ['ds','y']
    df['ds'] = pd.to_datetime(df['ds'])
    
    if resample == 'H':
        df = df[(df['ds'].dt.hour > 9) & (df['ds'].dt.hour < 22)]
    df = df[df['y'] > 0]
    
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
    df['is_sunday'] = df['ds'].apply(is_sunday)
    df['is_monday'] = df['ds'].apply(is_monday)
    df['is_tuesday'] = df['ds'].apply(is_tuesday)
    df['is_wednesday'] = df['ds'].apply(is_wednesday)
    df['is_thursday'] = df['ds'].apply(is_thursday)
    df['is_friday'] = df['ds'].apply(is_friday)
    df['is_saturday'] = df['ds'].apply(is_saturday)
    
    # Fit model
    m.fit(df)
    return m

#@st.cache
#def store_code():
#    store_codes = [os.path.splitext(f)[0] for f in os.listdir('./data/')]
#    return store_codes

# Prophet predict function that. Prerequisite: Fitted model
@st.cache
def predict_model(m, start,end, freq):
    future = pd.DataFrame({'ds': pd.date_range(start=start, end=end, freq=freq)})
    if freq == 'H':
        future = future[(future['ds'].dt.hour > 9) & (future['ds'].dt.hour < 22)]

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

    future['is_sunday'] = future['ds'].apply(is_sunday)
    future['is_monday'] = future['ds'].apply(is_monday)
    future['is_tuesday'] = future['ds'].apply(is_tuesday)
    future['is_wednesday'] = future['ds'].apply(is_wednesday)
    future['is_thursday'] = future['ds'].apply(is_thursday)
    future['is_friday'] = future['ds'].apply(is_friday)
    future['is_saturday'] = future['ds'].apply(is_saturday)
    forecast = m.predict(future)
    #forecast[['yhat_lower','yhat']] = forecast[['yhat_lower','yhat']].clip(lower=0)
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

    # Fit and predict using Prophet Model
    fit_m = fit_model(df_transform, holidays, select_resample, 'additive')
    pred_m = predict_model(fit_m, forecast_date , forecast_date + datetime.timedelta(days=1), select_resample)

    pred_m = pred_m.set_index('ds')
    pred_m[['yhat_lower_f','yhat_upper_f','y_final']] = inv_boxcox(pred_m[['yhat_lower','yhat_upper','yhat']], transform_lambda)

    # Setup empty dataframe for future merge
    df_full['ds'] = pred_m.index
    df_full = df_full.set_index('ds')

    df['{}'.format(df_display)] = pred_m[['yhat_lower_f','yhat_upper_f','y_final']]
    df['{}'.format(df_display)].columns = [['{}_y_lower'.format(df_display), '{}_y_upper'.format(df_display), '{}_y_final'.format(df_display)]]
    df_full = pd.concat([df_full, df['{}'.format(df_display)]],axis=1)

sales_col_opt = df_full.filter(like='Sales_y_final')
sales_col_opt.columns = ['Dinein','Pickup','Delivery']
sales_col_opt['Total Sales'] = sales_col_opt.sum(axis=1)
plt_sales = px.line(sales_col_opt,x=sales_col_opt.index, y='Total Sales')
st.plotly_chart(plt_sales,use_container_width=True)
forecast_daily_sales = int(sales_col_opt['Total Sales'].sum())
st.write('Total Sales of the day is: ', forecast_daily_sales)

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