import pandas as pd
import numpy as np
import datetime 
import streamlit as st

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

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
def fit_model(df, resample, mode):
    # Initial model set up
    m = Prophet(
        growth='linear',
        seasonality_mode=mode,
        uncertainty_samples=100,
        daily_seasonality=True
    )
    m.add_country_holidays(country_name='VN')
    if resample == 'H':
        m.add_seasonality('sundays', period=1, prior_scale=10, fourier_order=2, mode=mode, condition_name='is_sunday')
        m.add_seasonality('mondays', period=1, prior_scale=1.5, fourier_order=2, mode=mode, condition_name='is_monday')
        m.add_seasonality('tuesdays', period=1, prior_scale=1.5, fourier_order=2, mode=mode, condition_name='is_tuesday')
        m.add_seasonality('wednesdays', period=1, prior_scale=1.5, fourier_order=2, mode=mode, condition_name='is_wednesday')
        m.add_seasonality('thursdays', period=1, prior_scale=1.5, fourier_order=2, mode=mode, condition_name='is_thursday')
        m.add_seasonality('fridays', period=1, prior_scale=1.5, fourier_order=2, mode=mode, condition_name='is_friday')
        m.add_seasonality('saturday', period=1, prior_scale=5, fourier_order=2, mode=mode, condition_name='is_saturday')

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


# Prophet predict function that. Prerequisite: Fitted model
@st.cache()

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