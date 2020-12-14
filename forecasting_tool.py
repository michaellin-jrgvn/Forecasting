# Install Libraries (This may need to be done first each time the notebook is used here.  Takes a few minutes to install)
from IPython.display import clear_output
try:
  !pip install pystan
  !pip install fbprophet
except:
  pass
finally:
  clear_output()
  print('All Loaded')

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64
from io import BytesIO

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

import plotly.express as px

# Define holidays and events
tet_holiday_2021 = pd.DataFrame({
    'holiday': 'tet_holiday_2021',
    'ds': pd.date_range('2020-02-10',periods=7,freq='D'),
    'lower_window': -3,
    'upper_window':10,
    'prior_scale': 15
})
tet_holiday = pd.DataFrame({
    'holiday': 'tet_holiday',
    'ds': pd.date_range('2020-01-23',periods=6,freq='D'),
    'lower_window': -3,
    'upper_window':10,
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
    print('Preprocess data...')
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
def read_file():
    df = pd.read_csv('df_1820_1.csv')
    return df

st.set_page_config(
     page_title="Ballz",
     page_icon=":crystal_ball:",
     layout="wide",
     initial_sidebar_state="expanded",
)

st.title('Project CrystalBallz :crystal_ball:')
st.write('Project CrystalBallz helps you to see through the future. 2018-2020 data will be fitted to a forecasting algorithm to generate insight. Toggle the sidebar and follow the steps to generate forecast you need.')

df = read_file()
df_display = df.set_index('datetime')

st.sidebar.write('latest date of the current data set: ', pd.to_datetime(df_display.index[-1], format='%Y/%m/%d'))
st.sidebar.write('Please update your dataset if the data is not up-to-date.')

st.sidebar.subheader('1️⃣ - Forecast Date Range:')
forecast_start_date = st.sidebar.date_input('From:')
forecast_end_date = st.sidebar.date_input('To:') + + datetime.timedelta(days=1)

if forecast_start_date > forecast_end_date:
    st.sidebar.write('Warning: Forecast start date cannot be the same or later than the end date.')

st.sidebar.subheader('2️⃣ - Select Stores:')

@st.cache
def store_code(df):
    latest_store_list = df.store_code.unique().tolist()
    # Delete stores that has been closed down in 2020 that shouldn't exsit in Oct onwards
    store_del = ['E116','E117','E701','E120','D915']
    for s in store_del:
        latest_store_list.remove(s)
    # latest_store_list.extend(['D932','D423']) # Add Mia and Midori back to the list
    latest_store_list.sort()
    return latest_store_list

store_code_func = store_code(df)

store_select_option = st.sidebar.radio("Select options:", ('Individual/Multiple Stores', 'All Stores'))
if store_select_option == 'All Stores':
    st.sidebar.warning('⚡ The time requires to forecast all store could take up to 3 hours')
    store_code = store_code_func
else:
    store_code = st.sidebar.multiselect('Select store code to be forecasted:', store_code_func)

st.sidebar.subheader("3️⃣ - Ready for magic 🍄?")
if st.sidebar.button('Generate Forecast'):
    st.write('Inspect and check your data in both tabular and graphical formats')

    @st.cache(suppress_st_warning=True)
    def fit_pred_model(store_code):
        m_list = {}
        final = pd.DataFrame()
        with st.spinner('Wait for it...'):
            for i, code in enumerate(store_code):
                print('fitting shop code: ', code, i+1, '/96 stores')
                m = fit_model(df, code, holidays, 'all')
                m_list[code] = m
                print(m_list)
                forecast = predict_model(m, forecast_start_date,forecast_end_date, 'H')
                shop_yhat = forecast[['ds','yhat']]
                shop_yhat = shop_yhat.rename(columns={'yhat': code})
                final = pd.merge(final, shop_yhat.set_index('ds'), how='outer', left_index=True, right_index=True)
        return final

    final = fit_pred_model(store_code)
    fig = px.bar(final, x=final.index, y=final.columns)
    st.plotly_chart(fig, use_container_width=True)
    st.balloons()

    st.dataframe(final)

    def to_excel(df):

        # Generate dataframe with different time samples
        df_d = df.resample('D').sum().T
        df_d.loc['Total']= df_d.sum()

        df_w = df.resample('W').sum().T
        df_w.loc['Total']= df_w.sum()

        df_m = df.resample('M').sum().T
        df_m.loc['Total']= df_m.sum()

        df_summary = df_m.loc['Total']
        df_h = df.T

        # Set up Excel file writer
        output = BytesIO()
        writer = pd.ExcelWriter(output)

        # Write each dataframe to a different worksheet.
        df_h.to_excel(writer, sheet_name='By hour')
        df_d.to_excel(writer, sheet_name='By day')
        df_w.to_excel(writer, sheet_name='By week')
        df_m.to_excel(writer, sheet_name='By month')
        df_summary.to_excel(writer, sheet_name='Summary')

        writer.save()
        processed_data = output.getvalue()
        return processed_data

    def get_table_download_link(df):
        val = to_excel(df)
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download file</a>' # decode b'abc' => abc

    st.subheader('Step 4 - Download Data')
    st.write('Click the link below to download the data for your own use:')
    st.markdown(get_table_download_link(final), unsafe_allow_html=True)