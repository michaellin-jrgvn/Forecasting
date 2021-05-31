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
    'ds': pd.to_datetime(['2020-03-08','2020-05-09','2020-05-31','2020-06-01','2020-10-01','2020-10-20','2020-10-31','2020-12-24','2020-12-25','2020-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
observance_2021 = pd.DataFrame({
    'holiday': 'observance_2021',
    'ds': pd.to_datetime(['2021-03-08','2021-05-09','2021-05-31','2021-06-01','2021-10-01','2021-10-20','2021-10-31','2021-12-24','2021-12-25','2021-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
observance_2019 = pd.DataFrame({
    'holiday': 'observance_2019',
    'ds': pd.to_datetime(['2019-03-08','2019-05-09','2019-05-31','2019-06-01','2019-10-01','2019-10-20','2019-10-31','2019-12-24','2019-12-25','2019-12-31']),
    'lower_window': 0,
    'upper_window': 0,
    'prior_scale': 3
})
observance_2018 = pd.DataFrame({
    'holiday': 'observance_2018',
    'ds': pd.to_datetime(['2018-03-08','2018-05-09','2018-05-31','2018-06-01','2018-10-01','2018-10-20','2018-10-31','2018-12-24','2018-12-25','2018-12-31']),
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

holidays = pd.concat((tet_holiday,national_holidays,observance_2021,observance_2020, observance_2019, observance_2018,women_day,v_day, covid_lockdown, school_holiday, teachers_day))

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

@st.cache
def keep_df(df):
    keep_df = df
    return keep_df

# Read Dataset
@st.cache
def read_file(store_code):
    try:
        df = pd.read_csv('./data/' + store_code + '.csv', parse_dates=['datetime'])
    except:
        print(store_code + 'file is not available')
        pass
    return df

st.set_page_config(
     page_title="Ballz",
     page_icon=":crystal_ball:",
     layout="centered",
     initial_sidebar_state="expanded",
)

#st.title('Project CrystalBallz :crystal_ball:')
#st.write('Project CrystalBallz helps you to see through the future. 2018-2020 data will be fitted to a forecasting algorithm to generate insight. Toggle the sidebar and follow the steps to generate forecast you need.')


df_display = read_file('D112').set_index('datetime')
st.sidebar.write('Latest date of the current data set: ', pd.to_datetime(df_display.index[-1], format='%Y/%m/%d'))
st.sidebar.write('Please update your dataset if the data is not up-to-date.')


# Expander - Forecast Generation Setting
with st.sidebar.beta_expander("Forecast Generator Setting", expanded=True):

    store_code_func = store_code()

    store_select_option = st.radio('1️⃣ - Select Stores:', ('by Individual/Multiple Stores','by AC','by Region','All Stores'))
    if store_select_option == 'All Stores':
        st.warning('⚡ The time requires to forecast all store could take up to 3 hours')
        store_code = store_code_func['Store Code']
    elif store_select_option == 'by Individual/Multiple Stores':
        store_selected = st.multiselect('Select store code to be forecasted:', store_code_func['full_name'])
        store_code = store_code_func[store_code_func['full_name'].isin(store_selected)]['Store Code']
    elif store_select_option == 'by AC':
        ac_selected = st.multiselect('Select AC area to be forecasted:', store_code_func['AC'].unique())
        store_code = store_code_func[store_code_func['AC'].isin(ac_selected)]['Store Code']
    elif store_select_option == 'by Region':
        region_selected = st.selectbox('Select the region to be forecasted:', store_code_func['Region'].unique())
        store_code = store_code_func[store_code_func['Region']== region_selected]['Store Code']

    start_date = datetime.date.today()
    end_date = datetime.date.today() + datetime.timedelta(days=1)
    forecast_date_range = st.date_input('2️⃣ - Forecast Date Range:', value=(start_date, end_date))
    
    st.subheader("3️⃣ - Ready for magic 🍄?")

    @st.cache(suppress_st_warning=True)
    def fit_pred_model(store_code):
        ''' Fit and predict model in a for loop '''
        m_list = {}
        forecast_by_store = {}
        final = pd.DataFrame()
        with st.spinner('Wait for it...'):
            for i, code in enumerate(store_code):
                try:
                    print('fitting shop code: ', code, i+1, '/96 stores')
                    df=read_file(code)
                    m = fit_model(df, code, holidays, 'all')
                    m_list[code] = m
                    forecast = predict_model(m, forecast_date_range[0],forecast_date_range[1] + datetime.timedelta(days=1), 'H')
                    forecast_by_store[code] = forecast
                    shop_yhat = forecast[['ds','yhat']]
                    shop_yhat = shop_yhat.rename(columns={'yhat': code})
                    final = pd.merge(final, shop_yhat.set_index('ds'), how='outer', left_index=True, right_index=True)
                except:
                    st.warning('No data is available for ' + store_code)
                    pass
        st.balloons()
        return final, df, forecast_by_store
    
    @st.cache(suppress_st_warning=True)
    def past_data(store_code,start_date,end_date):
        ''' Generate historical data for comparison '''
        df_2018 = pd.DataFrame()
        df_2019 = pd.DataFrame()
        df_lm = pd.DataFrame()
        for store in (store_code):
            store_opening = pd.to_datetime(store_code_func[store_code_func['Store Code'] == store]['Opening Date'])
            if (store_opening < (datetime.date.today() + pd.offsets.DateOffset(years=-1))).bool():
                df = read_file(store)
                df.datetime = pd.to_datetime(df.datetime,errors='ignore')
                try:
                    past_2018_df = df.set_index('datetime')
                    past_2018_df.index = pd.to_datetime(past_2018_df.index)
                    start_range = (forecast_date_range[0] + pd.offsets.DateOffset(years=-2)).strftime('%Y-%m-%d %H:%M:%S')
                    end_range= (forecast_date_range[1] + datetime.timedelta(days=1) + pd.offsets.DateOffset(years=-2)).strftime('%Y-%m-%d %H:%M:%S')
                    past_2018_df = past_2018_df.loc[start_range:end_range].resample('H').sum()
                    past_2018_df = past_2018_df['bill_size']
                    past_2018_df = past_2018_df.reset_index()
                    past_2018_df = past_2018_df.rename(columns={'datetime':'ds','bill_size': store})
                    df_2018 = pd.merge(df_2018, past_2018_df.set_index('ds'), how='outer', left_index=True, right_index=True)
                except:
                    st.warning('No data in 2018')
                    pass
                try:
                    past_2019_df = df.set_index('datetime')
                    past_2019_df.index = pd.to_datetime(past_2019_df.index)
                    start_range = (forecast_date_range[0] + pd.offsets.DateOffset(years=-1)).strftime('%Y-%m-%d %H:%M:%S')
                    end_range= (forecast_date_range[1] + datetime.timedelta(days=1) + pd.offsets.DateOffset(years=-1)).strftime('%Y-%m-%d %H:%M:%S')
                    past_2019_df = past_2019_df.loc[start_range:end_range].resample('H').sum()
                    past_2019_df = past_2019_df['bill_size']
                    past_2019_df = past_2019_df.reset_index()
                    past_2019_df = past_2019_df.rename(columns={'datetime':'ds','bill_size': store})
                    df_2019 = pd.merge(df_2019, past_2019_df.set_index('ds'), how='outer', left_index=True, right_index=True)
                except:
                    st.warning('No data in 2019')
                    pass

                try:
                    past_lm_df = df.set_index('datetime')
                    past_lm_df.index = pd.to_datetime(past_lm_df.index)
                    start_range = (forecast_date_range[0] + pd.offsets.DateOffset(months=-1)).strftime('%Y-%m-%d %H:%M:%S')
                    end_range = (forecast_date_range[1] + datetime.timedelta(days=1) + pd.offsets.DateOffset(months=-1)).strftime('%Y-%m-%d %H:%M:%S')
                    past_lm_df = past_lm_df.loc[start_range:end_range].resample('H').sum()
                    past_lm_df = past_lm_df['bill_size']
                    past_lm_df = past_lm_df.reset_index()
                    past_lm_df = past_lm_df.rename(columns={'datetime':'ds','bill_size': store})
                    df_lm = pd.merge(df_lm, past_lm_df.set_index('ds'), how='outer', left_index=True, right_index=True)
                except:
                    st.warning('No data in last month')
                    pass
            else:
                st.warning('insufficient data for '+ store_code)
        df_2018['total'] = df_2018.sum(axis=1)
        df_2019['total'] = df_2019.sum(axis=1)
        df_lm['total'] = df_lm.sum(axis=1)             
        return df_2018, df_2019, df_lm

    final, df_past, forecast_by_store = fit_pred_model(store_code)


final = final.loc[~(final<=0).all(axis=1)]
df_past = df_past[['datetime','bill_size']]
df_past = df_past.set_index('datetime')


if len(final) > 0:    
    # Expander - Forecast Filter and Fine tune
    with st.sidebar.beta_expander("Forecast Filter and Fine tuning", expanded=True):
        min_value = final.index.min().to_pydatetime()
        max_value = final.index.max().to_pydatetime()
        test = st.slider(label='fine tuning range', min_value=min_value,max_value=max_value,value=(min_value,max_value))
        st.sidebar.write(test)

    st.title('Generated Forecast')
    # Select box to display data rsampled by Hour, Day, Week, and Month
    resample_data =[['Hour','H'],['Day','D'],['Week','W'],['Month','M']]
    resample_df = pd.DataFrame(resample_data, columns=['name','id'])
    resample_values = resample_df['name'].tolist()
    resample_id = resample_df['id'].tolist()
    dic = dict(zip(resample_id,resample_values))
    data_resample_option = st.selectbox('Data resample by:',resample_id,format_func=lambda x:dic[x])
    final_edit = final.resample(data_resample_option).sum()
    final_edit['total'] = final_edit.sum(axis=1)

    keep_df(final)
    df_2018, df_2019, df_lm = past_data(store_code, start_date, end_date)


    def SSSG(df_past, df, resample):
        df_sssg = pd.DataFrame()

        if resample == 'D':
            df_past = df_past.resample('D').sum()
            df = df.resample('D').sum()
            df_past = df_past.reset_index()
            df = df.reset_index()
            df_past.ds = df_past.ds.dt.dayofyear
            df.ds = df.ds.dt.dayofyear  
            df_sssg = pd.merge(df_past,df, how='outer',left_on='ds',right_on='ds',suffixes=('(LY)','(Fcst)')) 
            df_sssg = df_sssg.set_index('ds')
            df_sssg['total'] = (df_sssg['total(Fcst)'].sub(df_sssg['total(LY)']).div(df_sssg['total(LY)'])).mul(100)
        if resample == 'W':
            df_past = df_past.resample('W').sum()
            df = df.resample('W').sum()
            df_past = df_past.reset_index()
            df = df.reset_index()
            df_past.ds = df_past.ds.dt.strftime('%U')
            df.ds = df.ds.dt.strftime('%U')  
            df_sssg = pd.merge(df_past,df, how='outer',left_on='ds',right_on='ds',suffixes=('(LY)','(Fcst)')) 
            df_sssg['total'] = (df_sssg['total(Fcst)'].sub(df_sssg['total(LY)']).div(df_sssg['total(LY)'])).mul(100)
        
        if resample == 'M':
            df_past = df_past.resample('M').sum()
            df = df.resample('M').sum()
            df_past = df_past.reset_index()
            df = df.reset_index()
            df_past.ds = df_past.ds.dt.strftime('%B')
            df.ds = df.ds.dt.strftime('%B')
            df_sssg = pd.merge(df_past,df, how='outer',left_on='ds',right_on='ds',suffixes=('(LY)','(Fcst)')) 
            df_sssg['total'] = (df_sssg['total(Fcst)'].sub(df_sssg['total(LY)']).div(df_sssg['total(LY)'])).mul(100)

        if resample == 'H':
            st.warning('SSSG is not available when data is resampled by Hour. Please change data resample to by Day/Week/Month for SSSG display.')
        
        return df_sssg
    

    with st.beta_expander("Inspect Aggregated Store", expanded=True):
        compare_past = st.checkbox('Compare Past Data')
        if compare_past:
            ly_shift = st.sidebar.slider('Shift past LY plot', min_value=-30, max_value=30, value=0, step=1)

            sssg = SSSG(df_2019, final_edit,data_resample_option)
            df_2018_resampled = df_2018.resample(data_resample_option).sum()
            df_2019_resampled = df_2019.resample(data_resample_option).sum()
            df_lm_resampled = df_lm.resample(data_resample_option).sum()

            fig = make_subplots(rows=3,cols=1, subplot_titles=('Aggregated Forecast','Aggregate (LY)', 'SSSG(%)'))
            fig.append_trace(go.Scatter(x=final_edit.index, y=final_edit.total, name= 'Aggregated Forecast',fill='tozeroy'), row=1, col=1)
            fig.append_trace(go.Scatter(x=df_2019_resampled.index.shift(ly_shift), y=df_2019_resampled['total'].shift(ly_shift), name='Aggregate (LY)',fill='tozeroy'), row=2, col=1)
            if data_resample_option != 'H':
                fig.append_trace(go.Scatter(x=final_edit.index,y=sssg['total'], name='SSSG (%)',fill='tozeroy'),row=3,col=1)
            #fig.append_trace(go.Scatter(x=df_2018_resampled.index, y=df_2018_resampled['total'], name= 'Aggregate (2Ys)',fill='tozeroy'), row=3, col=1)
            #fig.append_trace(go.Scatter(x=df_lm_resampled.index, y=df_lm_resampled['total'], name='Aggregate (Last Month)',fill='tozeroy'), row=4, col=1)
            fig.update_layout(showlegend=False,height=900)
            st.plotly_chart(fig, use_container_width=False)
            
        else:
            fig = px.area(final_edit, x=final_edit.index, y=final_edit.total)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.beta_columns([2, 3])
        with col1:
            st.subheader('Forecast Total')
            forecast_total=pd.DataFrame()
            forecast_total['Total'] = final.sum(axis=0)
            st.dataframe(forecast_total)
        with col2:
            st.subheader('Statistical Description')
            st.dataframe(final.resample(data_resample_option).sum().describe().T)


        st.subheader('Details')
        # Streamlit is current having a bug to convert datetime to the timezone of the server. 
        # So, it is advised to convert datetime to string to display the time correctly
        final_display = final.reset_index()
        final_display = final_display.set_index('ds')
        final_display = final_display.resample(data_resample_option).sum()
        final_display = final_display.reset_index()
        final_display.ds = pd.to_datetime(final_display.ds).dt.strftime('%Y-%m-%d %H:%M').astype(str)
        final_display = final_display.set_index('ds')
        st.dataframe(final_display.T)




    def to_excel(df, store_code_func):
        # Set up Excel file writer
        output = BytesIO()
        writer = pd.ExcelWriter(output)

        # Set store code as index ready to merge with df
        store_code_func = store_code_func.set_index('Store Code')
 
        resample_key = ['H','D','W','M']
        resample_name = ['By Hour','By Day','By Week','By Month']
        
        for k, n in zip(resample_key, resample_name):
            if k is 'H':
                df1 = df.T
            else:
                # Generate dataframe with different time samples
                df1 = df.resample(k).sum().T
            df2 = store_code_func.merge(df1, left_index=True, right_index=True)
            df2 = df2.drop(['full_name', 'Opening Date'], axis=1)
            df2.loc['Total']= df2.sum()
            df2.loc['Total',['Store','AC','Region','Concept','Province']] = ""

            # Write each dataframe to a different worksheet.
            df2.to_excel(writer, sheet_name=n)

        writer.save()
        processed_data = output.getvalue()
        return processed_data

    def get_table_download_link(df):
        val = to_excel(df, store_code_func)
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download file</a>' # decode b'abc' => abc

    st.sidebar.subheader('Download Data')
    st.sidebar.write('Click the link below to download the data for your own use:')
    st.sidebar.markdown(get_table_download_link(final), unsafe_allow_html=True)

with st.beta_expander("Inspect Individual Store", expanded=True):
    store_display = store_code_func[store_code_func['Store Code'].isin(store_code)]['full_name'].reset_index().drop('index',axis=1)
    selected_store_display = st.selectbox('Select stores', store_display, index=1)
    selected_store_code = store_code_func[store_code_func['full_name'] == selected_store_display]['Store Code']
    selected_store_data = final_edit[selected_store_code]
    individual_store_chart = px.area(selected_store_data)
    individual_store_chart.update_layout(height=300)
    individual_store_chart.update_layout(showlegend=False)
    st.plotly_chart(individual_store_chart, use_container_width=True)
