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
from optimization import optimize_labour

from sklearn.preprocessing import MinMaxScaler

import random

import simpy

np.random.seed(42)
random.seed(42)

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
# st.plotly_chart(plt_sales,use_container_width=True)
forecast_daily_sales = int(sales_col_opt['Total Sales'].sum())
st.write('Total Sales of the day is: ', forecast_daily_sales)

def simulation_df(df):
    channels = ['Dinein','Pickup','Delivery']
    simulation_df = pd.DataFrame()
    for channel in channels:
        ## Assign random minutes for each row
        # Preprocessing
        df['{} - TC'.format(channel)] = df['{} - TC'.format(channel)].round(0).astype('int')

        # Setup to generate random TC using normal distribution
        mu_tc  = np.log(df['{} - TC'.format(channel)][['{} - TC_y_final'.format(channel)]].to_numpy()[:,0])
        sigma_tc = np.log(df['{} - TC'.format(channel)]['{} - TC_y_final'.format(channel)]).std()
        df['{} - TC'.format(channel)]['simulate_TC'] = np.random.lognormal(mu_tc, sigma_tc).round(0).astype('int')

        # Duplicate rows based on projected TC / hour
        repeat_array = df['{} - TC'.format(channel)].reset_index()[['simulate_TC']].to_numpy()[:,0]
        channel_preprocess_df = df['{} - TC'.format(channel)].loc[df['{} - TC'.format(channel)].index.repeat(repeat_array)]

        # Assign random minutes for each ticket and sort according to time
        channel_preprocess_df['minutes'] = np.random.randint(0, 59, channel_preprocess_df.shape[0])
        channel_preprocess_df.index= channel_preprocess_df.index + pd.to_timedelta(channel_preprocess_df[['minutes']].to_numpy()[:,0], unit='m')
        channel_preprocess_df = channel_preprocess_df.sort_index()

        # Duplicate same number of row in ta dataframe as tc dataframe
        dinein_ta_df = df['{} - TA'.format(channel)].loc[df['{} - TA'.format(channel)].index.repeat(repeat_array)]
        dinein_ta_df.index = channel_preprocess_df.index

        # Set up to generate random TA using normal Distribution
        mu_ta = dinein_ta_df['{} - TA_y_final'.format(channel)]
        sigma_ta = dinein_ta_df['{} - TA_y_final'.format(channel)].std()
        channel_preprocess_df['bill_size'] = np.random.normal(mu_ta,sigma_ta).clip(dinein_ta_df['{} - TA_y_lower'.format(channel)],dinein_ta_df['{} - TA_y_upper'.format(channel)]).round(0).astype(int)
        
        # Remove all unnecessary columns except index and bill_size and add channel Details
        channel_preprocess_df = channel_preprocess_df[['bill_size']]
        channel_preprocess_df['channel'] = channel
        simulation_df = pd.concat([simulation_df, channel_preprocess_df])
        simulation_df = simulation_df.sort_index()
    return simulation_df

@st.cache()
def loop_simulation(simulation_df):
    df_sim_full = {}
    df_sim_sum = pd.DataFrame()
    loop = 10
    for i in range(loop):
        df_sim_full[i] = simulation_df(df)
        agg_h=df_sim_full[i].resample('H').sum()
        plt_sales.add_trace(go.Scatter(x=agg_h.index,y=agg_h.bill_size,mode='markers'))
        my_bar.progress((i+1)/loop)
        df_sim_sum = df_sim_sum.append(agg_h.sum(),ignore_index=True)
    return df_sim_full, agg_h, plt_sales, df_sim_sum

my_bar = st.progress(0)
df_sim_full, agg_h, plt_sales, df_sim_sum = loop_simulation(simulation_df)
st.write(df_sim_sum.describe())
sim_sum_hist = px.histogram(df_sim_sum, x='bill_size')
st.plotly_chart(plt_sales, use_container_width=True)
st.plotly_chart(sim_sum_hist, use_container_width=True)

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
forecast_spmh = regression_table.Gradient * df_sim_sum.bill_size.mean() + regression_table['y-intercept']
st.write('Minimum SPMH from regression is: ', int(forecast_spmh))
manhour_allowed = df_sim_sum.bill_size.mean() / forecast_spmh
st.write('Maximum manhour allowance from regression is:', int(manhour_allowed))

makers_capacity = 3+1
cashiers_capacity = 2+1
riders_capacity = 6+1
oven_capacity = 4
dispatchers_capacity = 2+1
csr_capacity = 2+1

summary_kpi_container = st.beta_container()
col1,col2 = summary_kpi_container.beta_columns(2)

# Iterate through the capacities of resources
@st.cache()
def run_ops_simulation(makers_capacity,cashiers_capacity,dispatchers_capacity,riders_capacity,oven_capacity,csr_capacity):
    scenario = 0
    random.seed(42)
    time_df = pd.DataFrame(index=df_sim_full[0].index, columns=['cashier_time','make_time','oven_time','dispatch_time','foh_dinein_dispatch_time','foh_pickup_dispatch_time','foh_table_cleaning_time','order_await_delivery','delivery_time','delivery_return_time'])
    capacity_df = pd.DataFrame(index=df_sim_full[0].index, columns=['cashiers','csr','csr_serving','makers','dispatchers','riders'])
    scenario_kpi_df = pd.DataFrame(columns=['scenario','cashiers','csr','makers','dispatchers','riders','TPMH','SPMH','u14 hitrate','u14 max','u30 hitrate','u30 max'])
    scenario_df = {}
    def generate_order(i):
        if i <= len(df_sample):
            timeout = df_sample.loc[i,['timeout']].values[0]
            #print('time out value: ', timeout)
        return timeout

    def cashier(env, cashiers, i, channel):
        if channel == 'Delivery':
            time_df.iloc[i]['cashier_time'] = 0
            capacity_df.iloc[i]['cashiers'] = 0
        else:
            start_time = env.now
            with cashiers.request() as request:
                yield request
                capacity_df.iloc[i]['cashiers'] = cashiers.count
                yield env.timeout(random.randint(1,3))
            time_df.iloc[i]['cashier_time'] = env.now-start_time

    def boh_process_order(env, makers, oven, i, channel):
        make_start_time = env.now
        with makers.request() as request:
            yield request
            capacity_df.iloc[i]['makers'] = makers.count
            #print('Total makers occupied: ', makers.count)
            #print('Order making in process ', i)
            yield env.timeout(random.randint(2,3))
        time_df.iloc[i]['make_time'] = env.now-make_start_time
        
        with oven.request() as request:
            oven_start_time = env.now
            yield request
            yield env.timeout(7)
        time_df.iloc[i]['oven_time'] = env.now-oven_start_time

        with dispatchers.request() as request:
            dispatch_start_time = env.now
            yield request
            capacity_df.iloc[i]['dispatchers'] = dispatchers.count
            #print('cut, pack and dispatch')
            yield env.timeout(2)
            dispatch_end_time = env.now
        time_df.iloc[i]['dispatch_time'] = env.now-dispatch_start_time

        if channel == 'Delivery':
            with riders.request() as request:
                drive_time = random.randint(4,10)
                customer_waiting_time = random.randint(3,5)
                yield request
                capacity_df.iloc[i]['riders'] = riders.count
                out_delivery_time = env.now
                yield env.timeout(drive_time)
                yield env.timeout(customer_waiting_time)
                delivery_complete_time = env.now
                # Driver return to store
                yield env.timeout(drive_time)
            time_df.iloc[i]['order_await_delivery'] = out_delivery_time-dispatch_end_time
            time_df.iloc[i]['delivery_time'] = delivery_complete_time-out_delivery_time
            time_df.iloc[i]['delivery_return_time'] = env.now-delivery_complete_time

        else:
            time_df.iloc[i]['order_await_delivery'] = 0
            time_df.iloc[i]['delivery_time'] = 0
            time_df.iloc[i]['delivery_return_time'] = 0
            capacity_df.iloc[i]['riders'] = 0
    
    def foh_order(env, csr, cashiers, i, channel):
        if channel == 'Dinein':
            foh_service_time = env.now
            with csr.request() as request:
                busing_time = 1
                yield request
                capacity_df.iloc[i]['csr_serving'] = csr.count
                yield env.timeout(busing_time)
            time_df.iloc[i]['foh_dinein_dispatch_time'] = env.now - foh_service_time
            time_df.iloc[i]['foh_pickup_dispatch_time'] = 0
            customer_eating_time = np.random.randint(30,45)
            yield env.timeout(customer_eating_time)
            with csr.request() as request:
                clean_up_request = env.now
                cleaning_time = np.random.randint(3,5)
                table_set_up_time = np.random.randint(2,3)
                yield request
                capacity_df.iloc[i]['csr_cleaning'] = csr.count
                yield env.timeout(cleaning_time)
                yield env.timeout(table_set_up_time)
            time_df.iloc[i]['foh_table_cleaning_time'] = env.now - clean_up_request
        elif channel == 'Pickup':
            foh_pickup_dispatch = env.now
            with cashiers.request() as request:
                foh_dispatch = random.randint(1,2)
                yield request
                capacity_df.iloc[i]['cashiers'] = cashiers.count
                yield env.timeout(foh_dispatch)
            time_df.iloc[i]['foh_pickup_dispatch_time'] = env.now - foh_pickup_dispatch
            time_df.iloc[i]['foh_dinein_dispatch_time'] = 0
            time_df.iloc[i]['foh_table_cleaning_time'] = 0
        else:
            time_df.iloc[i]['foh_pickup_dispatch_time'] = 0
            time_df.iloc[i]['foh_dinein_dispatch_time'] = 0
            time_df.iloc[i]['foh_table_cleaning_time'] = 0

    def new_order(env, makers,i, total_order, df_sample,cashiers,csr):
        while True:
            if i < total_order:
                yield env.timeout(generate_order(i))
                channel = df_sample.loc[i,['channel']].values[0]
                env.process(cashier(env,cashiers,i,channel))
                env.process(boh_process_order(env,makers,oven,i,channel))
                env.process(foh_order(env,csr,cashiers,i,channel))
                i+=1
            else:
                yield env.timeout(1) 

    ###### Use sales forecast simulation 0 for process simulation (This one will need improvement) #####
    df_sample = df_sim_full[0].copy()

    df_sample = df_sample.reset_index()
    df_sample['timeout'] = df_sample['index'].diff().dt.seconds.div(60)
    df_sample['timeout'].fillna(0.0,inplace=True)
    total_order = len(df_sample)

    print('Starting Simulation')
    print('Total order: ', total_order)

    # simulate the operation process by iterating all resources capacities 
    for j in range(1, makers_capacity):
        for k in range(1, cashiers_capacity):
            for l in range(1, dispatchers_capacity):
                for m in range(1, riders_capacity):
                    for n in range(1, csr_capacity):
                        # Create simulation enviornment and define resources capacities
                        env = simpy.Environment()
                        i=0
                        makers = simpy.Resource(env, capacity = j)
                        cashiers = simpy.Resource(env, capacity = k)
                        oven = simpy.Resource(env, capacity=oven_capacity)
                        dispatchers = simpy.Resource(env, capacity=l)
                        riders = simpy.Resource(env, capacity = m)
                        csr = simpy.Resource(env, capacity = n)
                        #res = MonitoredResource(env, j)
                        #st.write(res.data)
                        env.process(new_order(env, makers,i, total_order, df_sample,cashiers,csr))
                        print('processing...',j,k,l,m,n)
                        store_opening_time = 8
                        store_closing_time = 24
                        total_opening_hours = (store_closing_time - store_opening_time) * 60
                        env.run(until=total_opening_hours)
                        print('Simulation completed')

                        total_manhour = (j+k+l+m+n)*14+16+8
                        TPMH = total_order / total_manhour
                        SPMH = df_sample.bill_size.sum() / total_manhour

                        # Determine u14 hitrate
                        u14_df = time_df.copy()
                        u14_df['total_time'] = u14_df[['make_time','oven_time','dispatch_time']].sum(axis=1)
                        fail_u14 = u14_df[u14_df['total_time']> 14]['total_time'].count()
                        u14_hitrate = round(1-(fail_u14/total_order),2)
                        u14_max = u14_df['total_time'].max()

                        # Determine u30 hitrate
                        deli_df = time_df[time_df['delivery_time']>0]
                        total_deli_order = len(deli_df)
                        deli_df['total_time'] = deli_df.sum(axis=1)
                        pass_u30 = deli_df[deli_df['total_time'].sub(deli_df['delivery_return_time']) <= 30]['total_time'].count()
                        u30_hitrate = round((pass_u30/total_deli_order),2)
                        u30_max = deli_df['total_time'].sub(deli_df['delivery_return_time']).max()

                        # Insert data to dataframe
                        scenario_data = pd.DataFrame([[scenario,k,j,l,m,n,TPMH,SPMH,u14_hitrate,u14_max,u30_hitrate,u30_max]],columns=['scenario','cashiers','makers','dispatchers','riders','csr','TPMH','SPMH','u14 hitrate','u14 max','u30 hitrate','u30 max'])
                        scenario_kpi_df = scenario_kpi_df.append(scenario_data)
                        scenario_kpi_df['u30 absolute var'] = np.abs(scenario_kpi_df['u30 hitrate'] - 0.9)
                        scenario_kpi_df['u14 absolute var'] = np.abs(scenario_kpi_df['u14 hitrate'] - 0.9)
                        scenario_kpi_df = scenario_kpi_df.sort_values(['u30 max','u30 absolute var','u14 max','u14 absolute var','SPMH'], ascending=(True,True,True,True,False))
                        scenario_df[scenario] = time_df
                        
                        #with st.beta_expander('SPMH: ' + round(scenario_data['SPMH'],0).to_string(index=False) + ' U14 Hit Rate: '+ round(scenario_data['u14 hitrate'],1).to_string(index=False)+ ' U30 Hit Rate: '+ round(scenario_data['u30 hitrate'],1).to_string(index=False)):
                        #    time_df_plot = px.area(time_df)
                        #    st.plotly_chart(time_df_plot)

                        scenario +=1
    return scenario_df, scenario_kpi_df

scenario_df, scenario_kpi_df = run_ops_simulation(makers_capacity,cashiers_capacity,dispatchers_capacity,riders_capacity,oven_capacity,csr_capacity)
scenario_kpi_df = scenario_kpi_df.reset_index(drop=True)

#option = st.selectbox('Select scneario:', range(len(scenario_df)))

#st.write(scenario_df[option])

u30_plot = px.scatter(scenario_kpi_df,x='SPMH',y='u30 hitrate',hover_data=['scenario','u30 max','u14 hitrate','u14 max'])
col1.plotly_chart(u30_plot,use_container_width=True)
u14_plot = px.scatter(scenario_kpi_df,x='SPMH',y='u14 hitrate',hover_data=['scenario','u14 max','u30 hitrate','u30 max'])
col2.plotly_chart(u14_plot,use_container_width=True)

# Use minmax scaler to normalize all variable to determine the optimum capacity arrangement
scaler = MinMaxScaler()
scenario_kpi_df[['u30 hitrate trans','u30 max trans','u14 hitrate trans','u14 max trans','SPMH trans','u30 abs var trans','u14 abs var trans']] = scaler.fit_transform(scenario_kpi_df[['u30 hitrate','u30 max','u14 hitrate','u14 max','SPMH', 'u30 absolute var','u14 absolute var']])
scenario_kpi_df[['optimum score']] = -scenario_kpi_df['u30 abs var trans']-scenario_kpi_df['u30 max trans'] - scenario_kpi_df['u14 abs var trans'] -scenario_kpi_df['u14 max trans'] + scenario_kpi_df['SPMH trans']*1.5
scenario_kpi_df = scenario_kpi_df.sort_values('optimum score', ascending=False)
st.subheader('Recommended Capcity Arrangements:')
st.write(scenario_kpi_df[['scenario','cashiers','csr','makers','dispatchers','riders','TPMH','SPMH','u14 hitrate','u14 max','u30 hitrate','u30 max']].head(3))

# set optimal arrnagement the first line of the dataframe
optimal = scenario_kpi_df.iloc[0,:]
optimal_details = scenario_df[optimal.scenario]

# plot simulation time chart at optimal level
time_df_plot = px.area(optimal_details)
st.plotly_chart(time_df_plot)

# Resample the scenario into 30mins timeframe and determine the manpower requirement in every 30mins
occupancy_30m = scenario_df[optimal.scenario].resample('30min').sum() / 30
manpower_requirement = occupancy_30m[['cashier_time','foh_dinein_dispatch_time','foh_pickup_dispatch_time','foh_table_cleaning_time','make_time','dispatch_time','delivery_time','delivery_return_time']]
manpower_requirement['csr_time'] = manpower_requirement['foh_dinein_dispatch_time'] + manpower_requirement['foh_pickup_dispatch_time'] + manpower_requirement['foh_table_cleaning_time']
manpower_requirement['rider_time'] = manpower_requirement['delivery_time'] + manpower_requirement['delivery_return_time']
manpower_requirement = manpower_requirement.drop(['delivery_time','delivery_return_time','foh_dinein_dispatch_time','foh_pickup_dispatch_time','foh_table_cleaning_time'],axis=1)

# Set ceiling of each column equal to the optimal manpower requirement
manpower_requirement['make_time'] = manpower_requirement['make_time'].clip(0,scenario_kpi_df.iloc[0,:]['makers'])
manpower_requirement['cashier_time'] = manpower_requirement['cashier_time'].clip(0,scenario_kpi_df.iloc[0,:]['cashiers'])
manpower_requirement['csr_time'] = manpower_requirement['csr_time'].clip(0,scenario_kpi_df.iloc[0,:]['csr'])
manpower_requirement['dispatch_time'] = manpower_requirement['dispatch_time'].clip(0,scenario_kpi_df.iloc[0,:]['dispatchers'])
manpower_requirement['rider_time'] = manpower_requirement['rider_time'].clip(0,scenario_kpi_df.iloc[0,:]['riders'])

st.write(manpower_requirement)
occupancy_30m_plt = px.area(manpower_requirement)
st.plotly_chart(occupancy_30m_plt)

# Preprocess dataframe ready for roster optimization
# Shifting time index by -60mins for roster to cater for TM preparation (30mins), ready for demand (30mins)
roster_df = manpower_requirement.shift(periods=-30,freq='min')
# factorize all requirement by 80% to prevent overstaffing
# roster_df = roster_df * 0.8
st.write(roster_df)

# Run Linear Programming on each station
make_roster, make_schedule = optimize_labour(roster_df, 'make_time')
rider_roster, rider_schedule = optimize_labour(roster_df,'rider_time')
cashier_roster, cashier_schedule = optimize_labour(roster_df,'cashier_time')
dispatcher_roster, dispatcher_schedule = optimize_labour(roster_df,'dispatch_time')
csr_roster, csr_schedule = optimize_labour(roster_df,'csr_time')
merged_roster = pd.concat([cashier_roster, csr_roster, make_roster, dispatcher_roster, rider_roster],axis=1)
st.write(merged_roster)

# Generate roster schedule chart
merged_schedule = pd.concat([make_schedule,rider_schedule,cashier_schedule,dispatcher_schedule,csr_schedule])
merged_schedule.reset_index(drop=True, inplace=True)
st.write(merged_schedule)
schedule_plt = px.timeline(merged_schedule,x_start='start_time',x_end='end_time',y=merged_schedule.index, color='resource')
st.plotly_chart(schedule_plt)
hours_per_shift = 4
final_MH = merged_roster.sum() * hours_per_shift

# Conclude the final SPMH and MH from optimal schedule
st.write('Final SPMH based on roster: ',round(df_sim_sum.bill_size.mean()/(final_MH.sum() +16+8),0))
st.write('Total hours arranged: ', final_MH.sum()+16+8)