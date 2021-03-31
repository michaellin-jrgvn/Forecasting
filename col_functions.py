import streamlit as st
import pandas as pd
import datetime
import glob
import plotly.express as px
import numpy as np
from io import BytesIO
import base64
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose



@st.cache()
def col_process(df):
    #Preprocess data
    # drop rows with code is null - e.g. A Team, Commissionary, etc
    df.dropna(0, how='any', inplace=True)
    # remove 'grand total' rows
    df1 = df[(df['Date'] != 'Grand Total') & (df['Store Name'] != 'Total') & (df['Date'] != 'Total All Store')]
    return df1

@st.cache()
def col_total_percent(df): 
    actual_sales = df['Actual sales'].sum()
    total_col = df['Total COL $ (included Holiday and paid leave days)'].sum()
    total_col_percent = 100 * total_col / actual_sales
    return total_col_percent, actual_sales

@st.cache()
def read_col_files():
    all_files = glob.glob("./col_data/*.xls")
    col = pd.DataFrame()
    trans = pd.DataFrame()
    for file in all_files:
        print('read: ', file)
        df = pd.read_excel(file, sheet_name='HR Data ', parse_dates=['Date'])
        df_trans = pd.read_excel(file, sheet_name='Sales Data', parse_dates=['date'])
        df = col_process(df)
        col = pd.concat([col, df])
        trans = pd.concat([trans,df_trans])
        # drop duplicates
        col = col.drop_duplicates(subset=['Actual sales','Total COL $ (included Holiday and paid leave days)'])
        trans = trans.drop_duplicates(subset=['date','shopcode','Sale','Trans'])
        # col['Date'] = pd.to_datetime(col['Date'])
        col = col.fillna(0)
    return col, trans

@st.cache
def store_code():
    df = pd.read_excel('./store_info/Tracking store by year.xls', nrows=100,usecols=['Store Code','Title', 'AC', 'Region','Province','Concept','Opening Date'],parse_dates=['Opening Date'])
    df.rename(columns={'Title':'Store'}, inplace=True)
    df['full_name'] = df['Store Code'] + '-' + df['Store']
    return df

@st.cache(allow_output_mutation=True)
def data_filter(df, trans, store_code):
    data_filter = df[df['Code'].isin(store_code)]
    trans_filter = trans[trans['shopcode'].isin(store_code)]
    return data_filter, trans_filter

@st.cache(allow_output_mutation=True)
def filtered_data_merged(df, trans, store):
    filtered_data_merged = df.merge(store,how='left',left_on='Code',right_on='Store Code')
    filtered_data_merged = filtered_data_merged.drop(['Store Code','Store','Opening Date','full_name'], axis=1)
    filtered_data_merged['Date'] = pd.to_datetime(filtered_data_merged['Date'])

    filtered_trans_merged = trans.merge(store,how='left',left_on='shopcode',right_on='Store Code')
    filtered_trans_merged = filtered_trans_merged.drop(['Store Code','Store','Opening Date','full_name'], axis=1)
    filtered_trans_merged['date'] = pd.to_datetime(filtered_trans_merged['date'])

    filtered_data_merged = filtered_data_merged.set_index(['Date','Code'])
    trans['ordertype desc'] = trans['ordertype desc'].str.strip()

    # extract sales/trans by channel and merge to filtered_data_merged dataframe
    for type in trans['ordertype desc'].unique():
        disc = trans[trans['ordertype desc'] == type].set_index(['date','shopcode'])[['Sale','Trans']]
        disc.index.rename(['Date','Code'], inplace=True)

        sales_column = type + ' sales'
        trans_column = type + ' trans'
        disc.columns = [[sales_column, trans_column]]

        filtered_data_merged = filtered_data_merged.merge(disc, left_index=True, right_index=True, suffixes=('_left','_right'))
    filtered_data_merged.rename(columns=''.join,inplace=True)

    filtered_data_merged.reset_index(inplace=True)
    # rename columns COL%, COL$ and COL.1% to forecast COL%, forecast COL$ and actual COL%
    filtered_data_merged = filtered_data_merged.rename(columns={'COL $':'Forecast COL $','COL %':'Forecast COL %','COL %.1':'Actual COL %'})

    # Assign column types
    filtered_data_merged = filtered_data_merged.astype({
            'Code': 'string',
            'Store Name': 'string',
            'Forecast Sales': 'int64',
            'Forecast Hours': 'int64',
            'Forecast COL $': 'float64',
            'Forecast COL %': 'float64',
            'Actual sales': 'float64',
            'Total actual hours (included Holiday and paid leave days)':'int64',
            'Actual hours of MNGT': 'int64',
            'Actual hours of TMs Full time': 'int64',
            'Actual hours of TMs Part time': 'int64',
            'Total actual hours (excluded Holiday and paid leave days)': 'int64',
            'Hours of holiday/paid leave days': 'int64',
            'Total COL $ (included Holiday and paid leave days)':'float64',
            'COL $  of TM Full time':'float64',
            'COL $  of TM Part time':'float64',
            'COL $ Management':'float64',
            'Total COL (excluded Holiday and paid leave days)':'float64',
            'COL of holidays/paid leave days':'float64',
            'Actual COL %':'float64',
            'COL Val %':'float64',
            'Working hour in work shift':'int64',
            'Over time in normal day':'int64',
            'Over time in weekend':'int64',
            'Over time in holiday': 'int64',
            'working hour in normal day (Night)': 'int64',
            'Over time in normal day (Night)':'int64',
            'Over time in weekend (Night)':'int64',
            'Over time in holiday (Night)':'int64',
            '13th salary':'float64',
            'BSC bonus':'float64',
            'PA bonus':'float64',
            'Meal Allowance':'float64',
            'Insurance contribution Amount per day':'float64'
        })

    # Add extra columns Total Transaction, SPMH and TPMH
    filtered_data_merged['Total Transaction'] = filtered_data_merged['Pickup trans']+filtered_data_merged['Dinein trans']+filtered_data_merged['Delivery trans']
    if filtered_data_merged['Forecast Hours'].empty:
        filtered_data_merged['Forecast SPMH'] = 0
    else: 
        filtered_data_merged['Forecast SPMH'] = filtered_data_merged['Forecast Sales'] / filtered_data_merged['Forecast Hours']
    if filtered_data_merged['Total actual hours (included Holiday and paid leave days)'].empty:
        filtered_data_merged['Actual SPMH'] = 0
    else:
        filtered_data_merged['Actual SPMH'] = filtered_data_merged['Actual sales'] / filtered_data_merged['Total actual hours (included Holiday and paid leave days)']
        filtered_data_merged['Actual TPMH'] = filtered_data_merged['Total Transaction'] / filtered_data_merged['Total actual hours (included Holiday and paid leave days)']

    # Add MAPE
    def MAPE(actual, forecast):
        mape = 100* (abs(actual - forecast) / actual)
        mape.replace(np.inf,0, inplace=True)
        mape = mape.fillna(0)
        return mape

    filtered_data_merged['Sales MAPE'] = MAPE(filtered_data_merged['Actual sales'], filtered_data_merged['Forecast Sales'])

    filtered_data_merged['Labour MAPE'] = MAPE(filtered_data_merged['Total actual hours (included Holiday and paid leave days)'],filtered_data_merged['Forecast Hours'])


    # Add COL% Variance
    filtered_data_merged['COL% Variance'] = (filtered_data_merged['Actual COL %'] - filtered_data_merged['Forecast COL %']) / filtered_data_merged['Actual COL %']
    filtered_data_merged['COL% MAPE'] = MAPE(filtered_data_merged['Actual COL %'], filtered_data_merged['Forecast COL %'])
    filtered_data_merged['TA'] = filtered_data_merged['Actual sales'].div(filtered_data_merged['Total Transaction'])
    filtered_data_merged = filtered_data_merged.fillna(0)
    return filtered_data_merged

def to_excel(df):

    # Set up Excel file writer
    output = BytesIO()
    writer = pd.ExcelWriter(output)

    # Write each dataframe to a different worksheet.
    df.to_excel(writer, sheet_name='Summary')

    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download file</a>' # decode b'abc' => abc

@st.cache()
def load_baseline():
    # import baseline COL data and merge with current COL Dataframe
    bl_df = pd.read_excel('./Data/Baseline/COL Baseline - Jun 20.xls', sheet_name='Base line JUN 2019', header=0)
    return bl_df

def regression_table(fig, breakdown):
    regression_df = px.get_trendline_results(fig)
    regression_df = regression_df.set_index(breakdown)
    m=[]
    y=[]
    r2=[]
    for store in regression_df.index:
        _m = regression_df.loc[store].px_fit_results.params[1]
        m.append(_m)
        _y = regression_df.loc[store].px_fit_results.params[0]
        y.append(_y)
        _r2 = regression_df.loc[store].px_fit_results.rsquared
        r2.append(_r2)
    regression_df['Gradient'] = m
    regression_df['y-intercept'] = y
    regression_df['R2 Score'] = r2
    # st.dataframe(regression_df[['Gradient','y-intercept','R2 Score']].sort_values(breakdown))
    regression_df.dropna(inplace=True)
    return regression_df

def regression_plot(filtered_data_merged, sidebar_name, x_axis, y_axis):
    # Plot chart with all the settings

    if len(x_axis) > 0 and len(y_axis) > 0:
        with st.sidebar.beta_expander(sidebar_name, expanded=True):
            # Data grouping and splitplot
            breakdown_option = [None, 'AC','Region','Province','Concept','Store Name','Code']
            breakdown = st.selectbox('Data grouping', breakdown_option)
            split_plot = st.selectbox('Split plot by', breakdown_option)

            # Check box for add trendline
            if st.checkbox('Add trend line', value=True):
                trendline = "ols"
            else:
                trendline = None

            # Remove 0 values in the dataset to plot more accurate regression line
            if st.checkbox('Remove 0 values'):
                filtered_data_merged = filtered_data_merged[(filtered_data_merged[x_axis] > 0) & (filtered_data_merged[y_axis] > 0)]
            else:
                # Define 0 value
                zero_value_x_axis = filtered_data_merged[filtered_data_merged[x_axis] == 0][x_axis]
                zero_value_y_axis = filtered_data_merged[filtered_data_merged[y_axis] == 0][y_axis]

                st.subheader('No. of 0 values at x-axis: ')
                st.info(zero_value_x_axis.count())
                st.subheader('No. of 0 values at y-axis:')
                st.info(zero_value_y_axis.count())

            # Add boxplot for x-axis and y-axis
            if st.checkbox('Add boxplots for axes'):
                marginal = "box"
            else:
                marginal = None
                
        fig = px.scatter(filtered_data_merged,x=x_axis,y=y_axis,color=breakdown,opacity=0.5,facet_col=split_plot, facet_col_wrap=2,trendline=trendline,marginal_x=marginal, marginal_y=marginal, height=600).update_layout(autosize=True)
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(rangemode='tozero')
        st.plotly_chart(fig, use_container_width=True)

        if breakdown == None:
            if trendline:
                # Add trendline with variables
                regression_df = px.get_trendline_results(fig)
                st.subheader('Properties of regression line')
                st.write('Gradient:')
                st.info(round(regression_df.iloc[0].px_fit_results.params[1],2))
                st.write('y-intercept:')
                st.info(round(regression_df.iloc[0].px_fit_results.params[0],2))
                st.write('R2 Score')
                st.info(round(regression_df.iloc[0].px_fit_results.rsquared,2))
        else:
            if trendline:
                st.subheader('Properties of regression line')
                # Add trendline by stores with table
                regression_df = regression_table(fig, breakdown)
                st.dataframe(regression_df)
                st.subheader('Gradient-R2 Score Scatterplot')
                y_r2_plot = px.scatter(regression_df,x='Gradient',y='y-intercept', size='R2 Score',color=regression_df.index)
                y_r2_plot.add_hline(y=regression_df['y-intercept'].mean())
                y_r2_plot.add_vline(x=regression_df['Gradient'].mean())
                st.plotly_chart(y_r2_plot, use_container_width=True)

                # Download Data Section
                st.sidebar.subheader('Download Regression Data')
                st.sidebar.write('Click the link below to download the data for your own use:')
                st.sidebar.markdown(get_table_download_link(regression_df), unsafe_allow_html=True)

def spmh_time_series(data, resample):
    ts_df = filtered_data_merged.set_index('Date')
    spmh_df = ts_df.resample('D').mean()
    spmh_df = spmh_df[['Actual SPMH']]
    spmh_df['SPMH Moving Average'] = spmh_df['Actual SPMH'].ewm(span=7,adjust=False).mean()
    spmh_ts_plot = px.line(spmh_df)

    return spmh_df, spmh_ts_plot
