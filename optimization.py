import pulp as pl
import pandas as pd
from datetime import datetime, timedelta

def optimize_labour(ds, resource):
    # Initiate class
    model = pl.LpProblem("Roster Arrangement",pl.LpMinimize)
    time_slot = list(range(len(ds.index)))

    # Define Decision variables
    x = pl.LpVariable.dicts('', time_slot, lowBound=0, cat='Integer')

    # Define Objective function
    model += pl.lpSum(x[i] for i in time_slot)

    # Define Constraints
    model += x[0] >= ds.iloc[0,:][resource]
    model += x[0] + x[1] >= ds.iloc[1,:][resource]
    model += x[0] + x[1] + x[2] >= ds.iloc[2,:][resource]
    model += x[0] + x[1] + x[2] + x[3] >= ds.iloc[3,:][resource]
    model += x[0] + x[1] + x[2] + x[3] + x[4] >= ds.iloc[4,:][resource]
    model += x[0] + x[1] + x[2] + x[3] + x[4] + x[5] >= ds.iloc[5,:][resource]
    model += x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] >= ds.iloc[6,:][resource]
    model += x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] >= ds.iloc[7,:][resource]
    model += x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] >= ds.iloc[8,:][resource]
    model += x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] >= ds.iloc[9,:][resource]
    model += x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] >= ds.iloc[10,:][resource]
    model += x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] >= ds.iloc[11,:][resource]
    model += x[5] + x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] >= ds.iloc[12,:][resource]
    model += x[6] + x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] >= ds.iloc[13,:][resource]
    model += x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] >= ds.iloc[14,:][resource]
    model += x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] >= ds.iloc[15,:][resource]
    model += x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] >= ds.iloc[16,:][resource]
    model += x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] >= ds.iloc[17,:][resource]
    model += x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] >= ds.iloc[18,:][resource]
    model += x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] >= ds.iloc[19,:][resource]
    model += x[13] + x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] >= ds.iloc[20,:][resource]
    model += x[14] + x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21] >= ds.iloc[21,:][resource]
    model += x[15] + x[16] + x[17] + x[18] + x[19] + x[20] + x[21] + x[22] >= ds.iloc[22,:][resource]
    model += x[16] + x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] >= ds.iloc[23,:][resource]
    model += x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] >= ds.iloc[24,:][resource]
    model += x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] >= ds.iloc[25,:][resource]
    model += x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] >= ds.iloc[26,:][resource]
    model += x[20] + x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] >= ds.iloc[27,:][resource]
    model += x[21] + x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[28] >= ds.iloc[28,:][resource]
    model += x[22] + x[23] + x[24] + x[25] + x[26] + x[27] + x[28] + x[29] == 0
    #model += x[23] + x[24] + x[25] + x[26] + x[27] + x[28] + x[29] + x[30] == 0

    # Solve Model
    model.solve()

    print('Status',pl.LpStatus[model.status])

    # Create resources dataframe
    df = pd.DataFrame([v.name,v.varValue] for v in model.variables())
    df[0] = df[0].str.replace('_','').astype('int64')
    df = df.sort_values(by=0,ascending=True)
    df = df.drop(0,axis=1)
    df.rename(columns={1: resource},inplace=True)
    df['time'] = ds.index
    df = df.set_index('time')

    # Create roster dataframe
    shift_hours = 4
    roster_time = pd.Series(df.index.tolist())

    # Repeat rows based on the resource requirement the particular time in index
    roster_start_time = roster_time.repeat(df[resource])
    roster_df = pd.DataFrame()
    roster_df['start_time'] = roster_start_time
    roster_df['end_time'] = roster_df.start_time + timedelta(hours=shift_hours)
    roster_df['resource'] = resource

    return df, roster_df
    
