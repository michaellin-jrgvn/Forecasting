import pulp as pl
import pandas as pd

def optimize_labour(ds, resource):
    # Initiate class
    model = pl.LpProblem("Roster Arrangement",pl.LpMinimize)
    time_slot = list(range(len(ds.index)))

    # Define Decision variables
    x = pl.LpVariable.dicts('', time_slot, lowBound=0, cat='Integer')
    print(len(x))

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
    

    # Solve Model
    model.solve()

    print('Status',pl.LpStatus[model.status])

    df = pd.DataFrame([v.name,v.varValue] for v in model.variables())
    df[0] = df[0].str.replace('_','').astype('int64')
    df = df.sort_values(by=0,ascending=True)
    df = df.drop(0,axis=1)
    df.rename(columns={1: resource},inplace=True)
    df['time'] = ds.index
    df = df.set_index('time')

    return df
    
