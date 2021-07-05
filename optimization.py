import pulp as pl
import pandas as pd
import numpy as np
from datetime import timedelta
import itertools

def optimize_labour(ds, resource, capacity, hours_per_shift):
    # Initiate class
    model = pl.LpProblem("Roster Arrangement",pl.LpMinimize)
    time_slot = list(range(len(ds.index)))
    total_shift_interval = hours_per_shift * 2

    # Define Decision variables
    x = pl.LpVariable.dicts('', time_slot, lowBound=0, cat='Integer')

    # Define Objective function
    model += pl.lpSum(x[i] for i in time_slot)

    # Define Constraints
    # outer loop to handle number of rows
    constraint = None
    for index, slot in enumerate(time_slot):
        if index < total_shift_interval:
            # incrementing number at each column
            constraint = constraint + x[index] 
            model += constraint >= ds.iloc[index, :][resource]
            # print(constraint)
        else:
            constraint = None
            i = index - total_shift_interval
            for j in range(1+i, total_shift_interval+i+1):
                constraint = constraint + x[j]
            if index < len(time_slot)-1:
                if index >= 12 and index <= 15:
                    model += constraint == ds.iloc[index, :][resource]
                else:
                    model += constraint >= ds.iloc[index,:][resource]
            else:
                model += constraint == ds.iloc[index,:][resource]
            # print(constraint) 
                
    # Constraints to ensure the resources allocated does not exceed the station capacities
    for i in time_slot:
        model += x[i] <= capacity
        model += x[i] >= 0
    # print(model)
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
    # print(df)

    # Create roster dataframe
    roster_time = pd.Series(df.index.tolist())

    # Repeat rows based on the resource requirement the particular time in index
    roster_start_time = roster_time.repeat(df[resource])
    roster_df = pd.DataFrame()
    roster_df['start_time'] = roster_start_time
    roster_df['end_time'] = roster_df.start_time + timedelta(hours=hours_per_shift)
    roster_df['resource'] = resource
    roster_df['status'] = 'normal'
    roster_df['hours'] = (roster_df['end_time'] -
                          roster_df['start_time']).dt.total_seconds()/3600
    
    df_total_roster = df.rolling(8).sum().fillna(df.iloc[:8].cumsum())
    # test = pd.concat([df_total_roster, ds[resource]], axis=1)
    # print(test)

    for x,y in itertools.product(range(len(roster_df)), repeat=2):
        if x != y and roster_df.iloc[x]['status'] != 'duplicated' and roster_df.iloc[y]['status'] != 'duplicated':
            # print(resource,x,y)
            # Check if the current bar overlap towards end_time with the next bar
            if (roster_df.iloc[x]['end_time'] > roster_df.iloc[y]['start_time']) and (roster_df.iloc[x]['start_time'] < roster_df.iloc[y]['start_time']):
                # print('forward opportunity')
                # If yes, check if time period of the next bar has spare resources or not
                overlap_start_time = roster_df['start_time'].iloc[y].strftime('%H:%M')
                overlap_end_time = roster_df['end_time'].iloc[x].strftime('%H:%M')
                end_time = roster_df['end_time'].iloc[y].strftime('%H:%M')

                overlap_test = ds.between_time(overlap_start_time, overlap_end_time, include_end=False)[resource] < df_total_roster.between_time(
                    overlap_start_time, overlap_end_time, include_end=False)[resource]
                non_overlap_test = ds.between_time(overlap_end_time, end_time)[resource] <= df_total_roster.between_time(overlap_end_time, end_time)[resource]
                # print('overlap test:',overlap_test)
                # print('non_overlap_test:',non_overlap_test)
                if overlap_test.all() and non_overlap_test.all():
                    #print(start_time, end_time,'overlapped','opportunity')
                    roster_df['end_time'].iloc[x] = roster_df['end_time'].iloc[y]
                    roster_df['status'].iloc[y] = 'duplicated'
                    roster_df['status'].iloc[x] = 'combined'

            # Check if the current bar overlap towards start_time with the next bar
            if (roster_df.iloc[x]['start_time'] < roster_df.iloc[y]['end_time']) and (roster_df.iloc[x]['start_time'] > roster_df.iloc[y]['start_time']):
                # print('backward opportunity')
                # If yes, check if time period of the next bar has spare resources or not
                overlap_start_time = roster_df['start_time'].iloc[x].strftime('%H:%M')
                overlap_end_time = roster_df['end_time'].iloc[y].strftime(
                    '%H:%M')
                start_time = roster_df['start_time'].iloc[y].strftime('%H:%M')

                overlap_test = ds.between_time(overlap_start_time, overlap_end_time, include_end=False)[
                    resource] < df_total_roster.between_time(overlap_start_time, overlap_end_time,include_end=False)[resource]
                non_overlap_test = ds.between_time(overlap_end_time, start_time)[
                    resource] <= df_total_roster.between_time(overlap_end_time, start_time)[resource]
                # print(overlap_test.all())
                # print(non_overlap_test.all())
                if overlap_test.all() and non_overlap_test.all():
                    #print(start_time, end_time,'overlapped','opportunity')
                    roster_df['start_time'].iloc[x] = roster_df['start_time'].iloc[y]
                    roster_df['status'].iloc[y] = 'duplicated'
                    roster_df['status'].iloc[x] = 'combined'


    # Combine consecutive 4 hours shift together
    for x, y in itertools.product(range(len(roster_df)), repeat=2):
        if x != y and roster_df['status'].iloc[x] == 'normal' and roster_df['status'].iloc[y] == 'normal':
            if roster_df.iloc[x]['end_time'] - roster_df.iloc[x]['start_time'] == timedelta(hours=4):
                if roster_df.iloc[x]['end_time'] == roster_df.iloc[y]['start_time']:
                    roster_df['end_time'].iloc[x] = roster_df['end_time'].iloc[y]
                    roster_df['status'].iloc[y] = 'duplicated'
                    roster_df['status'].iloc[x] = 'combined'
                elif roster_df.iloc[x]['start_time'] == roster_df.iloc[y]['end_time']:
                    roster_df['start_time'].iloc[x] = roster_df['start_time'].iloc[y]
                    roster_df['status'].iloc[y] = 'duplicated'
                    roster_df['status'].iloc[x] = 'combined'
    roster_df = roster_df[roster_df.status != 'duplicated']

    roster_df['hours'] = (roster_df['end_time'] -
                          roster_df['start_time']).dt.total_seconds()/3600


    return roster_df
    
