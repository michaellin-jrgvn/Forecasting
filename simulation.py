import simpy
import random
import itertools
import datetime as dt

cashier_time = []
make_time = []
oven_time = []
dispatch_time = []
delivery_time = []

def generate_order(i):
    if i <= len(df_sample):
        timeout = df_sample.loc[i,['timeout']].values[0]
        #print('time out value: ', timeout)
    return timeout

def cashier(env, cashiers, i):
    start_time = env.now
    with cashiers.request() as request:
        yield request
        yield env.timeout(random.randint(1,3))
    cashier_time.append(env.now-start_time)

def boh_process_order(env, makers, oven, i, channel):
    make_start_time = env.now
    with makers.request() as request:
        yield request
        #print('Total makers occupied: ', makers.count)
        #print('Order making in process ', i)
        yield env.timeout(random.randint(2,3))
    make_time.append(env.now-make_start_time)

    with oven.request() as request:
        oven_start_time = env.now
        yield request
        #print(oven.count)
        #print('Pizza going into the oven')
        yield env.timeout(7)
    oven_time.append(env.now-oven_start_time)

    with dispatchers.request() as request:
        dispatch_start_time = env.now
        yield request
        #print('cut, pack and dispatch')
        yield env.timeout(2)
    dispatch_time.append(env.now-dispatch_start_time)

    if channel == 'Delivery':
        with riders.request() as request:
            riders_start_time = env.now
            yield request
            yield env.timeout(random.randint(4,10))
        delivery_time.append(env.now-dispatch_start_time)
    else:
        delivery_time.append(0)


def new_order(env, makers,i, total_order, df_sample):
    while True:
        if i < total_order:
            yield env.timeout(generate_order(i))
            channel = df_sample.loc[i,['channel']].values[0]
            env.process(cashier(env,cashiers,i))
            env.process(boh_process_order(env,makers,oven,i,channel))
            i+=1
            #print('new order current time now: ',env.now)
        else:
            yield env.timeout(1)
