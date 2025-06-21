import numpy as np
import pandas as pd
from multiprocessing import Pool
import time
import os as os
from obspy import read, Stream, UTCDateTime
from kavutil import get_data2, run4catalog


def test_multi(a):
    return a**2, a*2

def test_multi2(a):
    x = a-np.mean(a)
    y = a-np.mean(a)
    z = np.mean(a)
    return x, y, z

def multiproctest(day):
    print('Day: ', day)
    dfpartial = dict({'year': [day[0].year, day[0].month],   'day': [day[0].day, day[0].month]})
    dfpartial = pd.DataFrame(dfpartial)
    return dfpartial

def get_days_list(datetime_start: str,
                  datetime_end:   str,
                  stationdir,
                  stationid,
                  zstr:           str='000000.pri0',):
    datetime_start = pd.to_datetime(datetime_start)
    datetime_end   = pd.to_datetime(datetime_end)
    days           = pd.date_range(start=datetime_start, end=datetime_end, freq='D')
    dayslist       = np.array([], dtype='datetime64')


    if len(stationdir) == 2:
        dayslist = [np.append(dayslist, day) for day in days if os.path.exists(os.path.join(rootdata, stationdir[0], stationid[0] + day.strftime('%y%m%d') + zstr)) and os.path.exists(os.path.join(rootdata, stationdir[1], stationid[1] + day.strftime('%y%m%d') + zstr))]
    if len(stationdir) == 1:
        dayslist = [np.append(dayslist, day) for day in days if os.path.exists(os.path.join(rootdata, stationdir, stationid + day.strftime('%y%m%d') + zstr))]
    dayslist = np.array(dayslist)
    return dayslist



if __name__ == '__main__':

    run4catalog(datetime_start  = '2023-04-04 00:00', datetime_end    = '2023-04-05 12:00:00')

    
    # from kavutil import org_days, get_data, compute_simple_ratio, compute_banded_ratio, compute_kava, rolling_stats, fusetriggers, get_days_list, get_data3, multiprocchild
    # flag_horizontal = False
    # datetime_start = '2023-04-04'
    # # datetime_end = '2023-04-04 23:59:59'
    # datetime_end = '2023-04-06 12:00:00'
    # datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)
    # # breakpoint()

    # # dayslist = get_days_list(datetime_start, datetime_end, stationdir, stationid)
    # dayslist = get_days_list(datetime_start=datetime_start,
    #                          datetime_end=datetime_end,
    #                          stationdir=stationdir,
    #                          stationid=stationid)
    # print(dayslist)
    # breakpoint()

    
    # dfresult = multiprocparent(dayslist)

    # breakpoint()

    '''
    # data = [Stream(), Stream()]
    # for i in range(len(data)):
    #     data[i], days = get_data2(datetime_start, datetime_end,
    #                         rootdata=rootdata,
    #                         stationdir=stationdir[i],
    #                         stationid=stationid[i],
    #                         flag_horizontal=flag_horizontal,
    #                         daysreturn=True)
    # datatd, datafd = data.copy(), data.copy()
    # time = data[0][0].times(reftime=data[0][0].stats.starttime)
    # timepandas = pd.date_range(start=pd.Timestamp(data[0][0].stats.starttime.datetime),
    #                            periods=data[0][0].stats.npts,
    #                            freq=str(data[0][0].stats.delta)+'S')
    # shift = shifttime/data[0][0].stats.delta

    # p        = Pool(4)
    # input_a  = [1,2,3,4,5]
    # result_a = p.map(test_multi, input_a)
    # print(result_a)
    
    # p2       = Pool(4)
    # input_b  = [data[0][i] for i in range(len(data[0]))]
    # result_b = p2.map(test_multi2, input_b)
    # breakpoint()
    '''


# def plot_overview2():