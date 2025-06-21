'''
Setup for individual plotting
'''
import numpy as np
# from scipy.io import wav as wav
import matplotlib.pyplot as plt
import pandas as pd
import os as os
import time as time
from tqdm import tqdm

from obspy import UTCDateTime, Stream, Trace
from obspy.core import read, Stream
from kavutil import get_data3, get_days_list
from kav_init import *

# SET PARAMETERS
datetime_start, datetime_end = '2023-04-05', '2023-04-06'
plottime_start, plottime_end = '2023-04-05T00:00:00', '2023-04-05T09:00:00'
setsize                      = (600,450)


stationdir, stationid   = 'KAV11', 'c0491'
days                    = pd.date_range(start=datetime_start, end=datetime_end, freq='D')
dayslist                = np.array([], dtype='datetime64')
listofinclompletedays   = np.array(['2023-07-04',
                                    '2023-05-03',
                                    '2023-06-08',
                                    '2023-07-26'], dtype=str)
# print(listofinclompletedays)
for d in listofinclompletedays:
    if pd.to_datetime(d) in days:
        days = days.drop(pd.to_datetime(d))
# print(days)

daytest = 'c0941230401000000.pri0'
if len(stationdir) == 2:
    dayslist = [np.append(dayslist, day) for day in days if os.path.exists(os.path.join(rootdata, stationdir[0], stationid[0] + day.strftime('%y%m%d') + zstr)) and os.path.exists(os.path.join(rootdata, stationdir[1], stationid[1] + day.strftime('%y%m%d') + zstr))]
elif len(stationdir) != 2:
    dayslist = [np.append(dayslist, day) for day in days if os.path.exists(os.path.join(rootdata, stationdir, daytest))]#stationid + day.strftime('%y%m%d') + zstr))]
dayslist = np.array(dayslist)

print(dayslist)

for i, idate in enumerate(dayslist[0]):
    print(idate)
    # --- Read in data -----------------------------------------------------------------------------------------------------
    st11    = get_data3(idate, stationdir='KAV11',stationid='c0941')
    st04    = get_data3(idate, stationdir='KAV04',stationid='c0939')
    st00    = get_data3(idate, stationdir='KAV00',stationid='c0bdd')
    streams = [st11, st04, st00]

    for streamer in streams:
        streamer[0].stats.channel = 'z'

    st11[0].stats.station, st04[0].stats.station, st00[0].stats.station = 'KAV11', 'KAV04', 'KAV00'
    st11[0].stats.network, st04[0].stats.network, st00[0].stats.network = 'Nggatokae I', 'Nggatoke II','Nggatokae II'

    st11 = st11.slice(starttime=UTCDateTime(plottime_start), endtime=UTCDateTime(plottime_end))
    st04 = st04.slice(starttime=UTCDateTime(plottime_start), endtime=UTCDateTime(plottime_end))
    st00 = st00.slice(starttime=UTCDateTime(plottime_start), endtime=UTCDateTime(plottime_end))
    
    st11.plot(type='dayplot', size=setsize)
    st04.plot(type='dayplot', size=setsize)
    st00.plot(type='dayplot', size=setsize)
    
    # st11, st04, st00 = [streamer.slice(starttime=UTCDateTime(plottime_start), endtime=UTCDateTime(plottime_end)) for streamer in streams]

    # for streamer in streams:
    #     streamer.plot(type='dayplot')

    st11_cut = st11.slice(starttime=UTCDateTime('2023-04-05T00:50:00'), endtime=UTCDateTime('2023-04-05T01:00:00'))
    st11_cut.plot()

breakpoint()