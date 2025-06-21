import  numpy               as np
import  pandas              as pd
import  matplotlib.pyplot   as plt
from    obspy               import read, Trace, UTCDateTime, Stream
from kav_init import *
from kavutil import *
from datetime import timedelta
from tqdm import tqdm

def create_utc(time, tracestats):
    """
    Create an UTCDateTime object from a time in seconds and a tracestats object.

    Parameters
    ----------
    time : float
        The time in seconds.
    tracestats : stats object, obspy
        The stats object of the trace, containing information like starttime, year, month, day, etc.
    """
    year                = tracestats.starttime.year
    month               = tracestats.starttime.month
    day                 = tracestats.starttime.day
    minutes, seconds    = divmod(time, 60)
    hours, minutes      = divmod(minutes, 60)
    utc_time            = UTCDateTime(year=year, month=month, day=day, hour=int(hours), minute=int(minutes), second=int(seconds))
    return utc_time

def comp_rsam(trace, rsamrate=60, print_as_trace=True):
    """
    Compute the RSAM of a trace.
    
    Parameters
    ----------
    trace : trace object, obspy
        The trace to compute the RSAM of.
    rsamrate : int, optional
        The rate at which the RSAM is computed. The default is 60 seconds.
    sr : int, optional
        The sampling rate of the trace. The default is 200 Hz
    """
    tr2edit     = trace.copy()
    data        = tr2edit.data
    sr          = tr2edit.stats.sampling_rate


    ileft       = np.arange(0, len(data), sr * rsamrate, dtype=int)
    if ileft[-1] == len(data):
        ileft = ileft[:-1]
    iright      = np.array(ileft + sr * rsamrate, dtype=int)
    if iright[-1] > len(data):
        iright[-1] = len(data)

    rsam        = np.zeros(len(ileft))
    for i in range(len(ileft)):

        rsam[i] = np.mean(np.abs(data[ileft[i]:iright[i]] - np.mean(data[ileft[i]:iright[i]])))


    if print_as_trace:
        time            = tr2edit.times(reftime=tr2edit.stats.starttime)
        starttimeitem   = time[ileft[0]]
        endtimeitem     = time[ileft[-1]]

        starttime_rsam  = create_utc(starttimeitem, tracestats= tr2edit.stats)
        endtime_rsam    = create_utc(endtimeitem,   tracestats= tr2edit.stats)

        rsam_trace                  = Trace(data=rsam)
        rsam_trace.stats.starttime  = starttime_rsam
        rsam_trace.stats.npts       = len(rsam)
        rsam_trace.stats.delta      = rsamrate

    return rsam_trace

def rsam_evol(moi=[4], doi=[5], yoi=yoi, rootdata=rootdata, stationdir_org=stationdir, stationdir_data='KAV11', stationid='c0941', merge_traces=True):
    """
    Compute the RSAM of a trace.

    Parameters
    ----------
    moi : list, optional
        The months of interest. The default is [4]. (April 2023)
    doi : list, optional
        The days of interest. The default is [5].
    yoi : int, optional
        The year of interest. The default is yoi.
    rootdata : str, optional
        The path to the data. The default is rootdata.
    stationdir_org : str, optional
        The station directory input for organising days. The default is stationdir.
    stationdir_data : str, optional
        The station directory inut to fetch the data. The default is 'KAV11'.
    stationid : str, optional
        The station id. The default is 'c0941'.
    merge_traces : bool, optional
        Merge the traces. Default to True. For more detail look up obspy.core.stream.Stream.merge().

    LBitzan, 04/2024

    """
    for imoi in moi:
        if doi == []:
            doi = org_days(stationdir_org, imoi = imoi)
        for idoi in doi:
            st, _ = get_data(year=yoi, month=imoi, day=idoi, rootdata=rootdata, stationdir=stationdir_data, stationid=stationid)
            tr    = st[0].copy()
            rsam  = comp_rsam(tr)
            if 'stream_rsam' in locals():
                stream_rsam.append(rsam.copy())
                print("trace appended to stream_rsam")
            else:
                stream_rsam = Stream()
                stream_rsam.append(rsam)
                print("stream_rsam created")
        if len(moi) > 1:  
            doi = []
    
    if merge_traces == True:
        stream_rsam.merge()

    return stream_rsam


def comp_rsam_v2(tr: Trace, rsamrate: int=60):
    trace = tr.copy()
    data = trace.data
    sr = trace.stats.sampling_rate

    ileft = np.arange(0, len(data), sr * rsamrate, dtype=int)
    if ileft[-1] == len(data):
        ileft = ileft[:-1]
    iright = np.array(ileft + sr * rsamrate, dtype=int)
    if iright[-1] > len(data):
        iright[-1] = len(data)
    rsam = np.zeros(len(ileft))
    rsam = [np.mean(np.abs(data[ileft[i]:iright[i]] - np.mean(data[ileft[i]:iright[i]]))) for i in range(len(ileft))]
    
    time_array = pd.date_range(start=pd.to_datetime((trace.stats.starttime).datetime),
                               end=pd.to_datetime((trace.stats.endtime).datetime),
                               freq=f'{rsamrate}S')

    return rsam, time_array

def rsam_compute(datetime_start: str, datetime_end: str, stationdir: str='KAV11', stationid: str='c0941'):

    if 'KavScripts' not in os.getcwd():
        if 'KavScripts' not in os.listdir():
            print('Working repository not found. Check path.')
            return
        else:
            os.chdir('KavScripts')
            print('Changed directory to: ', os.getcwd())
    elif 'KavScripts' in os.getcwd():
        print('Script already started from working repository: ', os.getcwd())

    datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)
    dayslist = get_days_list(datetime_start, datetime_end, stationdir, stationid)
    # preallocate arrays for rsam and time, so that later only the values need to be appended. For rsam it should be an array iof numpy floats, for time we append pandas date_range objects
    # rsam_array = np.array([],dtype='float')
    # time_array = np.array([],dtype='datetime64[ns]')
    rsam_list, time_list = [], []

    for i, idate in enumerate(tqdm(dayslist)):
        print('Processing day: ', idate[0])
        st = get_data3(idate[0], rootdata=rootdata, stationdir=stationdir, stationid=stationid)
        rsam_partial, time_partial = comp_rsam_v2(st[0].copy())
        # rsam_array = np.append(rsam_array, rsam_partial)
        # time_array = np.append(time_array, time_partial)
        rsam_list.append(rsam_partial)
        time_list.append(time_partial)
 
    # concatenate the lists to arrays
    rsam_array = np.concatenate(rsam_list)
    time_array = pd.concat(time_list)

    # create a DataFrame with the rsam and time arrays and save it as a pickle file
    rsam_df = pd.DataFrame({'rsam':rsam_array, 'time':time_array})
    rsam_df.to_pickle(rootdata+'/results/rsam_'+stationdir+'.pkl')
    
    return rsam_df



        




    
'''
arange plot
adjust ylim to reasonable values
plot together with event distribution for scaling

chekc statistcs for patricks events
is there a shark trigger within 2-5 seconds of the event?

'''


# del rsam2tot, rsam1tot, rsam2, rsam1

