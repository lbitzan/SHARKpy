import numpy as np
import pandas as pd
from obspy.core import read, Stream, Trace
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import os as os
import datetime
from kav_init import rootdata
from matplotlib.dates import DateFormatter
from datetime  import timedelta
import time as time
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import pickle
import plutil as plutil
import kav_init as ini
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.dates as mdates, matplotlib.gridspec as gridspec


# # Moving mean "movmean()"
# def movmean(x, k):
#     """Moving average of x with window size k."""
#     import numpy as np
#     x = np.asarray(x)
#     return np.convolve(x, np.ones(k) / k, mode='valid')

# # Moving standard deviation "movstd()"
# def movstd(x, k):
#     """Moving standard deviation of x with window size k."""
#     import numpy as np
#     x = np.asarray(x)
#     return np.sqrt(movmean(x**2, k) - movmean(x, k)**2)

# compute envelopes of a signal "hl_envelopes()"
# def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
#     """
#     Input :
#     s: 1d-array, data signal from which to extract high and low envelopes
#     dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
#     split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
#     Output :
#     lmin,lmax : high/low envelope idx of input signal s
#     """
#     import numpy as np
#     # locals min      
#     lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
#     # locals max
#     lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
#     if split:
#         # s_mid is zero if s centered around x-axis or more generally mean of signal
#         s_mid = np.mean(s) 
#         # pre-sorting of locals min based on relative position with respect to s_mid 
#         lmin = lmin[s[lmin]<s_mid]
#         # pre-sorting of local max based on relative position with respect to s_mid 
#         lmax = lmax[s[lmax]>s_mid]

#     # global min of dmin-chunks of locals min 
#     lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
#     # global max of dmax-chunks of locals max 
#     lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
#     return lmin,lmax

# Moving mean "myplot"
def myplot(
        axoi        = None, 
        facecolor   = '0.9', 
        gridcolor   = 'w', 
        gridalpha   = 0.5,
        legfontsize = 'x-small',
        legloc      = 'best',
        handles     = None):
    """
    Set customised plot parameters!
    Sets up Grid and background facecolor.
    Input:
    axoi: ax of interest, where customised plotting is applied
    Optional:
    facecol  - 'string'default light grey
    gridcol  - default white
    gridalph - default 0.5
    Output:
    Output is the input axis.
    """
    import matplotlib.pyplot as plt
    
    if axoi is None:
        axoi = plt.gca()
    axoi.grid(color = gridcolor, alpha = gridalpha)
    axoi.set_facecolor(facecolor)
    if handles is not None:
        if len(handles) == 2:
            axoi.legend(handles=handles[0]+handles[1], loc=legloc, fontsize=legfontsize)
        elif len(handles) == 3:
            axoi.legend(handles=handles[0]+handles[1]+handles[2], loc=legloc, fontsize=legfontsize)
    else:
        axoi.legend(loc=legloc, fontsize=legfontsize)
    
    return axoi

def myplot_noleg(axoi=None, facecolor='0.9', gridcolor='w', gridalpha=0.5):
    import matplotlib.pyplot as plt
    
    if axoi is None:
        axoi = plt.gca()
    axoi.grid(color = gridcolor, alpha = gridalpha)
    axoi.set_facecolor(facecolor)
    
    return axoi

def rolling_stats(x, func, window=3):
    """
    Function, for calculation rolling statistics of a timeseries

    Parameters
    ----------
    x : array
        Timeseries.
    func : function
        The corresponding statistical function (e.g. np.mean).
    window : int, optional
        Window size of the rolling window. The default is 3.

    Returns
    -------
    array
        rolling statistics timeseries.

    by P. Laumann, 01/2024
    """

    # Padding array, so result has same length as input
    x = np.pad(x,(0, window-1),'edge')

    return func(sliding_window_view(x, window),-1)


def compute_kava(x, station, frequency_bands, fmin=3, fmax=60, nfft=None, Fs=None, noverlap=None, detrend='none', scale_by_freq=True, cmap='viridis',appl_waterlvl=False):
    ''' 
    x:               input signal of obspy trace object
    frequency_bands: matrice of frequency bands of interest
    fmin:            minimum frequency of interest
    fmax:            maximum frequency of interest
    nfft:            length of the FFT window
    Fs:              sampling frequency
    noverlap:        int. number of points to overlap between segments. 
    detrend:         detrend method
    scale_by_freq:   scale the PSD by the scaling factor 1 / Fs
    cmap:            colormap for plotting
    waterlvl:        water level kava computation
    
    by L.Bitzan, 2024
    '''
    import  numpy               as      np
    import  matplotlib.pyplot   as      plt
    from    obspy.signal        import  PPSD
    from    kav_init            import  waterlvl

    if Fs is None:
        Fs = x.stats.sampling_rate
    if nfft is None:
        nfft = int(Fs)
    if noverlap is None:
        noverlap = int(Fs/2)
    if station == 'KAV00':
        fmin = 0.1

    waterlevel = 0.
    if appl_waterlvl is True:
        waterlevel = waterlvl[station]

    fbands = frequency_bands[station]
    
    d = x.copy()
    d.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    dspec, dfreq, dtime, dcax = plt.specgram(d.data, NFFT=nfft, Fs=Fs, noverlap=noverlap, detrend=detrend, scale_by_freq=scale_by_freq, cmap=cmap)
    plt.close()

    dt      = dtime[1] - dtime[0]

    ####
    if station == 'KAV11':
        numerator   = np.sum(dspec[4:8], axis=0) + np.sum(dspec[14:31], axis=0) + np.sum(dspec[40:49], axis=0)
        denominator = np.sum(dspec[8:13], axis=0) + np.sum(dspec[33:39], axis=0) + np.sum(dspec[49:], axis=0)
    else:
        nbands  = np.shape(fbands)[0]

        numerator_partials      = np.zeros([ nbands, len(dtime) ])
        numerator               = np.zeros(len(dtime))

        for i in range(nbands):
            numerator_partials[i] = np.sum(dspec[fbands[i][0]:fbands[i][1]], axis=0)
            numerator += numerator_partials[i]

    denominator = np.sum(dspec[int(fmin):fmax], axis=0) - numerator + int(22) # waterlevel
    print('new waterlevel')
    kava        = numerator / denominator
    ###
    ### ADAPTED algorhythm
    '''
    nbands  = np.shape(fbands)[0]

    numerator_partials      = np.zeros([ nbands, len(dtime) ])
    numerator               = np.zeros(len(dtime))

    for i in range(nbands):
        numerator_partials[i] = np.sum(dspec[fbands[i][0]:fbands[i][1]], axis=0)
        numerator += numerator_partials[i]

    denominator = np.sum(dspec[int(fmin):fmax], axis=0) - numerator + int(waterlevel)
    kava        = numerator / denominator
    #'''
    ###

    return kava, dtime, dt, dspec, dfreq, dcax


def get_data(year=23, month=4, day=5, rootdata='data', stationdir='KAV11', stationid='c0941', zstr='000000.pri0', estr='000000.pri1', wstr='000000.pri2'):
    '''
    Retrieve data from Kavachi station directories.
    Input:
    year:      int, year of interest
    month:     int, month of interest
    day:       int, day of interest
    rootdata:  string, path to data
    stationdir:string, station directory
    stationid: string, station id
    zstr:      string, z-component
    estr:      string, e-component
    wstr:      string, w-component
    Output:
    st:        obspy stream object
    stri:      obspy stream object all components


    '''

    import numpy as np
    from obspy.core import read
    
    filename    = '{stat}{year}{month:02d}{day:02d}'
    file        = filename.format(stat=stationid, year=year, month=month, day=day)
    
    st          =      read(rootdata +'/'+  stationdir +'/'+ file + zstr)
    stri        = st + read(rootdata +'/'+  stationdir +'/'+ file + estr) + read(rootdata +'/'+  stationdir +'/'+ file + wstr)

    return st, stri

# def compute_banded_ratio(x1, x2, freqbands, shifttime=ini.shifttime, stats=['KAV04','KAV11'], zerophase=True, corners=2, window=20):
#     '''
#     Compute the amplitude ratio of two stations based on their largest characteristic frequency bands.
    
#     Input:\n
#     x1:         obspy stream object, remote station\n
#     x2:         obspy stream object, close station\n
#     freqbands:  dictionary, frequency bands of interest for each station\n
#     shifttime:  float, time shift between stations in seconds\n
#     stats:      list of strings, two items, station names\n
#     zerophase:  bool, True: zero-phase filter, False: minimum-phase filter\n
#     corners:    int, number of corners for bandpass filter [0, 2, 4]\n
#     window:     int, window size for rolling statistics\n

#     Output:\n
#     ratio:      array, amplitude ratio\n
#     ratiotime:  array, time axis for amplitude ratio


#     LuBi, 2024
#     '''
#     # from kavutil import rolling_stats
    
#     a = x1.copy()
#     b = x2.copy()

#     fmin = np.zeros(2)
#     fmax = fmin.copy()
#     ind  = np.zeros(2, dtype=int)


#     for i in range(2):
#         fdiff       = (np.diff(freqbands[stats[i]])).flatten()
#         fdiffmax    = np.amax(fdiff)

#         ind[i]      = np.where(fdiff == fdiffmax)[0][0]

#         fmin[i]     = freqbands[stats[i]][ind[i]][0]
#         if fmin[i] < 0.1:
#             fmin[i] = 0.1
#         fmax[i]     = freqbands[stats[i]][ind[i]][1]
    
#     # if stats[0] in ['KAV00', 'KAV10']:
#     #     a = apply_bb_tf_inv(a)
#     #     a = simulate_45Hz(a)
#     # if stats[1] in ['KAV00', 'KAV10']:
#     #     b = apply_bb_tf_inv(b)
#     #     b = simulate_45Hz(b)

#     a.filter('bandpass', freqmin=fmin[0], freqmax=fmax[0], corners=corners, zerophase=zerophase)
#     b.filter('bandpass', freqmin=fmin[1], freqmax=fmax[1], corners=corners, zerophase=zerophase)

#     a.decimate(factor=2, no_filter=True)
#     b.decimate(factor=2, no_filter=True)

#     ratiotime   = b.times(reftime=b.stats.starttime)
#     idxshift_bandedratio = shifttime / b.stats.delta

#     b_amplitude = rolling_stats(np.abs(b.data), np.amax, window=int(window))
#     a_amplitude = rolling_stats(np.abs(a.data), np.amax, window=int(window))

#     ratio       = b_amplitude[:-int(idxshift_bandedratio)] / a_amplitude[int(idxshift_bandedratio):]
#     ratiotime   = ratiotime[:-int(idxshift_bandedratio)]

#     return ratio, ratiotime

# def compute_simple_ratio(x1, x2, shifttime=2.25, stats=['KAV04','KAV11'], zerophase=True, corners=2, window=20):
#     '''
#     Compute the amplitude ratio of two stations.
    
#     Input:\n
#     x1:         obspy stream object, remote station\n
#     x2:         obspy stream object, close station\n
#     shifttime:  float, time shift between stations in seconds\n
#     stats:      list of strings, two items, station names\n
#     zerophase:  bool, True: zero-phase filter, False: minimum-phase filter\n
#     corners:    int, number of corners for bandpass filter [0, 2, 4]\n
#     window:     int, window size for rolling statistics\n

#     Output:\n
#     ratio:      array, amplitude ratio\n
#     ratiotime:  array, time axis for amplitude ratio


#     LuBi, 2024
#     '''
#     from kav_init import fmin, fmax
    
#     a    = x1.copy()
#     b    = x2.copy()    
    
#     #TODO: define
#     # if stats[0] in ['KAV00', 'KAV10']:
#     #     a = apply_bb_tf_inv(a)
#     #     a = simulate_45Hz(a)
#     # if stats[1] in ['KAV00', 'KAV10']:
#     #     b = apply_bb_tf_inv(b)
#     #     b = simulate_45Hz(b)

#     a.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=corners, zerophase=zerophase)
#     b.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=corners, zerophase=zerophase)

#     a.decimate(factor=2, no_filter=True)
#     b.decimate(factor=2, no_filter=True)

#     ratiotime            = b.times(reftime=b.stats.starttime)
#     idxshift_bandedratio = shifttime / b.stats.delta

#     b_amplitude = rolling_stats(np.abs(b.data), np.amax, window=int(window))
#     a_amplitude = rolling_stats(np.abs(a.data), np.amax, window=int(window))

#     ratio       = b_amplitude[:-int(idxshift_bandedratio)] / a_amplitude[int(idxshift_bandedratio):]
#     ratiotime   = ratiotime[:-int(idxshift_bandedratio)]

#     return ratio, ratiotime


def org_days(stationdir, yoi=23, imoi=4):
    '''
    Organise days of interest in a list of datetime objects.
    Input:
    stationdir: list of strings, station directory
    years:  list of integers, years of interest
    months: list of integers, months of interest
    Output:
    days:   list of integers
    '''

    if stationdir == ['KAV04', 'KAV11']:
        if   (imoi == 2) & (yoi % 4 == 0):
            doi = np.arange(1,30,1)
        # elif (imoi == 2) & (yoi % 4 != 0):
        #     doi = np.arange(1,29,1)
        # elif  imoi in [1,3,5,7,8,10,12]:
        #     doi = np.arange(1,32,1)
        # elif  imoi in [4,6,9,11]:
        #     doi = np.arange(1,31,1)

        elif (imoi == 2) & (yoi == 23):
            doi = np.arange(7,29,1)
        elif imoi == 3:
            doi = np.arange(1,32,1)
        elif imoi == 4:
            doi = np.arange(1,31,1)
        elif imoi == 5:
            doi = [1,2]
            doi = np.concatenate((doi, np.arange(23,32,1)))
        elif imoi == 6:
            doi = [1,2,3,5,6,7]
            doi = np.concatenate((doi, np.arange(9,31,1)))
        elif imoi == 7:
            doi = np.arange(1,4,1)
            doi = np.concatenate((doi, np.arange(5,26,1)))

    # elif stationdir == ['KAV00', 'KAV11']:
    else:
        if   (imoi == 2) & (yoi % 4 == 0):
            doi = np.arange(1,30,1)
        elif (imoi == 2) & (yoi % 4 != 0):
            doi = np.arange(1,29,1)
        elif  imoi in [1,3,5,7,8,10,12]:
            doi = np.arange(1,32,1)
        elif  imoi in [4,6,9,11]:
            doi = np.arange(1,31,1)
    
    days = doi
    return days


def read_catalog(catalogfile = 'catalog.txt'):
    '''
    Read catalogue file.
    Input:
    cataloguefile: string, name of catalogue file
    Output:
    catalogue:     pandas dataframe
    '''
    headers     = ['date','ratio','kavachi_index']
    dtypes      = [str, float, float]

    if catalogfile.endswith('.pkl'):
        catalog = pd.read_pickle(catalogfile)
    else:
        catalog = pd.read_csv(catalogfile, sep=',', header=0, names=headers, parse_dates=['date'])
    return catalog


def fft_slice(data, time, dt, start, end):
    '''
    Parameters
    ----------
    data : array
        data array.
    time : array datetime
        datetime array.
    dt : int
        timestep between sample points in seconds.
    start : datetime
        starttime of slice of timeseries.
    end : datetime
        endtime of slice of timeseries.

    Returns
    -------
    f : array
        computed frequency spectrum.
    y : array
        power, amplitude of frequencies.

    '''
    import scipy as spy
    import numpy as np
    import pandas as pd
    
    df = pd.DataFrame({'data': data,
                       'time': time})
    df.drop(df[df.time <= start].index, inplace=True)
    df.drop(df[df.time >= end  ].index, inplace=True)

    series          = pd.Series(data= df.data)
    series.index    = pd.to_datetime(df.time)
    series          = series.resample("1D").interpolate("linear")

    df = pd.DataFrame({"data": series.values,
                       "time": series.index })
    
    prepdata    = (df.data - df.data.mean())*spy.signal.windows.hann(len(df))
    pdzero      = np.zeros(int(len(prepdata)/2))
    prepdata    = np.concatenate((pdzero, prepdata, pdzero))
    
    fnts        = np.fft.fft(prepdata)
    fs          = 1/ dt
    p2          = abs(fnts) / len(fnts)
    p1          = p2[0:len(fnts) // 2]
    p1[1:-2]    = 2*p1[1:-2]
    
    f           = fs / len(fnts) * (np.arange(len(fnts) // 2))
    
    return f, p1, df
    
# Add colorbar outside of the plot
def add_colorbar_outside(im,ax,return_cbar=False):
    fig = ax.get_figure()
    bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    width = 0.01
    eps = 0.01 #margin between plot and colorbar
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax)
    if return_cbar == True:
        return cbar
    
# Retrieve seismic trace data based on datetime timestamp, station directory, and station ID
def retrieve_trace_data(selected_events, stationdir, stationid):
    """
    Retrieve seismic trace data corresponding to the events provided.
    Based on a numpy datetime timestamp, stationname and id.

    Parameters
    ----------
    selected_events : pd.DataFrame
        DataFrame containing the selected events.
    stationdir : str
        Subdirectory to access the data.
    stationid : str
        Station ID to access the data.

    Returns
    -------
    Stream
        Stream object containing the retrieved trace data.
    """
    from obspy import UTCDateTime, Stream, Trace, read
    from kav_init import rootdata
    # Determine which column to use for event time
    if 'puretime' in selected_events.columns:
        tvar = 'puretime'
    elif 'date' in selected_events.columns:
        tvar = 'date'
    elif 'Eventtime(UTC)' in selected_events.columns:
        tvar = 'Eventtime(UTC)'
    else:
        raise ValueError("No valid time column found in the selected events DataFrame.")

    stream = Stream()
    timestamps = []

    for index, event in selected_events.iterrows():
        event_time = event[tvar]
        
        start_time  = UTCDateTime(event_time - timedelta(seconds=10))  # 10 seconds before the event
        end_time    = UTCDateTime(event_time + timedelta(seconds=30))  # 30 seconds after the event

        # Read and append data files to the stream
        if start_time.day != end_time.day:
            filename1   = f"{rootdata}/{stationdir}/{stationid}{start_time.strftime('%y%m%d')}000000.pri0"
            filename2   = f"{rootdata}/{stationdir}/{stationid}{end_time.strftime('%y%m%d')}000000.pri0"
            if not os.path.exists(filename1) or not os.path.exists(filename2):
                print(f"Files for {start_time.strftime('%Y-%m-%d')} or {end_time.strftime('%Y-%m-%d')} not found.\n Skip this events!\n")
                continue
            tr1, tr2    = read(filename1)[0], read(filename2)[0]
            trace       = tr1.__add__(tr2)
            trace       = trace.slice(start_time, end_time, nearest_sample=True)
            stream.append(trace)
            timestamps.append(event_time)
        else:
            filename    = f"{rootdata}/{stationdir}/{stationid}{event_time.strftime('%y%m%d')}000000.pri0"
            trace       = read(filename, starttime=start_time, endtime=end_time)[0]
            stream.append(trace)
            timestamps.append(event_time)
    timestamps = pd.to_datetime(np.array(timestamps))
    return stream, timestamps 

# Reads in event catalogs as dataframes from txt or pkl files
def read_catalogs_from_files(
        catalog_1:      str,
        catalog2:       str,
        drop_details:   bool = False):
    """
    Read two catalogs from files and return them as pandas DataFrames.
    Parameters:
    -----------
    catalog_1: str
        Name of the first catalog file. Individual calibration for shark results.
    catalog_2: str
        Name the second catalog file. Individual calibration for array results.
    Returns:
    --------
    df1: pd.DataFrame
        DataFrame containing the first catalog.
    df2: pd.DataFrame
        DataFrame containing the second catalog.
    events1: pd.Series
        Series containing the events of the first catalog.
    events2: pd.Series
        Series containing the events of the second catalog.

    """
    from kav_init import rootdata, kav_angle, flag_matching
    catalogfile, catalogfile_array = catalog_1, catalog2

    if flag_matching == True:
        headers             = ['event1_idx','event1time','event2_idx','time_diff']
        catalog             = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\t', names=headers)
        catalog['puretime'] = pd.to_datetime(catalog['event1time'])
        catalog.set_index(catalog['event1_idx'], inplace=True)
        catalog.drop(columns=['event1time', 'time_diff','event1_idx'], inplace=True)
    elif flag_matching == False and catalogfile.endswith('.pkl'):
        # Import shark catalog
        print('Reading in catalog from pkl file.')
        catalog = pd.read_pickle(rootdata+'/'+catalogfile)
        catalog['puretime'] = pd.to_datetime(catalog['date'])
        if 'Norm_amplitude_ratio[]' in catalog.columns:
            catalog.rename(columns={'Norm_amplitude_ratio[]':'ratio'}, inplace=True)
            catalog.rename(columns={'Adv_KaVA_idx':'shark'}, inplace=True)
        if 'kavachi_index' in catalog.columns:
            catalog.rename(columns={'kavachi_index':'shark'}, inplace=True)
        catalog.drop(columns=['date'], inplace=True)
        catalog.set_index(np.arange(len(catalog)),inplace=True)
    else:
        headers             = ['date','ratio','shark']
        catalog             = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\s+', names=headers) #parse_dates=['date'])
        catalog['puretime'] = pd.to_datetime(catalog.index + ' ' + catalog.date, format='%d-%b-%Y %H:%M:%S,')
        if drop_details == True:
            catalog.drop(columns=['ratio','shark'], inplace=True)
        catalog.drop(columns=['date'], inplace=True)
        catalog.set_index(np.arange(len(catalog)),inplace=True)

    # Import array catalog
    headers_arr         = ['dstart','tstart','dend','tend','baz','v_app','rmse']
    catalog_arr         = pd.read_csv(rootdata+'/'+catalogfile_array, header=0, sep='\s+', names=headers_arr)
    
    catalog_arr.drop(catalog_arr[catalog_arr.baz <= kav_angle[0] ].index, inplace=True)
    catalog_arr.drop(catalog_arr[catalog_arr.baz >= kav_angle[1] ].index, inplace=True)
    
    catalog_arr['puretime']     = pd.to_datetime(catalog_arr.dstart + ' ' + catalog_arr.tstart, format='%Y-%m-%d %H:%M:%S.%f')
    
    if drop_details == True:
        catalog_arr.drop(columns    =['dstart','tstart','dend','tend','v_app','rmse'], inplace=True)
        df2 = pd.DataFrame({'event':    catalog_arr['puretime']}).set_index(catalog_arr.index)
        df1 = pd.DataFrame({'event':    catalog['puretime']    }).set_index(catalog.index)
        events1, events2 = df1.event, df2.event

        return df1, df2, events1, events2

    else:
        catalog_arr['puretime_end'] = pd.to_datetime(catalog_arr.dend + ' ' + catalog_arr.tend, format='%Y-%m-%d %H:%M:%S.%f')
        catalog_arr.drop(columns    =['dstart','tstart','dend','tend'], inplace=True)
        df2 = catalog_arr.copy()
        df1 = catalog.copy()

        return df1, df2
    
def fusetriggers(kavatrigger, ratiotrigger, ratiotime, triggerresttime = 20):
    '''
    Fuse triggers of kava and ratio
    
    Parameters:
    -----------
    kavatrigger : np.array
        Trigger array for kava index
    ratiotrigger : np.array
        Trigger array for amplitude ratio. Has to be of same length as kavatrigger.
    ratiotime : np.array
        Time array for amplitude ratio. Has to be of same length as kavatrigger.
    nbreak : int
        Number of seconds to pause before next event can be triggered. Default is 10 seconds.

    Returns:
    --------
    eventmarker : np.array
        Array of triggered events.

    07/2024, created LuB
    '''
    nbreak  = triggerresttime # seconds to break before next event
    # print(nbreak)
    switch  = True
    a1, b1    = kavatrigger.copy(), ratiotrigger.copy()
    step1   = np.zeros_like(a1)
    dt      = ratiotime[1] - ratiotime[0]
    counter = 0

    aend    = np.zeros_like(a1)
    aend[1:]= np.where(a1[1:] - a1[:-1] == 1, aend[1:] + 1, aend[1:])
    aend[0] = 0

    # astart      = np.zeros_like(a1)
    # astart[1:]  = np.where(a1[1:] - a1[:-1] == -1, astart[1:] + 1, astart[1:])
    # astart[0]   = 0

    # snip1 = int(50*10/dt)
    # snip2 = int(100*60/dt)
    # plt.plot(ratiotime[snip1:snip2], aend[snip1:snip2], 'r-', label= 'aend', alpha=.3)
    # plt.plot(ratiotime[snip1:snip2], a1[snip1:snip2], 'b:', label= 'a1', alpha=.3)
    # plt.plot(ratiotime[snip1:snip2], b1[snip1:snip2], color='orange', label= 'b1', alpha=.3)
    # plt.legend(loc='best')
    # plt.show()
    
    # switchrec = np.array([False]*len(a1))
    # counterrec = np.zeros_like(a1)

    for i in range(len(a1)):
        # switchrec[i] = switch
        # counterrec[i] = counter
        if aend[i] == 1:
            counter = 0
        
        if a1[i] == 1 and b1[i] == 1 and switch == True:
            step1[i]    = 1
            switch      = False
            counter     = 0
        elif a1[i] == 0 and switch == False and counter >= (nbreak/dt):
            switch = True

        counter += 1
    
    eventmarker = step1
    # breakpoint()
    return eventmarker

def fusetriggers_v2(pause: float | None):
    # breakpoint()
    # --- 0. Preallocations
    t       = ini.kavatime_slim
    q0      = ini.shark_qualifier_slim
    q2      = ini.pre_peaks_idx
    if pause is None:
        pause = ini.triggerresttime
    pausebit= int(pause/(ini.dtspec))


    predetect                       = np.zeros_like(q0, dtype=int)
    drop, rise, prevent, detections = predetect.copy(), predetect.copy(), predetect.copy(), predetect.copy()

    # --- 1. Finalise detections
    # 1.0: Predetections, Rise and Drop of shark_qualifier
    predetect[q2]   = 1
    drop[1:]        = np.where((q0[:-1] == 1) & (q0[1:] == 0), 1, 0) # DROP
    rise[1:]        = np.where((q0[:-1] == 0) & (q0[1:] == 1), 1, 0) # RISE

    # # 1.1: Prevent: Scan and mark regions to prevent false detections
    i = 0
    while i < len(q0):
        if predetect[int(i)] == 1 and prevent[int(i)] == 0:
            start   = int(i)
            end     = int(i + pausebit)

            while True:
                # Step 1: Look for rise within pause
                if np.any(rise[int(i+1):int(min(end, len(q0)))] == 1):
                    rise_index  = i + 1 + np.argmax(rise[int(i+1):int(min(end, len(q0)))] == 1)

                    # Then go to next drop
                    if np.any(drop[int(rise_index):] == 1):
                        drop_index  = rise_index + np.argmax(drop[int(rise_index):] == 1)
                        end         = drop_index + pausebit
                        i           = drop_index  # For next rise check
                        continue  # Repeat Step 1
                    else:
                        break

                # Step 2: If pause ends and q0 is still 1
                if end < len(q0) and q0[int(end)] == 1:
                    if np.any(drop[int(end):] == 1):
                        drop_index  = end + np.argmax(drop[int(end):] == 1)
                        end         = drop_index + pausebit
                        i           = drop_index
                        continue  # Go back to Step 1
                    else:
                        break

                # Nothing more to do
                break

            prevent[int(start):int(min(end, len(prevent)))] = 1
            i = end
        else:
            i += 1

    # 1.2: Final detections
    detections[1:] = np.where((prevent[:-1] == 0) & (prevent[1:] == 1), 1, 0) # Detections are marked where prevent changes from 0 to 1

    # --- 2. Return
    ini.peaks_idx   = np.where(detections == 1)[0]
    print(f"--> Of {len(ini.pre_peaks_idx)} pre-detections, {len(ini.peaks_idx)} detections remain after filtering. Entry added to ini class.")
    detect_time     = t[ini.peaks_idx]
    detect_shark    = ini.kavaprod2[ini.peaks_idx]

    # --- 2.1 Plot control
    if hasattr(ini, 'visualise_detectionfilter'):
        if ini.visualise_detectionfilter is True:
            min20           = 3333*.75
            slot            = 3
            sequ            = np.arange(min20*(slot-1),min20*slot+1, dtype=int)
            prepeaksinsequ  = ini.pre_peaks_idx[(ini.pre_peaks_idx < min20*slot) & (ini.pre_peaks_idx >= min20*(slot-1))]
            peaksinsequ     = ini.peaks_idx[(ini.peaks_idx < min20*slot) & (ini.peaks_idx >= min20*(slot-1))]
            peaksrmvinsequ  = np.setdiff1d(prepeaksinsequ, peaksinsequ)

            fig = plt.figure(figsize=(7,4))
            gs = gridspec.GridSpec(4, 1, hspace=0)
            axs = [fig.add_subplot(gs[i]) for i in range(4)]

            norm = ini.kavaprod2_slim / ini.kavaprod2_slim.max()
            axs[0].set_title(f"{ini.idate} UTC+11")
            axs[0].plot(        t[sequ],            norm[sequ], label='shark', color='tab:blue', alpha=0.7)
            axs[0].scatter(     t[prepeaksinsequ],  norm[prepeaksinsequ], color='orange', label='peaks', alpha=1)

            axs[1].plot(        t[sequ],            rise[sequ], color='tab:purple', label='rise')
            axs[1].plot(        t[sequ],            drop[sequ], color='tab:green', label='drop')
            axs[1].fill_between(t[sequ], 0, q0[sequ], color='tab:orange', label='shark\nthreshold', alpha=.5)

            axs[2].plot(        t[sequ],            prevent[sequ], 'k-', lw=0.5, alpha=1)
            axs[2].fill_between(t[sequ], 0, prevent[sequ], facecolor='none', edgecolor='k', hatch='////', label='suppress\ndetections', alpha=.5)

            axs[3].plot(        t[sequ],            norm[sequ], label='shark', color='tab:blue', alpha=0.7)
            axs[3].scatter(     t[peaksinsequ],     norm[peaksinsequ], color='red', label='detections')#, s=50)
            axs[3].scatter(     t[peaksrmvinsequ],  norm[peaksrmvinsequ], color='orange', label='peaks')#, s=50)
            axs[3].set_xlabel('time')
            # axs[3].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            # fig.autofmt_xdate()

            [ax.vlines(t[peaksinsequ], -.1, 1.2, color='r', ls='--', lw=0.5, alpha=0.8) for ax in axs]
            [ax.set_ylim(-.1, 1.2) and ax.set_yticks([0, 1]) and ax.legend(loc='upper right',fontsize=8) for ax in axs  for ax in axs]
            [ax.spines['top'].set_visible(False) for ax in axs[1:]]
            [axs[i].spines['bottom'].set_visible(False) for i in range(len(axs)-1)]

            plt.tight_layout()
            plt.savefig(os.path.join(ini.rootouts, 'detections_filtered_'+ini.idate.strftime('%Y%m%d')+'.png'), dpi=300)
            plt.show()

    return

def select_events(df1,
                  year=None,
                  month=None,
                  day=None,
                  hour=None,
                  minute=None):
    """
    Select specific events from the catalog based on year, month, day, hour, and optional minute.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing the catalog.
    year : int
        Year of the events.
    month : int
        Month of the events.
    day : int
        Day of the events.
    hour : int
        Hour of the events.
    minute : int, optional
        Minute of the events, by default None.

    Returns
    -------
    pd.Series
        DataFrame containing the selected events.
    """
    selected_events = None

    if minute is not None:
        if type(minute) == int:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month) & (df1['puretime'].dt.day == day) & (df1['puretime'].dt.hour == hour) & (df1['puretime'].dt.minute == minute)]
        elif type(minute) == list:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month) & (df1['puretime'].dt.day == day) & (df1['puretime'].dt.hour == hour) & (df1['puretime'].dt.minute >= minute[0]) & (df1['puretime'].dt.minute <= minute[1])]
    
    if minute is None:
        if type(hour) == list:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month) & (df1['puretime'].dt.day == day) & (df1['puretime'].dt.hour >= hour[0]) & (df1['puretime'].dt.hour <= hour[1])]
        elif type(hour) == int:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month) & (df1['puretime'].dt.day == day) & (df1['puretime'].dt.hour == hour)]
            
    if hour == None and minute == None:
        if type(day)== int:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month) & (df1['puretime'].dt.day == day)]
        elif type(day) == list:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month) & (df1['puretime'].dt.day >= day[0]) & (df1['puretime'].dt.day <= day[1])]
            #
    if day == None and hour == None and minute == None:
        # print("Check")
        if type(month) == int:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month == month)]
        elif type(month) == list:
            selected_events = df1[(df1['puretime'].dt.year == year) & (df1['puretime'].dt.month >= month[0]) & (df1['puretime'].dt.month <= month[1])]
            #
    if month == None and day == None and hour == None and minute == None:
        selected_events = df1[(df1['puretime'].dt.year == year)]
        #
    if year == None and month == None and day == None and hour == None and minute == None:
        selected_events = df1

    if selected_events is None or selected_events.empty:
        print('No events found for the specified time.')
        return None
    else:
        return selected_events


def select_events_v2(df:             pd.DataFrame,
                     datetime_start: pd.Timestamp | str,
                     datetime_end:   pd.Timestamp | str,
                     asstring=False):
    '''
    Select events from catalog, in between on start-/end date.

    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame containing the catalog.
    datetime_start: datetime64 | str
        Start date of interest.
    datetime_end: datetime64 | str
        End date of interest.
    asstring: bool, optional
        If True, read in datetime_start/_end as str.

    Returns:
    --------
    selected_events: pd.DataFrame
        DataFrame containing the selected events.

    @ LB 2024/08/09
    '''
    if asstring:
        datetime_start = pd.to_datetime(datetime_start)
        datetime_end   = pd.to_datetime(datetime_end)
    # Check wether either the key 'puretime' or 'date' is present in the DataFrame
    if 'puretime' in df.columns:
        tvar = 'puretime'
    elif 'date' in df.columns:
        tvar = 'date'
    elif 'Eventtime(UTC)' in df.columns:
        tvar = 'Eventtime(UTC)'
    else:
        raise ValueError('No datetime column found in DataFrame. Please provide a DataFrame with a "puretime", "date" or "Eventtime(UTC)" column.')

    print(f"Column '{tvar}' found in dataframe.")
    selected_events = df[(df[tvar] >= datetime_start) & (df[tvar] <= datetime_end)]

    # if 'puretime' in df.columns:
    #     selected_events = df[(df['puretime'] >= datetime_start) & (df['puretime'] <= datetime_end)]
    #     print(' "puretime" column found in DataFrame.')
    # elif 'date' in df.columns:
    #     selected_events = df[(df['date']     >= datetime_start) & (df['date']     <= datetime_end)]
    #     print('Found "date" column in DataFrame containing datetimes.')
    # else:
    #     print('No datetime column found in DataFrame.')
    #     return None
    if selected_events.empty:
        raise ValueError(f'No events found between {datetime_start} and {datetime_end}. Please check the catalog.')
    else:
        print(f'Selected {len(selected_events)} events between {datetime_start} and {datetime_end}.')
        return selected_events

    
def get_data3(day:            pd.Timestamp,
              rootdata:       str='D:/data_kavachi_both',
              stationdir:     str='KAV11',
              stationid:      str='c0941',
              zstr:           str='000000.pri0',
              flag_horizontal:bool=False,
              estr:           str='000000.pri1',
              wstr:           str='000000.pri2'):
    zfilename = stationid + day.strftime('%y%m%d') + zstr
    setpath = os.path.join(rootdata, stationdir, zfilename)
    stream = read(setpath)
    if flag_horizontal:
        xfilename, yfilename = stationid + day.strftime('%y%m%d') + estr, stationid + day.strftime('%y%m%d') + wstr
        stream += read(os.path.join(rootdata, stationdir, xfilename))
        stream += read(os.path.join(rootdata, stationdir, yfilename))
    return stream

def get_data4(day:            pd.Timestamp,
              rootdata:       str='D:/data_kavachi_both',
              stationdir:     str='KAV11',
              stationid:      str='c0941',
              zstr:           str='000000.pri0',
              flag_horizontal:bool=ini.use_three_components,
              estr:           str='000000.pri1',
              wstr:           str='000000.pri2'):
    '''
    Reads in data for a given day and station.

    Parameters
    ----------
    day : pd.Timestamp
        The day for which to read the data.
    rootdata : str, optional
        The root directory where the data is stored. Default is 'D:/data_kavachi_both'.
    stationdir : str, optional
        The directory of the station. Default is 'KAV11'.
    stationid : str, optional
        The ID of the station. Default is 'c0941'.
    zstr : str, optional
        The string to append for the vertical component file. Default is '000000.pri0'.
    flag_horizontal : bool, optional
        Flag to indicate whether to read horizontal components. Default is True.
    estr : str, optional
        The string to append for the east component file. Default is '000000.pri1'.
    wstr : str, optional
        The string to append for the west component file. Default is '000000.pri2'.

    Returns
    -------
    Stream
        ObsPy Stream object containing the read data.
    '''

    zfilename   = stationid + day.strftime('%y%m%d') + zstr
    setpath     = os.path.join(rootdata, stationdir, zfilename)
    stream      = read(setpath)
    if flag_horizontal:
        xfilename, yfilename = stationid + day.strftime('%y%m%d') + estr, stationid + day.strftime('%y%m%d') + wstr
        if os.path.exists(os.path.join(rootdata, stationdir, xfilename)) and os.path.exists(os.path.join(rootdata, stationdir, yfilename)):
            stream += read(os.path.join(rootdata, stationdir, xfilename))
            stream += read(os.path.join(rootdata, stationdir, yfilename))
        else:
            print('--> No horizontal data found for', stationid, 'on', day.strftime('%y%m%d'))
    return stream

def get_days_list(datetime_start: str,
                  datetime_end:   str,
                  stationdir,
                  stationid,
                  zstr:           str='000000.pri0'):
    '''
    Get a list of days with available data for a specific station and time range.

    Parameters
    ----------
    datetime_start : str
        The start of the time range.
    datetime_end : str
        The end of the time range.
    stationdir : str
        The directory of the station.
    stationid : str
        The ID of the station.
    zstr : str, optional
        The string to append for the vertical component file. Default is '000000.pri0'.

    Returns
    -------
    np.ndarray
        Array of days with available data.
    '''
    datetime_start = pd.to_datetime(datetime_start)
    datetime_end   = pd.to_datetime(datetime_end)
    days           = pd.date_range(start=datetime_start, end=datetime_end, freq='D')
    dayslist       = np.array([], dtype='datetime64')
    listofinclompletedays = np.array(['2023-07-04','2023-05-03','2023-06-08','2023-07-26'], dtype=str)
    
    # print(listofinclompletedays)
    days = days.drop([pd.to_datetime(d) for d in listofinclompletedays if pd.to_datetime(d) in days])

    if len(stationdir) == 2:
        dayslist = [np.append(dayslist, day) for day in days if os.path.exists(os.path.join(rootdata, stationdir[0], stationid[0] + day.strftime('%y%m%d') + zstr)) and os.path.exists(os.path.join(rootdata, stationdir[1], stationid[1] + day.strftime('%y%m%d') + zstr))]
    if len(stationdir) != 2:
        dayslist = [np.append(dayslist, day) for day in days if os.path.exists(os.path.join(rootdata, stationdir, stationid + day.strftime('%y%m%d') + zstr))]
    dayslist = np.array(dayslist)
    if len(dayslist) == 0:
        if not os.path.exists(rootdata):
            raise FileNotFoundError(f"Data root directory '{rootdata}' does not exist.")
    return dayslist


def get_data2(datetime_start: str,
              datetime_end:   str,
              rootdata:       str='D:/data_kavachi_both',
              stationdir:     str='KAV11',
              stationid:      str='c0941',
              zstr:           str='000000.pri0',
              flag_horizontal:bool=False,
              estr:           str='000000.pri1',
              wstr:           str='000000.pri2',
              daysreturn:     bool=False):

    timer_start    = time.time()
    datetime_start = pd.to_datetime(datetime_start)
    datetime_end   = pd.to_datetime(datetime_end)
    days           = pd.date_range(start=datetime_start, end=datetime_end, freq='D')
    stream         = Stream()
    if daysreturn:
        days_return = np.array([], dtype='datetime64')
    
    print('Fetching data from ', datetime_start, ' to ', datetime_end, ' for station ', stationid,'/', stationdir)
    for _, day in enumerate(tqdm(days)):
        print(day)
        zfile = stationid + day.strftime('%y%m%d') + zstr
        # Check wether the file exists. If not, skip the day, continue with the next day, and print a warning.
        if not os.path.exists(os.path.join(rootdata, stationdir, zfile)):
            print(f'File {zfile} not found. Skipping day {day}.')
            continue
        st = read(os.path.join(rootdata, stationdir, zfile))
        stream += st
        if flag_horizontal:
            xfile, yfile = stationid + day.strftime('%y%m%d') + estr, stationid + day.strftime('%y%m%d') + wstr
            if not os.path.exists(os.path.join(rootdata, stationdir, xfile)):
                print(f'File {xfile} not found. Skipping day {day}.')
                continue
            if not os.path.exists(os.path.join(rootdata, stationdir, yfile)):
                print(f'File {yfile} not found. Skipping day {day}.')
                continue

            st      = read(os.path.join(rootdata, stationdir, xfile))
            stream += st
            st      = read(os.path.join(rootdata, stationdir, yfile))
            stream += st
        if daysreturn:
            days_return = np.append(days_return, day)

    # stream.merge()
    stream = stream.slice(starttime=UTCDateTime(datetime_start), endtime=UTCDateTime(datetime_end))
    timer_end = time.time()

    print('Data fetched in ', np.floor(timer_end - timer_start), 'seconds.')
    if daysreturn:
        return stream, days_return
    else:
        stream


def unify_triggertraces(kavaprodtrig, timespectral, ratiotrigger2, tampl, shiftspectral):
        if len(timespectral) > len(tampl):
            kavatrigger     = kavaprodtrig
            ratiotrigger    = np.round(np.interp(timespectral, tampl, ratiotrigger2))
        elif len(timespectral) < len(tampl): # standard case
            ratiotrigger    = ratiotrigger2
            kavatrigger     = np.round(np.interp(tampl, timespectral[:-np.int64(shiftspectral)], kavaprodtrig))
        elif len(timespectral) == len(tampl):
            kavatrigger, ratiotrigger     = kavaprodtrig, ratiotrigger2
            print("Number of indices in kava time and ampl time are equal.")
        else:
            print("Error in unify_triggertraces function.")

        return kavatrigger, ratiotrigger


def save_df_catalog(df: pd.DataFrame,
                    outputcatalogpkl: str,
                    outputcatalog: str = None):
    df['Amplitude_ratio']   = df['Amplitude_ratio'].round(decimals=2)
    df['Kavaproduct']       = df['Kavaproduct'].round(decimals=2)
    df.to_pickle(outputcatalogpkl)
    if outputcatalog is not None:
        df.to_csv(outputcatalog, sep='\t', index=False)
    return


def define_ylim(ax, ydata):
    '''
    Define the limits for the y-axis of a pyplot axis object. The limits are set symmetrically around zero.
    Their values is determined by the maximum absolute values of the plotted data with added 10 percent.

    Parameters:
    -----------
    ax: pyplot.axis
        Axis object to set the y-axis limits for.
    trace: obspy.Trace
    '''


    ylim = np.max(np.abs(ydata))
    ylim = 1.1 * ylim
    ax.set_ylim([-ylim, ylim])
    return


def comp_adv_kava(kava1, kava2, kava1time, shifttime=ini.shifttime,window=3, enhanced_separate=False, sigma=2):
    '''
    Compute the advanced KaVa index (also SHARK) as product by shifting the first KaVa index by a given time.
    The second KaVa index is not shifted.
    The time shift is given in seconds and is set in the ini file.
    The function returns the advanced KaVa index as a numpy array.
    The function is used in the compute_kava function.
    Parameters:
    ----------
    kava1: numpy array
        The first KaVa index.
    kava2: numpy array
        The second KaVa index.
    shifttime: int
        The time shift in seconds. Default is 0.
    window: int
        The window size for the rolling statistics. Default is 3.
    enhanced_separate: bool
        If True, the function returns the kava product and the advanced KaVa index separately. Default is False.
    kava1time: numpy array
        The time array of the first KaVa index.
    sigma: int
        The standard deviation for the Gaussian filter. Default is 2.

    Returns:
    -------
    kava_product: numpy array
        The advanced KaVa index as a numpy array.
    kava_product_export: numpy array
        The advanced KaVa index as a numpy array.
    '''
    # Multipy the first KaVa index with the second KaVa index
    dt                  = kava1time[1] - kava1time[0]
    shiftsp             = shifttime/dt
    kava_product        = kava2[:-np.int64(shiftsp)] * kava1[np.int64(shiftsp):]

    # Enhance signal
    maxenvelope         = rolling_stats(kava_product, np.amax, window=window)
    smoothed_envelope   = gaussian_filter(maxenvelope, sigma=sigma)

    # Export signal
    kavaproduct_export  = smoothed_envelope
    if enhanced_separate is True:
        return kava_product, kavaproduct_export
    else:
        return kavaproduct_export

def read_frequency_bands(freq_band_info_pkl_file, station=None):
    # Load frequency bands from the pickle file
    fbands_df = pd.read_pickle(freq_band_info_pkl_file)
    # Create a dictionary to store frequency bands
    frequency_bands = {}

    # Iterate through the columns of the DataFrame
    for column in fbands_df.columns:
        left_freq = fbands_df[column]['left_freq']
        right_freq = fbands_df[column]['right_freq']

        # Create a list of frequency ranges
        freq_ranges = [[left_freq[i], right_freq[i]] for i in range(len(left_freq))]

        # Assign the list to the dictionary with the column name as the key
        frequency_bands[column] = freq_ranges
    if station is not None and column in fbands_df.columns:
        frequency_bands = frequency_bands[station]
    return frequency_bands

def kava_sumup_and_ratio_1(fbands, dspec, dfreq, dtime, station, waterlevel, empiric_values: float = 0):
    """
    Compute the sum of the spectrogram values over the frequency bands and return the ratio.
    """

    # Initialize numerator and denominator
    numerator , denominator, artifact = np.zeros(len(dtime)), np.zeros(len(dtime)), np.zeros(len(dtime))

    # Compute the sum for each frequency band
    for band in fbands:
        fmin_band, fmax_band = band
        if fmin_band < ini.fminbandpass[station]:
            fmin_band = ini.fminbandpass[station]
        if fmax_band > ini.fmaxbandpass[station]:
            fmax_band = ini.fmaxbandpass[station]
            print('-> fmax_band > fmax\n fmax_band set to fmax')
        print('--> Frequency band: ', fmin_band, '-', fmax_band)

        # Find the indices corresponding to the frequency range
        freq_indices = np.where((dfreq >= fmin_band) & (dfreq <= (fmax_band+empiric_values)))[0]

        # Sum the spectrogram values over the frequency range
        numerator += np.sum(dspec[freq_indices], axis=0)
    
    # Compute the artifact if applicable
    if ini.exclude_artifact_fbands is True and ini.fbands_artifact[station] is not None:
        for band in ini.fbands_artifact[station]:
            fmin_artifact, fmax_artifact = band
            if fmin_artifact < ini.fminbandpass[station]:
                fmin_artifact = ini.fminbandpass[station]
            if fmax_artifact > ini.fmaxbandpass[station]:
                fmax_artifact = ini.fmaxbandpass[station]
                print('-> fmax_artifact > fmax\n fmax_artifact set to fmax')
            freq_indices_artifact = np.where((dfreq >= fmin_artifact) & (dfreq <= (fmax_artifact)))[0]
            artifact += np.sum(dspec[freq_indices_artifact], axis=0)
        print('--> Artifact frequency bands excluded from denominator.')

    # Compute the numerator
    denominator = np.sum(dspec[np.where((dfreq >= ini.fminbandpass[station]) & (dfreq <= ini.fmaxbandpass[station]))[0]], axis=0) - numerator + waterlevel - artifact

    # Compute kava
    kava        = numerator / denominator

    # Handle edge cases
    kava[np.isnan(kava)] = 0  # Replace NaNs with 0
    kava[np.isinf(kava)] = 0  # Replace infinities with 0

    return kava



def compute_kava_3(x, 
                station, 
                fmin=None, 
                fmax=None, 
                appl_waterlvl=ini.flag_wlvl, 
                flag_new_freqbands=ini.flag_compute_freqbands, 
                scale_by_freq=True,
                filter='gaussian', 
                freq_band_pkl='multimodal_freqanalysis_output.pkl',
                nfft=None,
                noverlap=None,
                Fs=None):
    '''
    Input:
    -----------
    x:               input signal of obspy trace object
    fmin:            minimum frequency of interest
    fmax:            maximum frequency of interest
    appl_waterlvl:   apply water level to kava computation
                        False | 'constant' | True
    flag_new_freqbands:  True | False
    filter:         'gaussian' | 'gaussian1d' | None
    freq_band_pkl:  path to frequency band information

    Output:
    -----------
    kava:           kava index
    dspec:          spectrogram
    dfreq:          frequency axis
    dtime:          time axis
    dt:             time step
    dspec_all:      spectrogram of all components
    '''
    if fmin is not None:
        ini.fminbandpass[station] = fmin
    if fmax is not None:
        ini.fmaxbandpass[station] = fmax

    # --- Setup data
    traces = x.copy()
    traces.filter('bandpass', freqmin=ini.fminbandpass[station], freqmax=ini.fmaxbandpass[station], corners=ini.bandpasscorners, zerophase=True)

    # --- Set spectrogram parameters
    if Fs is not None:
        pass
    # Check if ini has attribute 'samplingrate' for the station
    if Fs is None:
        if hasattr(ini, 'samplingrate'): # and station in ini.samplingrate:
            Fs               = ini.samplingrate
        else:
            ini.samplingrate = x[0].stats.sampling_rate
            Fs               = ini.samplingrate

    if nfft is None:
        nfft        = int(Fs)
    if noverlap is None:
        if hasattr(ini, 'specnoverlap_factor'):
            noverlap    = int(Fs/ini.specnoverlap_factor)
        else:
            noverlap    = int(Fs*.75) # int(Fs/2)
    if appl_waterlvl is True:
        waterlvl        = ini.waterlvl[station]
    elif appl_waterlvl == 'constant':
            waterlvl    = 1e-10
    elif appl_waterlvl == False:
            waterlvl    = 0.
    else:
        raise ValueError("Invalid value for waterlvl. Use 'constant', True, or False.")
    cmap = plt.get_cmap(ini.cmapnamespec)
    
    # --- Compute spectrogram.. ---
    dspec, dfreq, dtime, dcax = plt.specgram(traces[0].data.copy(),
                 Fs             = Fs,
                 pad_to         = 2**9, 
                 NFFT           = nfft,
                 detrend        = 'mean',
                 scale_by_freq  = scale_by_freq,
                 cmap           = cmap, #  scale = 'linear',Fc = 0,mode='default', sides='default', window='window_hanning',
                 )
    plt.close()
    # --- ..and for horizontal components ---
    if ini.use_three_components is  True:
        dspec_e,_,_,_ = plt.specgram(traces[1].data.copy(), Fs = Fs, pad_to = 2**9, NFFT = nfft, detrend = 'mean', scale_by_freq = scale_by_freq, cmap = cmap)
        plt.close()
        dspec_n,_,_,_ = plt.specgram(traces[2].data.copy(), Fs = Fs, pad_to = 2**9, NFFT = nfft, detrend = 'mean', scale_by_freq = scale_by_freq, cmap = cmap)
        plt.close()
        dspec_all     = np.sum([dspec, dspec_e, dspec_n], axis=0)
    else:
        dspec_all     = dspec.copy()

    # --- Set time step ---
    dt = dtime[1] - dtime[0]

    # --- Check if ini.freqbands exists
    if hasattr(ini, 'freqbands') and station in ini.freqbands and flag_new_freqbands == False:
        fbands                              = ini.freqbands[station]
        ini.use_empiric_freqbands           = 1
    elif flag_new_freqbands == True:
        # If ini.freqbands does not exist, create it
        freq_band_info_pkl_file_fullpath    = os.path.join(ini.rootdata,freq_band_pkl)
        fbands                              = read_frequency_bands(freq_band_info_pkl_file_fullpath, station)
        ini.freqbands[station]              = fbands
        ini.use_empiric_freqbands           = 0
    else:
        freq_band_info_pkl_file_fullpath    = os.path.join(ini.rootdata,freq_band_pkl)
        fbands                              = read_frequency_bands(freq_band_info_pkl_file_fullpath, station)
        ini.use_empiric_freqbands           = 0

    # --- Optional: Smooth the spectrogram for better band identification
    if filter == 'gaussian' or filter == 'default':
        dspec_all = gaussian_filter(dspec_all, sigma=1)
    elif filter == 'gaussian1d':
        dspec_all = gaussian_filter1d(dspec_all, sigma=1, axis=1)
    elif filter == None or filter == 'none':
        pass
    else:
        raise ValueError("Invalid filter type. Use 'gaussian', 'gaussian1d', or None|'none'.")
    
    # --- Sum up KaVa Idx
    kava = kava_sumup_and_ratio_1(fbands, dspec_all, dfreq, dtime, station, waterlvl, empiric_values=ini.use_empiric_freqbands)

    return kava, dtime, dt, dspec_all, dfreq, dcax
    


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
    """
    Compute the RSAM for a given time period and station.

    Parameters
    ----------
    datetime_start : str
        The start time of the period to compute RSAM for.
    datetime_end : str
        The end time of the period to compute RSAM for.
    stationdir : str, optional
        The directory of the station. The default is 'KAV11'.
    stationid : str, optional
        The ID of the station. The default is 'c0941'.
    
    Returns
    -------
    rsam_df : DataFrame
        A DataFrame containing the RSAM values and their corresponding timestamps.
    """
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
    # preallocate arrays for rsam and time, so that later only the values need to be appended. For rsam it should be an array of numpy floats, for time we append pandas date_range objects
    rsam_list, time_list = [], []

    for i, idate in enumerate(tqdm(dayslist)):
        print('Processing day: ', idate[0])
        st = get_data3(idate[0], rootdata=rootdata, stationdir=stationdir, stationid=stationid)
        rsam_partial, time_partial = comp_rsam_v2(st[0].copy())
        rsam_list.append(rsam_partial)
        time_list.append(time_partial)
 
    # concatenate the lists to arrays
    rsam_array = np.concatenate(rsam_list)
    time_array = pd.concat(time_list)

    # create a DataFrame with the rsam and time arrays and save it as a pickle file
    rsam_df = pd.DataFrame({'rsam':rsam_array, 'time':time_array})
    rsam_df.to_pickle(rootdata+'/results/rsam_'+stationdir+'.pkl')
    
    return rsam_df


