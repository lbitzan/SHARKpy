import numpy as np
import pandas as pd
from obspy.core import read, Stream, Trace
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import os as os
import datetime
from kav_init import rootdata, rootcode, kav_angle, flag_matching, stationdir, stationid, flag_bandedratio, shifttime, freqbands, triggerresttime, sectionlen, seasons
from kav_init import fmin, fmax, waterlvl, flag_wlvl, ratiorange, kavaprodemp, write_flag, rootouts, outputlabel, plot_flag, bbd_ids, cataloguefile, pkl_only
# from instrument_restitution import apply_bb_tf_inv, simulate_45Hz
from matplotlib.dates import DateFormatter
from datetime  import timedelta
import time as time
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import pickle
import plutil as plutil
import kav_init as ini
import kavutil as kt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.dates as mdates, matplotlib.gridspec as gridspec

# Moving mean "movmean()"
def movmean(x, k):
    """Moving average of x with window size k."""
    import numpy as np
    x = np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode='valid')

# Moving standard deviation "movstd()"
def movstd(x, k):
    """Moving standard deviation of x with window size k."""
    import numpy as np
    x = np.asarray(x)
    return np.sqrt(movmean(x**2, k) - movmean(x, k)**2)

# compute envelopes of a signal "hl_envelopes()"
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """
    import numpy as np
    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

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
    
    LuBi, 2021
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

def compute_banded_ratio(x1, x2, freqbands, shifttime=ini.shifttime, stats=['KAV04','KAV11'], zerophase=True, corners=2, window=20):
    '''
    Compute the amplitude ratio of two stations based on their largest characteristic frequency bands.
    
    Input:\n
    x1:         obspy stream object, remote station\n
    x2:         obspy stream object, close station\n
    freqbands:  dictionary, frequency bands of interest for each station\n
    shifttime:  float, time shift between stations in seconds\n
    stats:      list of strings, two items, station names\n
    zerophase:  bool, True: zero-phase filter, False: minimum-phase filter\n
    corners:    int, number of corners for bandpass filter [0, 2, 4]\n
    window:     int, window size for rolling statistics\n

    Output:\n
    ratio:      array, amplitude ratio\n
    ratiotime:  array, time axis for amplitude ratio


    LuBi, 2024
    '''
    from kavutil import rolling_stats
    
    a = x1.copy()
    b = x2.copy()

    fmin = np.zeros(2)
    fmax = fmin.copy()
    ind  = np.zeros(2, dtype=int)


    for i in range(2):
        fdiff       = (np.diff(freqbands[stats[i]])).flatten()
        fdiffmax    = np.amax(fdiff)

        ind[i]      = np.where(fdiff == fdiffmax)[0][0]

        fmin[i]     = freqbands[stats[i]][ind[i]][0]
        if fmin[i] < 0.1:
            fmin[i] = 0.1
        fmax[i]     = freqbands[stats[i]][ind[i]][1]
    
    # if stats[0] in ['KAV00', 'KAV10']:
    #     a = apply_bb_tf_inv(a)
    #     a = simulate_45Hz(a)
    # if stats[1] in ['KAV00', 'KAV10']:
    #     b = apply_bb_tf_inv(b)
    #     b = simulate_45Hz(b)

    a.filter('bandpass', freqmin=fmin[0], freqmax=fmax[0], corners=corners, zerophase=zerophase)
    b.filter('bandpass', freqmin=fmin[1], freqmax=fmax[1], corners=corners, zerophase=zerophase)

    a.decimate(factor=2, no_filter=True)
    b.decimate(factor=2, no_filter=True)

    ratiotime   = b.times(reftime=b.stats.starttime)
    idxshift_bandedratio = shifttime / b.stats.delta

    b_amplitude = rolling_stats(np.abs(b.data), np.amax, window=int(window))
    a_amplitude = rolling_stats(np.abs(a.data), np.amax, window=int(window))

    ratio       = b_amplitude[:-int(idxshift_bandedratio)] / a_amplitude[int(idxshift_bandedratio):]
    ratiotime   = ratiotime[:-int(idxshift_bandedratio)]

    return ratio, ratiotime

def compute_simple_ratio(x1, x2, shifttime=2.25, stats=['KAV04','KAV11'], zerophase=True, corners=2, window=20):
    '''
    Compute the amplitude ratio of two stations.
    
    Input:\n
    x1:         obspy stream object, remote station\n
    x2:         obspy stream object, close station\n
    shifttime:  float, time shift between stations in seconds\n
    stats:      list of strings, two items, station names\n
    zerophase:  bool, True: zero-phase filter, False: minimum-phase filter\n
    corners:    int, number of corners for bandpass filter [0, 2, 4]\n
    window:     int, window size for rolling statistics\n

    Output:\n
    ratio:      array, amplitude ratio\n
    ratiotime:  array, time axis for amplitude ratio


    LuBi, 2024
    '''
    from kavutil import rolling_stats
    from kav_init import fmin, fmax
    
    a    = x1.copy()
    b    = x2.copy()    
    
    #TODO: define
    # if stats[0] in ['KAV00', 'KAV10']:
    #     a = apply_bb_tf_inv(a)
    #     a = simulate_45Hz(a)
    # if stats[1] in ['KAV00', 'KAV10']:
    #     b = apply_bb_tf_inv(b)
    #     b = simulate_45Hz(b)

    a.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=corners, zerophase=zerophase)
    b.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=corners, zerophase=zerophase)

    a.decimate(factor=2, no_filter=True)
    b.decimate(factor=2, no_filter=True)

    ratiotime            = b.times(reftime=b.stats.starttime)
    idxshift_bandedratio = shifttime / b.stats.delta

    b_amplitude = rolling_stats(np.abs(b.data), np.amax, window=int(window))
    a_amplitude = rolling_stats(np.abs(a.data), np.amax, window=int(window))

    ratio       = b_amplitude[:-int(idxshift_bandedratio)] / a_amplitude[int(idxshift_bandedratio):]
    ratiotime   = ratiotime[:-int(idxshift_bandedratio)]

    return ratio, ratiotime


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


def eval_eventcount(catalogdirectory:   str,
                    
                    flag_individual:    bool=True,
                    thresholdinfo = None,
                    flag_plot:          bool=False):
    '''
    Counts the number of events in a series of catalogs. Then returns an array/pickle with the data.

    Parameters
    ----------
    catalogdirectory : string
        Path to a directory containing several catalog.txt files belonging to the same experiment.
    flag_individual : bool, optional
        If True, the thresholdvalues are read from the filenames. Default is True.
    thresholdinfo : list, optional
        List containing [start, end, step] of threshold values. Default is None. Must be given together with flag_individual=True.
    flag_plot : bool, optional
        If True, a plot is generated. Default is False.

    Returns
    -------
    neventdict : DataFrame
        parameter   -> varying parameter
        counts      -> counts of catalog with parameter respectively
    LuB 07/2024
    '''
    
    path        = catalogdirectory
    neventdict  = {}
    files       = [os.path.join(path, _) for _ in os.listdir(path) if '*.txt']
    files.sort()
    nfiles      = len(files)
    nevents     = np.zeros(nfiles)

    if flag_individual:
        threshold = [float(fname[-8:-4]) for fname in files]
    elif thresholdinfo is not None and flag_individual is False:
        threshold = np.arange(thresholdinfo[0], thresholdinfo[1]+1, thresholdinfo[2])
    else:
        threshold = np.arange(nfiles)

    for i, f in enumerate(files):
        with open(f) as file:
            nevents[i] = len(file.readlines())-1

    neventdict  = {'parameter':  threshold,
                  'counts':     nevents}
    neventdf    = pd.DataFrame(neventdict)
        
    
    if flag_plot:
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.plot(neventdf['parameter'], neventdf['counts'], ':x', color='tab:blue')
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Eventcounts / measuring campaign')
        axcopy = ax
        myplot(axoi=ax)
        pathout = os.getcwd()+'/eval_eventcount'
        while os.path.exists(pathout+'.png'):
            counter = 1
            pathout = pathout + '_' + str(counter) + '.png'
            counter += 1
        pathout = pathout + '.png';    print('Saving plot to: \n', pathout)

        plt.savefig(pathout, dpi=300)
        plt.show()
        
        return axcopy

    else:
        return neventdf

def eval_eventcount_withtremorphaseremoval(catalogdirectory: str,
                                           flag_individual: bool=True,
                                           thresholdinfo=None,
                                           flag_plot: bool=False,
                                           tremor_rate: int=7,
                                           interval: int=5,
                                           episode: list=['2023-05-30 01:00:00', '2023-05-30 01:30:00'],
                                           safedf: str='None'):
    '''
    Counts the number of events in a series of catalogs. Then returns an array/pickle with the data.

    Parameters
    ----------
    catalogdirectory : string
        Path to a directory containing several catalog.txt files belonging to the same experiment.
    flag_individual : bool, optional
        If True, the thresholdvalues are read from the filenames. Default is True.
    thresholdinfo : list, optional
        List containing [start, end, step] of threshold values. Default is None. Must be given together with flag_individual=True.
    flag_plot : bool, optional
        If True, a plot is generated. Default is False.
    tremor_rate : int, optional
        Threshold rate of events per interval. Default is 7.
    interval : int, optional
        Time interval in minutes. Default is 5.
    episode : list, optional
        Time interval of interest. Default is ['2023-05-30 01:00:00', '2023-05-30 01:30:00'].
    safedf : str, optional
        Name of the DataFrame to save. Defaults to None.

    Returns
    -------
    neventdict : DataFrame
        parameter   -> varying parameter
        counts      -> counts of catalog with parameter respectively
    LuB 07/2024
    '''
    from kavutil import rmv_tremorphases
    path        = catalogdirectory
    neventdict  = {}
    files       = [os.path.join(path, _) for _ in os.listdir(path) if '*.txt']
    files.sort()
    nfiles      = len(files)
    nevents     = np.zeros(nfiles)

    if flag_individual and thresholdinfo is None:
        threshold = [float(fname[-8:-4]) for fname in files]
    elif thresholdinfo is not None and flag_individual is True:
        threshold = np.arange(thresholdinfo[0], thresholdinfo[1]+1, thresholdinfo[2])
    else:
        threshold = np.arange(nfiles)

    for i, f in enumerate(files):


        df = rmv_tremorphases(catalog=f,
                              threshold_rate=tremor_rate,
                              interval=interval,
                              episode=episode,
                              remove_tremor=True)
        nevents[i] = len(df)
        if safedf != 'None':
            df.to_csv(safedf + '.csv', index=False)
        # breakpoint()

    neventdict  = {'parameter':  threshold,
                  'counts':     nevents}    
    neventdf    = pd.DataFrame(neventdict)
    return neventdf

    

def rmv_tremorphases(catalog:        str  = 'D:/data_kavachi_both/results/catalogs_withprominence_trigbreak/catalog_7.2_withprominence.trigbreak_00.5.txt',
                     threshold_rate: int  = 7,
                     interval:       int  = 5,
                     episode:        list = ['2023-05-30 01:00:00', '2023-05-30 01:30:00'],
                     plotting:       bool = False,
                     savedf:         str  = 'None',
                     remove_tremor:  bool = False):
    '''
    Remove tremor phases from the catalog based on a fix rate of events per specific time interval.

    Parameters:
    -----------
    catalog : str
        Name of the catalog file.
    threshold_rate : int, optional
        Threshold rate of events per interval. Default is 5.
    interval : int, optional
        Time interval in minutes. Default is 5.
    episode : list, optional
        Time interval of interest. Default is ['2023-05-30 01:00:00', '2023-05-30 01:30:00'].
    plotting : bool, optional
        If True, a plot is generated. Default is False.
    savedf : str, optional
        Name of the DataFrame to save. Default is 'None'.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing the tremor phases removed.

    2024/08/08 - created  - LuB
    2024/08/12 - add df column to store activity rate - LuB
    '''
    from kavutil import read_catalog, myplot
    path            = catalog
    freqinterval    = str(interval) + 'min'
    df              = read_catalog(path)
    episode         = pd.to_datetime(episode,format='mixed')
    # df.drop(columns=['ratio', 'kavachi_index'], inplace=True)
    df.drop(df.index[df['date'] < episode[0]], inplace=True)
    df.drop(df.index[df['date'] > episode[1]], inplace=True)
    df['date'].astype('datetime64[ns]')
    
    df['activity_rate'] = 1
    dfgby = df.groupby(pd.Grouper(key="date", freq=freqinterval))
    
    counts = dfgby['activity_rate'].transform('count')
    df.drop(columns=['activity_rate'], inplace=True)
    df = df.join(counts)
    df['tremor'] = df['activity_rate'] >= threshold_rate

    if remove_tremor is True:
        df.drop(df.index[df['tremor'] == True], inplace=True)
    
    # dfnew = dfgby.size().to_frame(name='activity_rate')
    # dfnew['date'] = dfnew.index
    # dfnew.reset_index(drop=True, inplace=True)
    # dfnew['tremor'] = dfnew['activity_rate'] >= threshold_rate
    # df['tremor']    = False
    # for i in tqdm(range(len(dfnew))):
    #     df.loc[(timedelta(minutes=0) <= df['date'] - dfnew['date'].iloc[i]) & (df['date'] - dfnew['date'].iloc[i] < timedelta(minutes=interval)), 'activity_rate'] = dfnew['activity_rate'].iloc[i]
    #     if dfnew['tremor'].iloc[i] == True:
    #         df.loc[(timedelta(minutes=0) <= df['date'] - dfnew['date'].iloc[i]) & (df['date'] - dfnew['date'].iloc[i] < timedelta(minutes=interval)), 'tremor'] = True

    if plotting:
        fig, ax     = plt.subplots(2,1, figsize=(12,8), height_ratios=[3,1], sharex=True)
        ax[0].set_title('Tremor phases based on event rate')
        date_form   = DateFormatter('%Y-%m-%d\n%H:%M:%S')
        data_gap    = pd.to_datetime(np.array(['2023-05-03 00:00:00', '2023-05-23 00:00:00'], dtype='datetime64[s]'))
        ax[0].plot(df['date'], df['activity_rate'], '-', color='tab:blue', alpha=.7, label='Counts/'+freqinterval)
        ax[0].fill_between(df['date'], 0, max(df['activity_rate'])*.75, where=df['tremor'],
                           color='tab:orange', label='Tremor phase', alpha=.3, interpolate=False)
        if episode[0] < data_gap[0] and episode[1] > data_gap[1]:
            ax[0].hlines(y=0, xmin=data_gap[0], xmax=data_gap[1], color='red', label='Data gap')
            ax[1].fill_between(data_gap, 0, 1, color='red', alpha=.3)
        ax[1].scatter(df['date'][df['tremor']],          np.ones(len(df['date'][df['tremor']]))*2/3,        color='orange',   alpha=.3, label='Events in tremor phase')
        ax[1].scatter(df['date'][df['tremor'] == False], np.ones(len(df['date'][df['tremor'] == False]))/3, color='tab:blue', alpha=.3, label='Events outside tremor phase')
        ax[1].set_ylim(0,1); ax[1].set_yticks([])
        ax[0].set_ylabel('Counts per '+freqinterval);         
        [axi.xaxis.set_major_formatter(date_form) for axi in ax]
        [myplot(axoi=axi) for axi in ax]
        fig.subplots_adjust(hspace=0)
        plt.tight_layout
        plt.savefig(rootdata+'/results/tremorphases_removed.png', dpi=400)
        plt.show()
    
    if savedf != 'None':
        # print(savedf)
        df.to_pickle(savedf)
    
    return df

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
    # breakpoint()
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

def assign_values_for_printing_plotting_pandas(timespectralpandas, tamplpandas, eventmarker, kavaproduct2, amplratio, shiftspectral):
    timefdpandasshifted = timespectralpandas[:-np.int64(shiftspectral)]
    if len(timefdpandasshifted) > len(tamplpandas):
        eventtime           = timefdpandasshifted[np.where(eventmarker == 1)]
        eventkava           = kavaproduct2[np.where(eventmarker == 1)]

        amplratio4events    = np.interp(timefdpandasshifted, tamplpandas, amplratio)
        eventratio          = amplratio4events[  np.where(eventmarker == 1)]
    elif len(timefdpandasshifted) < len(tamplpandas): # standard case
        eventtime           = tamplpandas[np.where(eventmarker == 1)]
        eventratio          = amplratio[np.where(eventmarker == 1)]

        kavaproduct4events  = np.interp(tamplpandas, timefdpandasshifted, kavaproduct2)
        eventkava           = kavaproduct4events[np.where(eventmarker == 1)]

    return eventtime, eventkava, eventratio

def assign_values_for_printing_plotting(timespectral, tampl, eventmarker, kavaproduct2, amplratio, shiftspectral):
        timespecshift           = timespectral[:-np.int64(shiftspectral)]
        if len(timespecshift) > len(tampl):
            eventtime           = timespecshift[np.where(eventmarker == 1)]
            eventkava           = kavaproduct2[np.where(eventmarker == 1)]

            amplratio4events    = np.interp(timespecshift, tampl, amplratio)
            eventratio          = amplratio4events[  np.where(eventmarker == 1)]
        elif len(timespecshift) < len(tampl): # standard case
            eventtime           = tampl[np.where(eventmarker == 1)]
            eventratio          = amplratio[np.where(eventmarker == 1)]

            kavaproduct4events  = np.interp(tampl, timespecshift, kavaproduct2)
            eventkava           = kavaproduct4events[np.where(eventmarker == 1)]
    
        return eventtime, eventkava, eventratio


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

def run_all(datetime_start: str,
            datetime_end:   str,
            flag_horizontal:bool=False,
            rrange=None,
            trigbreak=None):
    if not os.getcwd() == 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts':
        os.chdir('C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts')

    from kavutil import org_days, get_data, compute_simple_ratio, compute_banded_ratio, compute_kava, rolling_stats, fusetriggers #, myplot
    
    from kav_init import rootproject, rootcode, rootdata, rootouts, cataloguefile, outputlabel, month_flag, hour_flag, plot_flag, write_flag, freqbands
    from kav_init import stationdir, stationid, shifttime, ratiorange, flag_bandedratio, flag_wlvl, yoi, moi, doi, hoi, shifttime, ratiorange, triggerresttime

    if rrange:
        ratiorange = rrange

    if trigbreak:
        triggerresttime = trigbreak

    print('rootcode    :', rootcode); print('rootproject :', rootproject); print('rootdata    :', rootdata)
    print('')
    print('--- Script setup complete. ---')

    datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)

    dayslist = get_days_list(datetime_start, datetime_end, stationdir, stationid)

    # print('Dayslist: ', dayslist)
    print('\n---> Checked for available data. Preselection of days complete.\n')
    print('\n---> Compute, save and plot KavachiIndex information daywise. \n')
    dfresult = multiprocparent(dayslist)
    
    if write_flag:
        outputcatalog = rootouts + 'tbreak' +str(triggerresttime) + cataloguefile
        if pkl_only:
            outputcatalog=None
        outputcatalogpkl= rootouts+ 'tbreak' +str(triggerresttime) + cataloguefile[:-4] + '.pkl'
        save_df_catalog(dfresult, outputcatalogpkl, outputcatalog)
    
    return
    
    
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


def multiprocchild(day):

    # Create output directory corresponding to date
    # outputlabel  = outputlabel + '_'+str(triggerresttime)
    outputdir    = rootouts+outputlabel+'/'+outputlabel+ day[0].strftime('%Y%m')+'/' + outputlabel+ day[0].strftime('%Y%m%d')+'/' # <--- set path outputdir for  larger data set on external hard drive
    # outputcatalog= rootouts+ 'tbreak' +str(triggerresttime) + cataloguefile    #TODO: set back to original  # <--- set path and name for output catalogue for larger data set on external hard drive
    ### outputcatalog= rootouts+ 'evcnt_rrtest_'+ str(ratiorange[1])[-3:]+'_'+cataloguefile       # <--- set path and name for output catalogue for larger data set on external hard drive
    # print('outputdir: ', outputdir)
    # print('outputcatalog: ', outputcatalog)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    print('\n --> Computation started for ' + day[0].strftime('%Y-%m-%d')+'.')
    data = [Stream(), Stream()]
    for i in range(len(data)):
        data[i]     = get_data3(day[0], rootdata=rootdata, stationdir=stationdir[i], stationid=stationid[i])
    datatd, datafd, dataplt  = data.copy(), data.copy(), data.copy()
    time            = data[0][0].times(reftime=data[0][0].stats.starttime)
    timepandas      = pd.date_range(start    = pd.Timestamp(data[0][0].stats.starttime.datetime),
                                    periods  = data[0][0].stats.npts,
                                    freq     = str(data[0][0].stats.delta)+'S')
    shift = shifttime/data[0][0].stats.delta
    if flag_bandedratio:
        ratio2_1, tampl = compute_banded_ratio(datatd[0][0], datatd[1][0],
                                               freqbands=freqbands,
                                               stats=stationdir)
    elif flag_bandedratio == False:
        ratio2_1, tampl = compute_simple_ratio(datatd[0][0], datatd[1][0],
                                               stats=stationdir)
    else:
        print('\n Error  -->  flag_bandedratio not set correctly in kav_init.py . \n')
        return None
    tamplpandas = pd.date_range(start=pd.Timestamp(data[0][0].stats.starttime.datetime)+timedelta(seconds=tampl[0]),
                                periods=len(tampl),
                                freq=str(tampl[1]-tampl[0])+'S')
    
    # --- Spectrograms
    ''' Use d1, d2 as time series for spectrogram computations. Traces are filtered before between freqmin and freqmax.'''
    sr              = data[0][0].stats.sampling_rate
    nfft, noverlap  = int(sr), int(sr*.75)                              # sampling rate, number of fft points, overlap
    cspec           = plt.get_cmap('jet')                               # colormap for spectrogram 'viridis' 'plasma' 'inferno' 'magma' 'cividis' 'jet' 'gnuplot2'

    kava1, d1t, dtspec, d1spec, d1f, cax =  compute_kava(datafd[0][0], noverlap=noverlap, station= stationdir[0], frequency_bands=freqbands, cmap=cspec, appl_waterlvl=flag_wlvl)
    kava2, d2t, dtspec, d2spec, d2f, cax =  compute_kava(datafd[1][0], noverlap=noverlap, station= stationdir[1], frequency_bands=freqbands, cmap=cspec, appl_waterlvl=flag_wlvl)
    kava = np.array([kava1,  kava2])
    dspec= np.array([d1spec, d2spec])
    timefdpandas = pd.date_range(start=pd.Timestamp(datafd[0][0].stats.starttime.datetime)+timedelta(seconds=d1t[0]),
                                 periods=len(d1t),
                                 freq=str(d1t[1]-d1t[0])+'S')

    # --- Define time step between sample points
    dtsp            = d1t[1]-d1t[0]
    shiftsp         = shifttime/dtsp

    # --- Compute KaVA Index product according to time-shift
    kavaprod        = kava[1][:-np.int64(shiftsp)] * kava[0][np.int64(shiftsp):]
    kavaprod2       = rolling_stats(kavaprod, np.amax, window=10)    # max envelope of kavaprod 6

    '''Implement event trigger based on amplitude ratio'''
    ratiodelta = tampl[1] - tampl[0]
    ratiotriggerpeaks,_ = signal.find_peaks(ratio2_1, height=ratiorange, distance=2/ratiodelta) # 2/(tampl[1]-tampl[0]) = 2 sec
    # ratiotriggerpeaks,_ = signal.find_peaks(ratio2_1, height=ratiorange, prominence=50) # ---> TODO: Try out

    ratiotrigger2   = np.zeros_like(ratio2_1)
    ratiotrigger2[ratiotriggerpeaks] = 1

    pulselen        = dtsp/ratiodelta
    ratiotrigger2   = rolling_stats(ratiotrigger2, np.amax, window=np.int64(pulselen * .75))

    # Save value and time of ratio trigger peaks
    ratiopeaks, tratiopeaks, tratiopeakspandas     = ratio2_1[ratiotriggerpeaks], tampl[ratiotriggerpeaks], timepandas[ratiotriggerpeaks]

    # --- Implement event trigger based on KaVA Index
    kavaprodtrig    = np.zeros_like(kavaprod)
    kavaprodtrig    = np.where(kavaprod2 > kavaprodemp, kavaprodtrig + 1, kavaprodtrig)


    ''' Assign unisized trigger traces in boolean arrays '''
    kavatrigger, ratiotrigger = unify_triggertraces(kavaprodtrig, d1t, ratiotrigger2, tampl, shiftsp)

    # --- Fuse traces of trigger elements regarding time and frequency domain and group activies (look up at runPlotCatalog.py)
    eventmarker     = fusetriggers(kavatrigger, ratiotrigger, tampl, triggerresttime)

    ''' Assign event variables for printing/plotting. '''
    eventtime, eventkava, eventratio = assign_values_for_printing_plotting(d2t, tampl, eventmarker, kavaprod2, ratio2_1, shiftsp)

    eventtimepd, eventkavapd, eventratiopd = assign_values_for_printing_plotting_pandas(timefdpandas,
                                                                                        tamplpandas,
                                                                                        eventmarker,
                                                                                        kavaprod2,
                                                                                        ratio2_1,
                                                                                        shiftsp)

    # --- Write output to file ------------------------------------------------------------------------------------------
    '''
    Append line in catalogue for each event triggered by both ratio and kava prod at the same time.
    Note datetime as utc, amplitude ratio and kava product in its respective column. 
    If both triggers stay in phase for more than one indices, just append the first appearence.
    '''
    # def write_dfdaily(outputcatalog: str = None):
    #     '''
    #     Summarises the results of the event analysis and return them as a DataFrame.
    #     If chosen, the DataFrame is also written to a .txt file or appended respectively.
        
    #     Parameters:
    #     -----------
    #     pkl_only: bool, optional
    #         If True, only write the DataFrame to a pickle file. Default is False.
    #     outputcatalog: str
    #         Path to the output catalog file.
            
    #     Returns:
    #     --------
    #     df: pd.DataFrame
    #         DataFrame containing the results of the event analysis.
        
    #     '''
    #     if outputcatalog is not None:
    #         if not os.path.exists(outputcatalog):
    #             with open(outputcatalog, 'w') as file:
    #                 file.write('Eventtime(UTC), Amplitude_ratio, Kavaproduct\n')
    #         for idate, time2cata in enumerate(eventtimepd):
    #             with open(outputcatalog, 'a') as file:
    #                 file.write(f"{time2cata}, {eventratiopd[idate]}, {eventkavapd[idate]}\n")
        
    #     df = pd.DataFrame({'Eventtime(UTC)':   eventtimepd,
    #                        'Aplitude_ratio':   eventratiopd,
    #                        'Kavaproduct':      eventkavapd})
        
    #     print('Write ' + str(len(eventtimepd)) + ' events to catalog for '+ day[0].strftime('%Y-%m-%d') +' .')
        
    #     return df

    def write_dfdaily():
        '''
        Summarises the results of the event analysis and return them as a DataFrame.
        If chosen, the DataFrame is also written to a .txt file or appended respectively.
        
        Parameters:
        -----------
        pkl_only: bool, optional
            If True, only write the DataFrame to a pickle file. Default is False.
        outputcatalog: str
            Path to the output catalog file.
            
        Returns:
        --------
        df: pd.DataFrame
            DataFrame containing the results of the event analysis.
        
        '''
        df = pd.DataFrame({'Eventtime(UTC)':   eventtimepd,
                           'Amplitude_ratio':   eventratiopd,
                           'Kavaproduct':      eventkavapd})
        print('Write ' + str(len(eventtimepd)) + ' events to catalog for '+ day[0].strftime('%Y-%m-%d') +' .')
        
        return df

    if write_flag:
        dfdaily = write_dfdaily()


    # --- Plotting -----------------------------------------------------------------------------------------------------
    tr1, tr2        = dataplt[0][0].copy(), dataplt[1][0].copy()
    # tr1.filter('bandpass', freqmin=3, freqmax=99, zerophase=True); tr2.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)
    if stationdir[0] in bbd_ids:
        tr1.filter('bandpass', freqmin=0.1, freqmax=99, zerophase=True)
    else:
        tr1.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)
    
    if stationdir[1] in bbd_ids:
        tr2.filter('bandpass', freqmin=0.1, freqmax=99, zerophase=True)
    else:
        tr2.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)


    if plot_flag:
        plotintervals = pd.date_range(start=day[0], periods=60*60*24/sectionlen, freq=str(sectionlen)+'S')
        print('\n --> Plotting started for ' + day[0].strftime('%Y-%m-%d') + ' .\n')

        for iplot in tqdm(plotintervals):

            plutil.plot_overview(
                time_waveform       = time.copy(),
                time_amplitudes     = tampl.copy(),
                time_spectra        = d1t.copy(),
                time_ratiopeaks     = tratiopeaks.copy(),
                time_eventtime      = eventtime.copy(),
                trace1              = tr1.copy(),
                trace2              = tr2.copy(),
                ratioarray          = ratio2_1.copy(),
                ratiopeaks          = ratiopeaks.copy(),
                kavatrigger         = kavatrigger.copy(),
                spectrogram1        = d1spec.copy(),
                spectrogram2        = d2spec.copy(),
                specfrequencies1    = d1f.copy(),
                specfrequencies2    = d2f.copy(),
                kavaidx1            = kava1.copy(),
                kavaidx2            = kava2.copy(),
                kavaidxproduct      = kavaprod.copy(),
                kavaidxproduct2     = kavaprod2.copy(),
                eventkava           = eventkava.copy(),
                outputdir           = outputdir,
                h                   = iplot.hour,
                idoi                = iplot.day,
                imoi                = iplot.month,
                s                   = iplot.second+iplot.minute*60,
                ratiorange          = ratiorange)
            
            print('\n --> Plotted ' + iplot.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + (iplot+timedelta(seconds=sectionlen)).strftime('%Y-%m-%d %H:%M:%S') + ' .\n')

        print('\n --> Plotting finished for ' + day[0].strftime('%Y-%m-%d') + ' .\n')
        
    

    return dfdaily


def multiprocparent(dayslist):
    from multiprocessing import Pool
    pmain = Pool(4) #(len(dayslist))
    results = pmain.map(multiprocchild, dayslist)
    pmain.close()
    pmain.join()
    print('\n---> KavachiIndex information computed, saved and plotted daywise. \n')

    dfresult = results[0]
    if len(results) > 1:
        for ires in results[1:]:
            dfresult = pd.concat([dfresult, ires], ignore_index=True)
    
    print('\n --> Export merged event catalog for chosen time frame.')
    return dfresult

def save_df_catalog(df: pd.DataFrame,
                    outputcatalogpkl: str,
                    outputcatalog: str = None):
    df['Amplitude_ratio']   = df['Amplitude_ratio'].round(decimals=2)
    df['Kavaproduct']       = df['Kavaproduct'].round(decimals=2)
    df.to_pickle(outputcatalogpkl)
    if outputcatalog is not None:
        df.to_csv(outputcatalog, sep='\t', index=False)
    return


def run4catalog(datetime_start: str | np.datetime64,
                datetime_end:   str | np.datetime64,
                rrange          = None,
                trigbreak       = None):
    
    scriptstart = time.time()
    if not os.getcwd() == 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts':
        os.chdir('C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts')
            
    from kav_init import rootproject, rootcode, rootdata, rootouts, cataloguefile, outputlabel, month_flag, hour_flag, plot_flag, write_flag, freqbands
    from kav_init import stationdir, stationid, shifttime, ratiorange,pkl_only, flag_bandedratio, flag_wlvl, yoi, moi, doi, hoi, shifttime, ratiorange, triggerresttime

    if rrange:
        ratiorange = rrange
    if trigbreak:
        triggerresttime = trigbreak

    datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)
    dayslist = get_days_list(datetime_start, datetime_end, stationdir, stationid)

    # breakpoint()
    print('\n rootcode    :', rootcode); print('\n rootproject :', rootproject); print('\n rootdata    :', rootdata)
    print('')
    print('--- Script setup complete after '+ str(np.round(time.time() - scriptstart, 1))+' sec ---\n --> Investigate time from ' + datetime_start.strftime('%Y-%m-%d') + ' to '+ datetime_end.strftime('%Y-%m-%d'))

    print('Dayslist \n -------- \n', dayslist)

    dfresult = multiprocparent(dayslist)

    outputcatalog   = rootouts + 'tbreak' +str(triggerresttime)+'.' + cataloguefile
    outputcatalogpkl= rootouts + 'tbreak' +str(triggerresttime)+'.' + cataloguefile[:-4] + '.pkl'
    if write_flag:
        if pkl_only:
            outputcatalog=None
        save_df_catalog(dfresult, outputcatalogpkl, outputcatalog)

    print('\n --> Script finished after '+ str(np.round(time.time() - scriptstart,1))+' sec. ')
    # breakpoint()

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


def kavachi_freq_analysis(catalog: pd.DataFrame,
                          datetime_start: str,
                          datetime_end: str,
                          timebin: str = '60s'):
    raise NotImplementedError
    datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)
    catalog = catalog[(catalog['date'] >= datetime_start) & (catalog['date'] <= datetime_end)]

    # Transform the catalog to a continuous evenly spaced time series.
    # Define a custom length interval in seconds for the time steps. default tstep= '60s'
    tstep = timebin
    # Restructure the catalog to that evenly time-spaced time series and count the number of entries per time step.
    catalog.set_index('date', inplace=True)
    catalog_resampled = catalog.resample(tstep).size()
    # Save the results to a DataFrame with time and counts as columns.
    df_result = pd.DataFrame({'time': catalog_resampled.index, 'counts': catalog_resampled.values})

    # perform a Fourier transform on the counts to get the frequency spectrum.
    # Make sure to use adequate zeropadding to get a good frequency resolution.
    # Define the time step of the time series.
    dt      = pd.to_timedelta(tstep).total_seconds()
    N       = len(df_result)
    fs      = 1/dt
    freqs   = np.fft.fftfreq(N, dt)
    fft     = np.fft.fft(df_result['counts'])
    freqs   = freqs[:N//2]
    fft     = fft[:N//2]
    amp     = np.abs(fft)
    power   = np.abs(fft)**2
    df_fd = pd.DataFrame({'frequency': freqs, 'amplitude': amp, 'power': power})

    # Plot the results freq analysis
    fig, ax = plt.subplots(2,1, figsize=(6,4))
    ax[0].plot(df_fd['frequency'], df_fd['amplitude'], color='tab:blue', label='Amplitude spectrum')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].legend()
    ax[1].plot(df_fd['frequency'], df_fd['power'], color='tab:blue', label='Power spectrum')
    ax[1].set_ylabel('Power')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.plot(df_fd['frequency'], df_fd['power'], color='tab:blue', label='Power spectrum')
    ax.set_ylabel('Power')
    ax.set_xscale('log')
    plt.show()


    breakpoint()
    

    # Return the DataFrame.
    return df_result, df_fd
    

def event_tdiff_distr(catalog: pd.DataFrame,
                      stationdir = 'KAV11',
                      nbins = 30,
                      binmin: int=20,
                      binmax: int=60*60*24,
                      triggerrestingtime = 20):
    if stationdir == 'KAV11':
        periods = seasons[:2][:]
    elif stationdir == 'KAV00':
        periods = seasons
    else:
        raise ValueError('stationdir must be either KAV11 or KAV00')
    

    # Create a DataFrame with all events from the original catalog lying within the periods
    cata = pd.DataFrame()
    for period in periods:
        cata_period = catalog[(catalog['date'] >= period[0]) & (catalog['date'] <= period[1])]
        cata = pd.concat([cata, cata_period], ignore_index=True)
    
    # Calculate the time difference between consecutive events.
    timediff = np.diff(cata['date'])
    timediff = timediff.astype('timedelta64[s]').astype(int)

    #  create bins array from binmin to minmax with even bin values and spacing
    # breakpoint()
    bins     = np.linspace(binmin, binmax, nbins)
    bins = [int(b) for b in bins]
    countsbin= np.histogram(timediff, bins=bins)[0]
    histo = pd.DataFrame({'bins': bins[:-1], 'counts': countsbin})

    # define bin where counts are maximum
    highest = histo['bins'][histo['counts'].idxmax()]

    # Plot number of events per time difference as bar chart
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_title('Time difference distribution of events\nStation '+stationdir+', triggerrestingtime = '+str(triggerrestingtime)+'s')
    ax.scatter(histo['bins'], histo['counts'], color='tab:blue', alpha=.7, label='Event time difference distribution\n peak around '+str(highest)+ ' sec', marker='o')
    ax.set_xlabel('Time difference [s]')
    ax.set_ylabel('Number of events')
    ax.set_yscale('log')
    ax.legend(); ax.grid(); plt.tight_layout()
    plt.savefig(rootdata+'/results/timediff_distr.' + stationdir+'.'+str(triggerrestingtime) + '.png', dpi=300)
    plt.close() # plt.show()


    # breakpoint()

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
        fbands                              = kt.read_frequency_bands(freq_band_info_pkl_file_fullpath, station)
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