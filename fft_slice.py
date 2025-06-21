
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
    