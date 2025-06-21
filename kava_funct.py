import numpy as np
import matplotlib.pyplot as plt

def kava_funct(x, station, fmin = 3, fmax = 60,
               NFFT = None, Fs = None, Fc=None, detrend=None, window=None, noverlap=None,
               cmap=None, xextent=None, pad_to=None, sides=None, scale_by_freq=True, mode=None,
               scale=None, vmin=None, vmax=None, *, data=None, **kwargs):
    '''
    This function is used to compute the KaVA index along an array of seismic data.
    So far only two stations are specified.
    Input:
    x :         array
        Timeseries of seismic data
    station:    string
        station code
    Input (optional) for specgram:
    fmin:       int
        minimum frequency for specgram
    fmax:       int
        maximum frequency for specgram
    NFFT:       int
    FS:         float
    Fc:         float
    detrend:    string
    window:     string
    noverlap:   int
    cmap:       string
    xextend:    string
    pad_to:     int
    ...
    --- further check documentation of matplotlib.pyplot.specgram ---


    Output:
    kavac - KaVA index

    Extra:
    For best results, use the following parameters:
    Fs          - sr (sampling rate)
    nfft        - sr
    noverlap    - sr*.75
    
    '''

    if station == 'KAV04':
        fminstation = [4]
        fmaxstation = [8]

        kavac      = np.sum(d04spec[f1min1:f1max1,:],axis=0) / (np.sum(d04spec[fmin:f1min1,:],axis=0) +  np.sum(d04spec[f1max1:fmax],axis=0))

    elif station == 'KAV11':
        fminstation = [4, 12]
        fmaxstation = [7, 50]

    # --- SET PARAMETERS ---------------------------------------------------------------------------------------------------
    fmin = int(fmin)                        # for general filtering used in bandpass filter
    fmax = int(fmax)                        # for general filtering used in bandpass filter

    # Compute spectrogram
    spectrogram, freq, kavatime, cax = plt.specgram(x, NFFT=NFFT, Fs=Fs, Fc=Fc, detrend=detrend, window=window, noverlap=noverlap, 
                                                    cmap=cmap, xextent=xextent, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq,
                                                    mode=mode, scale=scale, vmin=vmin, vmax=vmax, data=data, **kwargs)
    # suppress image output
    plt.close()

    freqbands = len(fminstation)

    # compute indices for frequency bands
    for i in range(freqbands):
        if i == 0:
            freqbandsidxpartial = np.where((freq >= fminstation[i]) & (freq <= fmaxstation[i]+1))
            freqbandsidx        = freqbandsidxpartial
        else:
            freqbandsidxpartial = np.where((freq >= fminstation[i]) & (freq <= fmaxstation[i]+1))
            freqbandsidx        = np.concatenate((freqbandsidx, freqbandsidxpartial), axis=1)

    denominator_freqidx = np.delete(freq, freqbandsidx)

    # Compute numerator and denominator
    numerator   = np.sum(spectrogram[freqbandsidx,:], axis=0)
    denominator = np.sum(spectrogram[denominator_freqidx, :], axis=0)

    # Compute KaVA index
    kava = numerator / denominator
    
    # Return KaVA index, corresponding time axis and spectrogram
    return kava, kavatime,spectrogram
