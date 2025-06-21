    
import kav_init as ini
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal import PPSD
import os as os
import pandas as pd
import kavutil_2 as kt
from    obspy.core          import  read, Trace, UTCDateTime, Stream

from    obspy.signal        import  PPSD
from    scipy.signal        import  spectrogram
from    scipy.ndimage       import  gaussian_filter1d

from    scipy.stats         import  zscore
from    scipy.ndimage       import  gaussian_filter1d, gaussian_filter
from    scipy.signal        import  butter, filtfilt
from    array_ana_eval_package import read_frequency_bands
from obspy.signal.trigger import classic_sta_lta
from scipy.signal import filtfilt, butter

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
    
    return kava, dtime, dt, dspec, dfreq, dcax
   

   #TODO: Write an improved version of how to compute kava from the spectrogram. The most 
    #     sophisticated way would be to retrieve the prequency bands using the function from array_ana_eval_package.
    #     Therefore, there would be no need for empirical values for the frequency bands. Instead I have to define
    #     some threshold on how to put together the denominator. 
    #     I will also consider edge cases and ensure the function is robust.
    #     Optionally, I will also consider the case where the signal is not stationary and the frequency bands are not constant.


def compute_kava_3(x, 
                station, 
                fmin=None, 
                fmax=None, 
                appl_waterlvl=False, 
                flag_new_freqbands=False, 
                scale_by_freq=True,
                filter='gaussian', 
                freq_band_pkl='multimodal_freqanalysis_output.pkl'):
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
    if Fs is None:
        Fs          = x.stats.sampling_rate
    if nfft is None:
        nfft        = int(Fs)
    if noverlap is None:
        noverlap    = int(Fs/2)
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
    if hasattr(ini, 'freqbands') and station in ini.freqbands:
        fbands                  = ini.freqbands[station]
        empiric_values          = 1
    elif flag_new_freqbands == True:
        # If ini.freqbands does not exist, create it
        freq_band_info_pkl_file_fullpath = os.path.join(ini.rootdata,freq_band_pkl)
        fbands                           = kt.read_frequency_bands(freq_band_info_pkl_file_fullpath, station)
        ini.freqbands[station]           = fbands
        empiric_values                   = 0
    else:
        freq_band_info_pkl_file_fullpath = os.path.join(ini.rootdata,freq_band_pkl)
        fbands                           = kt.read_frequency_bands(freq_band_info_pkl_file_fullpath, station)

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
    kava = kt.kava_sumup_and_ratio(fbands, dspec_all, dfreq, dtime, empiric_values=ini.use_empiric_freqbands)

    return kava, dtime, dt, dspec_all, dfreq, dcax





def compute_kava_2(x, 
                   station, 
                   fmin=4.5, 
                   fmax=60, 
                   appl_waterlvl=False, 
                   flag_new_freqbands=False, 
                   scale_by_freq=True, 
                   cmapame='jet', 
                   z_only=False, 
                   filter='gaussian', 
                   freq_band_pkl='multimodal_freqanalysis_output.pkl'):
    ''' 
    x:               input signal of obspy trace object
    fmin:            minimum frequency of interest
    fmax:            maximum frequency of interest
    appl_waterlvl:   apply water level to kava computation
                     False | 'constant' | True
    '''

    import  numpy               as      np
    import  matplotlib.pyplot   as      plt
    import  os                  as      os
    import  pandas              as      pd
    import  kav_init            as      ini
    from    obspy.core          import  read, Trace, UTCDateTime, Stream

    from    obspy.signal        import  PPSD
    from    kav_init            import  waterlvl
    from    scipy.signal        import  spectrogram
    from    scipy.ndimage       import  gaussian_filter1d

    from    scipy.stats         import  zscore
    from    scipy.ndimage       import  gaussian_filter1d, gaussian_filter
    from    scipy.signal        import  butter, filtfilt
    from    array_ana_eval_package import read_frequency_bands

    # --- Load data ---
    f_z = os.path.join(ini.rootdata, 'KAV11', 'c0941'+ '230405000000.pri0')
    st  = read(f_z)
    x   = st[0].copy()
    d   = x.copy()
    d.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)

    # --- load horizontal components ---
    if z_only is not True:
        f_e, f_n    = os.path.join(ini.rootdata, 'KAV11', 'c0941'+ '230405000000.pri1'), os.path.join(ini.rootdata, 'KAV11', 'c0941'+ '230405000000.pri2')
        st_e, st_n  = read(f_e), read(f_n)
        x_e, x_n    = st_e[0].copy(), st_n[0].copy()
        d_e, d_n    = x_e.copy(), x_n.copy()
        d_e.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
        d_n.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)

    # --- Compute spectrogram ---
    if Fs is None:
        Fs          = x.stats.sampling_rate
    if nfft is None:
        nfft        = int(Fs)
    if noverlap is None:
        noverlap    = int(Fs/2)
    if station == 'KAV00':
        fmin        = 0.1
    if appl_waterlvl is True:
        waterlvl        = ini.waterlvl[station]
    elif appl_waterlvl == 'constant':
            waterlvl    = 1e-10
    elif appl_waterlvl == False:
            waterlvl    = 0.
    else:
        raise ValueError("Invalid value for waterlvl. Use 'constant', True, or False.")


    # --- Compute spectrogram for vertical component---
    cmap = plt.get_cmap(ini.cmapnamespec)
    dspec, dfreq, dtime, dcax = plt.specgram(d.data,
                 Fs             = Fs,
                 pad_to         = 2**9, 
                 NFFT           = nfft,
                 detrend        = 'mean',
                 scale_by_freq  = scale_by_freq,
                 cmap           = cmap, #  scale = 'linear',Fc = 0,mode='default', sides='default', window='window_hanning',
                 )
    plt.close()
    # -- ... and for horizontal components ---
    if z_only is not True:
        dspec_e,_,_,_ = plt.specgram(d_e.data,Fs=Fs,pad_to= 2**9,NFFT=nfft,detrend='mean',scale_by_freq=scale_by_freq,cmap=cmap)
        plt.close()
        dspec_n,_,_,_ = plt.specgram(d_n.data,Fs=Fs,pad_to= 2**9,NFFT=nfft,detrend='mean',scale_by_freq=scale_by_freq,cmap=cmap)
        plt.close()
        dspec_all     = np.sum([dspec, dspec_e, dspec_n], axis=0)

    # --- Set time step ---
    dt = dtime[1] - dtime[0]

    # --- Retrieve frequency bands dynamically
    
   
    # --- Check if ini.freqbands exists
    if hasattr(ini, 'freqbands') and station in ini.freqbands:
        fbands                  = ini.freqbands[station]
        empiric_values          = 1
    elif flag_new_freqbands == True:
        # If ini.freqbands does not exist, create it
        freq_band_info_pkl_file_fullpath = os.path.join(ini.rootdata,freq_band_pkl)
        fbands                           = read_frequency_bands(freq_band_info_pkl_file_fullpath, station)
        ini.freqbands[station]           = fbands
        empiric_values                   = 0
    else:
        freq_band_info_pkl_file_fullpath = os.path.join(ini.rootdata,freq_band_pkl)
        fbands                           = read_frequency_bands(freq_band_info_pkl_file_fullpath, station)

    # --- Optional: Smooth the spectrogram for better band identification
    if filter == 'gaussian' or filter == 'default':
        dspec = gaussian_filter(dspec, sigma=1)
    elif filter == 'gaussian1d':
        dspec = gaussian_filter1d(dspec, sigma=1, axis=1)
    elif filter == None or filter == 'none':
        pass
    else:
        raise ValueError("Invalid filter type. Use 'gaussian', 'gaussian1d', or None|'none'.")
    
    # --- plot one hour (10-11h) of the spectrogram for testing
    freqs2plotidx = np.where((dfreq <= fmax))[0]
    plt.pcolor(dtime[3600:7200], dfreq[freqs2plotidx], np.log10(dspec[freqs2plotidx, 3600:7200]), shading='auto', cmap='viridis')
    plt.show()




    def kava_sumup_and_ratio(fbands, dspec, dfreq, dtime, empiric_values: float = 0):
        """
        Compute the sum of the spectrogram values over the frequency bands and return the ratio.
        """
        # Initialize numerator and denominator
        numerator , denominator = np.zeros(len(dtime)), np.zeros(len(dtime))

        # Compute the sum for each frequency band
        for band in fbands:
            fmin_band, fmax_band = band
            if fmin_band < fmin:
                fmin_band = fmin
            if fmax_band > fmax:
                fmax_band = fmax
                print('fmax_band > fmax\n fmax_band set to fmax')

            # Find the indices corresponding to the frequency range
            freq_indices = np.where((dfreq >= fmin_band) & (dfreq <= (fmax_band+empiric_values)))[0]

            # Sum the spectrogram values over the frequency range
            numerator += np.sum(dspec[freq_indices], axis=0)

        # Compute the denominator
        denominator = np.sum(dspec[np.where((dfreq >= fmin) & (dfreq <= fmax))[0]], axis=0) - numerator + waterlvl
        
        # Compute kava
        kava        = numerator / denominator

        # Handle edge cases
        kava[np.isnan(kava)] = 0  # Replace NaNs with 0
        kava[np.isinf(kava)] = 0  # Replace infinities with 0

        return kava

    # --- Compute kava using the function defined above ---
    # kava        = kava_sumup_and_ratio(fbands, dspec_all, dfreq, dtime, empiric_values)

    kava            = kava_sumup_and_ratio(ini.freqbands[station], dspec_all, dfreq, dtime, empiric_values=1)
    kava_gauss      = kava_sumup_and_ratio(ini.freqbands[station], gaussian_filter(dspec_all, sigma=1), dfreq, dtime, empiric_values=1)

    kava_ana        = kava_sumup_and_ratio(fbands, dspec_all, dfreq, dtime, empiric_values=0)
    kava_ana_gauss  = kava_sumup_and_ratio(fbands, gaussian_filter(dspec_all, sigma=1), dfreq, dtime, empiric_values=0)



    # --- compute alternative kava indices ---
    dspec_z = dspec.copy()
    kava_z          = kava_sumup_and_ratio(ini.freqbands[station], dspec_z, dfreq, dtime, empiric_values=1)
    kava_z_gauss    = kava_sumup_and_ratio(ini.freqbands[station], gaussian_filter(dspec_z, sigma=1), dfreq, dtime, empiric_values=1)
    kava_z_ana      = kava_sumup_and_ratio(fbands, dspec_z, dfreq, dtime, empiric_values=0)
    kava_z_ana_gauss= kava_sumup_and_ratio(fbands, gaussian_filter(dspec_z, sigma=1), dfreq, dtime, empiric_values=0)

    store_fband_2ndin2nd = fbands[1][1]
    fbands[1][1] = 32.0
    kava_ana_2      = kava_sumup_and_ratio(fbands, dspec_all, dfreq, dtime, empiric_values=0)
    kava_ana_2_gauss= kava_sumup_and_ratio(fbands, gaussian_filter(dspec_all, sigma=1), dfreq, dtime, empiric_values=0)
    kava_ana_2_z      = kava_sumup_and_ratio(fbands, dspec_z, dfreq, dtime, empiric_values=0)
    kava_ana_2_z_gauss= kava_sumup_and_ratio(fbands, gaussian_filter(dspec_z, sigma=1), dfreq, dtime, empiric_values=0)
    fbands[1][1] = store_fband_2ndin2nd

    # --- Test plotting ---
    plt.plot(kava_z, 'r--',alpha=.5)
    plt.plot(kava, 'k',alpha=.5)
    plt.title('Kava Index')
    plt.show()

    # --- testplot kava and spectrogram ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    axright0, axright1 = ax[0].twinx(), ax[1].twinx()
    axright0.plot(dtime[:5000], kava[:5000], 'k', alpha=.5)
    # axright0.plot(dtime[:5000], kava_gauss[:5000], 'r--', alpha=.5)
    cax0 = ax[0].pcolor(dtime[:5000], dfreq, np.log10(dspec[:, :5000]), shading='auto', cmap=cmap) 
    # axright1.plot(dtime[:5000], kava_z[:5000], 'r--', alpha=.5)
    ax[1].pcolor(dtime[:5000], dfreq, np.log10(dspec_z[:, :5000]), shading='auto', cmap=cmapame)
    ax[0].set_title('Kava Index')
    ax[0].set_ylabel('Frequency [Hz]')
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [s]')
    plt.show()



    # --- Load data and compute kava for station KAV04 as well ---
    f_z_04, f_e_04, f_n_04 = os.path.join(ini.rootdata, 'KAV04', 'c0939'+ '230405000000.pri0'), os.path.join(ini.rootdata, 'KAV04', 'c0939'+ '230405000000.pri1'), os.path.join(ini.rootdata, 'KAV04', 'c0939'+ '230405000000.pri2')
    st_04, st_e_04, st_n_04 = read(f_z_04), read(f_e_04), read(f_n_04)
    x_04, x_e_04, x_n_04   = st_04[0].copy(), st_e_04[0].copy(), st_n_04[0].copy()
    d_z_04, d_e_04, d_n_04   = x_04.copy(), x_e_04.copy(), x_n_04.copy()
    d_z_04.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    d_e_04.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    d_n_04.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2, zerophase=True)
    dspec_04, dfreq_04, dtime_04, dcax_04 = plt.specgram(d_z_04.data, Fs=Fs, pad_to=2**9, NFFT=nfft, detrend='mean', scale_by_freq=scale_by_freq, cmap=cmap)
    plt.close()
    dspec_e_04, _, _, _ = plt.specgram(d_e_04.data, Fs=Fs, pad_to=2**9, NFFT=nfft, detrend='mean', scale_by_freq=scale_by_freq, cmap=cmap)
    plt.close()
    dspec_n_04, _, _, _ = plt.specgram(d_n_04.data, Fs=Fs, pad_to=2**9, NFFT=nfft, detrend='mean', scale_by_freq=scale_by_freq, cmap=cmap)
    plt.close()
    dspec_all_04 = np.sum([dspec_04, dspec_e_04, dspec_n_04], axis=0)
    dt_04 = dtime_04[1] - dtime_04[0]
    kava_04 = kava_sumup_and_ratio(ini.freqbands['KAV04'], dspec_all_04, dfreq_04, dtime_04, empiric_values=1)
    kava_04_gauss = kava_sumup_and_ratio(ini.freqbands['KAV04'], gaussian_filter(dspec_all_04, sigma=1), dfreq_04, dtime_04, empiric_values=1)
    kava_04_ana = kava_sumup_and_ratio(fbands, dspec_all_04, dfreq_04, dtime_04, empiric_values=0)
    kava_04_ana_gauss = kava_sumup_and_ratio(fbands, gaussian_filter(dspec_all_04, sigma=1), dfreq_04, dtime_04, empiric_values=0)
    kava_04_z = kava_sumup_and_ratio(ini.freqbands['KAV04'], dspec_z, dfreq_04, dtime_04, empiric_values=1)
    kava_04_z_gauss = kava_sumup_and_ratio(ini.freqbands['KAV04'], gaussian_filter(dspec_z, sigma=1), dfreq_04, dtime_04, empiric_values=1)
    kava_04_z_ana = kava_sumup_and_ratio(fbands, dspec_z, dfreq_04, dtime_04, empiric_values=0)
    kava_04_z_ana_gauss = kava_sumup_and_ratio(fbands, gaussian_filter(dspec_z, sigma=1), dfreq_04, dtime_04, empiric_values=0)
    store_fband_2ndin2nd = fbands[1][1]
    fbands[1][1] = 32.0
    kava_04_ana_2 = kava_sumup_and_ratio(fbands, dspec_all_04, dfreq_04, dtime_04, empiric_values=0)
    kava_04_ana_2_gauss = kava_sumup_and_ratio(fbands, gaussian_filter(dspec_all_04, sigma=1), dfreq_04, dtime_04, empiric_values=0)
    kava_04_ana_2_z = kava_sumup_and_ratio(fbands, dspec_z, dfreq_04, dtime_04, empiric_values=0)
    kava_04_ana_2_z_gauss = kava_sumup_and_ratio(fbands, gaussian_filter(dspec_z, sigma=1), dfreq_04, dtime_04, empiric_values=0)
    fbands[1][1] = store_fband_2ndin2nd

    shiftidx = int(ini.shifttime//dt)
    shark               = kava[:-shiftidx] * kava_04[shiftidx:]
    shark_ana           = kava_ana[:-shiftidx] * kava_04_ana[shiftidx:]
    shark_ana_2         = kava_ana_2[:-shiftidx] * kava_04_ana_2[shiftidx:]
    shark_gauss         = kava_gauss[:-shiftidx] * kava_04_gauss[shiftidx:]
    shark_ana_gauss     = kava_ana_gauss[:-shiftidx] * kava_04_ana_gauss[shiftidx:]
    shark_ana_2_gauss   = kava_ana_2_gauss[:-shiftidx] * kava_04_ana_2_gauss[shiftidx:]
    shark_z             = kava_z[:-shiftidx] * kava_04_z[shiftidx:]
    shark_z_ana         = kava_z_ana[:-shiftidx] * kava_04_z_ana[shiftidx:]
    shark_z_ana_2       = kava_ana_2_z[:-shiftidx] * kava_04_ana_2[shiftidx:]
    shark_z_gauss       = kava_z_gauss[:-shiftidx] * kava_04_z_gauss[shiftidx:]
    shark_z_ana_gauss   = kava_z_ana_gauss[:-shiftidx] * kava_04_z_ana_gauss[shiftidx:]
    shark_z_ana_2_gauss = kava_ana_2_z_gauss[:-shiftidx] * kava_04_ana_2_z_gauss[shiftidx:]

    # --- determine plot windows of variables ---
    # tmin, tmax = int(0//dt), int(60*60//dt)
    tmin, tmax = int(20*60//dt), int(30*60//dt)
    dtime = pd.to_datetime(dtime, unit='s')
    tspecplt = dtime[int(tmin):int(tmax)]
    medianfactor = 6
    f1 = dfreq[np.where(dfreq<=fmax)[0]]
    f2 = dfreq[np.where(dfreq<=fmax)[0]]
    spec1plt        = dspec[np.where(dfreq<=fmax)[0], tmin:tmax]
    spec2plt        = dspec[np.where(dfreq<=fmax)[0], tmin:tmax]
    #
    kava3plt            = kava[tmin:tmax]
    kava3pltgauss       = kava_gauss[tmin:tmax]
    kava3plt_ana        = kava_ana[tmin:tmax]
    kava3plt_ana_gauss  = kava_ana_gauss[tmin:tmax]
    kava3plt_ana_2      = kava_ana_2[tmin:tmax]
    kava3plt_ana_2_gauss= kava_ana_2_gauss[tmin:tmax]


    kavazplt            = kava_z[tmin:tmax]
    kavazpltgauss       = kava_z_gauss[tmin:tmax]
    kavazplt_ana        = kava_z_ana[tmin:tmax]
    kavazplt_ana_gauss  = kava_z_ana_gauss[tmin:tmax]
    kavazplt_ana_2      = kava_ana_2_z[tmin:tmax]
    kavazplt_ana_2_gauss= kava_ana_2_z_gauss[tmin:tmax]

    shark_plt            = shark[tmin:tmax]
    shark_ana_plt        = shark_ana[tmin:tmax]
    shark_ana_2_plt      = shark_ana_2[tmin:tmax]
    shark_gauss_plt      = shark_gauss[tmin:tmax]
    shark_ana_gauss_plt  = shark_ana_gauss[tmin:tmax]
    shark_ana_2_gauss_plt= shark_ana_2_gauss[tmin:tmax]
    shark_z_plt          = shark_z[tmin:tmax]
    shark_z_ana_plt      = shark_z_ana[tmin:tmax]
    shark_z_ana_2_plt    = shark_z_ana_2[tmin:tmax]
    shark_z_gauss_plt    = shark_z_gauss[tmin:tmax]
    shark_z_ana_gauss_plt= shark_z_ana_gauss[tmin:tmax]
    shark_z_ana_2_gauss_plt= shark_z_ana_2_gauss[tmin:tmax]



    print("\n".join([
        "Median Kava Indices:",
        f"  {np.round(np.median(kava), 2):>6} - Median Kava Index",
        f"  {np.round(np.median(kava_gauss), 2):>6} - Median Kava Index (Smoothed)",
        f"  {np.round(np.median(kava_ana), 2):>6} - Median Kava Index (Analytical)",
        f"  {np.round(np.median(kava_ana_gauss), 2):>6} - Median Kava Index (Analytical Smoothed)",
        f"  {np.round(np.median(kava_ana_2), 2):>6} - Median Kava Index (Analytical 2nd Version)",
        f"  {np.round(np.median(kava_ana_2_gauss), 2):>6} - Median Kava Index (Analytical 2nd Version Smoothed)",
        f"  {np.round(np.median(kava_z), 2):>6} - Median Kava Index (Vertical Only)",
        f"  {np.round(np.median(kava_z_gauss), 2):>6} - Median Kava Index (Vertical Only Smoothed)",
        f"  {np.round(np.median(kava_z_ana), 2):>6} - Median Kava Index (Vertical Only Analytical)",
        f"  {np.round(np.median(kava_z_ana_gauss), 2):>6} - Median Kava Index (Vertical Only Analytical Smoothed)",
        f"  {np.round(np.median(kava_ana_2_z), 2):>6} - Median Kava Index (Vertical Only Analytical 2nd Version)",
        f"  {np.round(np.median(kava_ana_2_z_gauss), 2):>6} - Median Kava Index (Vertical Only Analytical 2nd Version Smoothed)"
    ]))

    # --- plot sharks
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].pcolor(tspecplt, f1, np.log10(spec1plt), shading='nearest', cmap=cmap)
    # ax[1].plot(tspecplt, shark_plt, linestyle='--', alpha=0.7, label='Shark')
    ax[1].plot(tspecplt, shark_gauss_plt, linestyle='--', alpha=0.7, label='Shark - smoothed')
    # ax[1].hlines(medianfactor*np.median(shark),xmin=tspecplt[0],xmax=tspecplt[-1],label='median shark')
    ax[1].hlines(medianfactor*np.median(shark_gauss),xmin=tspecplt[0],xmax=tspecplt[-1],label='median shark smooth')

    # ax[1].plot(tspecplt, shark_z_plt, linestyle='--', alpha=0.7, label='Shark z')
    # ax[1].plot(tspecplt, shark_z_gauss_plt, linestyle='--', alpha=0.7, label='Shark z - smoothed')

    # ax[2].plot(tspecplt, shark_ana_plt, linestyle='--', alpha=0.7, label='Shark - f ana')
    ax[2].plot(tspecplt, shark_ana_gauss_plt, linestyle='--', alpha=0.7, label='Shark - f ana smoothed')
    ax[2].hlines(medianfactor*np.median(shark_ana),xmin=tspecplt[0],xmax=tspecplt[-1], linestyle=':',label='median shark ana')
    # ax[2].plot(tspecplt, shark_z_ana_plt, linestyle='--', alpha=0.7, label='Shark z - f ana')
    # ax[2].plot(tspecplt, shark_z_ana_gauss_plt, linestyle='--', alpha=0.7, label='Shark z - f ana smoothed')
    # ax[2].plot(tspecplt, shark_ana_2_plt, linestyle='--', alpha=0.7, label='Shark - f ana 2nd vers')
    # ax[2].plot(tspecplt, shark_ana_2_gauss_plt, linestyle='--', alpha=0.7, label='Shark - f ana 2nd vers smoothed')
    # ax[2].hlines(medianfactor*np.median(shark_ana_2),xmin=tspecplt[0],xmax=tspecplt[-1], linestyle=':',label='median shark ana 2nd vers')
    # ax[2].plot(tspecplt, shark_z_ana_2_plt, linestyle='--', alpha=0.7, label='Shark z - f ana 2nd vers')
    # ax[2].plot(tspecplt, shark_z_ana_2_gauss_plt, linestyle='--', alpha=0.7, label='Shark z - f ana 2nd vers smoothed')
    plt.setp(ax[0], title='Shark', ylabel='Frequency [Hz]', xlabel='Time [s]')
    plt.setp(ax[1], ylabel='Shark', xlabel='Time [s]')
    plt.setp(ax[2], ylabel='Shark', xlabel='Time [s]')
    [axi.legend(loc='upper right') for axi in ax]
    plt.show()


#TODO: 1) Change medianfactor. -> power 2 or times 6
#       2) Choose best shark   -> shark3D smooth or shark3D analytic smooth
#      3) TEST KDE as fitt(smoothing) and Plot !!
#       3) Implement! Especially for the shark3D - involve three components
# regarding find_peaks -> consider 30 seconds peak break -> what is common in seismology
#       4) visualise differences and why it is the best!   


#  For picking peaks/events with shark3D
# 1) Use the function find_peaks from scipy.signal
# 2) argue on what time interval and split time to use. Set something around 10 seconds.
# 3) Write down the time interval between the previous and the next peak.
# 4) Distinguish between 10-15, 15-20, 20-30, 30-60, 60-180, 180-open seconds.
# 5) Count the number of occurrences and the number of occurrences of each time interval.
# 6) Plot the interval count as histogram. maybe colorcode the interval length (amount of seconds) and plot over time. Intensity as alpha.

# Eventcount/detection
# I) Explain detection and refer to as activity or Kavachi rumble.
# II) Explain the difficulty depending on which topic to choose. Event/tumble/tremor/eruptio/burst/...
# III) Present and compare different catalogs for different band widths. and different stations.





    # --- plot spectrogram close station -------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(10, 6),sharex=True)
    axkavac = ax[0]
    axkavar = ax[1]

    #-------------------------------------------------------------------------------
    axkavac.plot(tspecplt, kavazplt_ana,         linestyle='--', alpha=0.7, label='KaVA z - f ana')
    axkavac.plot(tspecplt, kavazplt_ana_gauss,   linestyle='--', alpha=0.7, label='KaVA z - f ana smoothed')
    axkavac.plot(tspecplt, kavazplt_ana_2,       linestyle='--', alpha=0.7, label='KaVA z - f ana 2nd vers')
    axkavac.plot(tspecplt, kavazplt_ana_2_gauss, linestyle='--', alpha=0.7, label='KaVA z - f ana 2nd vers smoothed')

    # Plot median values with the same color but linestyle ':'
    axkavac.hlines(y=medianfactor*np.median(kava_z_ana),         xmin=tspecplt[0], xmax=tspecplt[-1], color='C0', linestyle=':', label='Median KaVA z - f ana')
    axkavac.hlines(y=medianfactor*np.median(kava_z_ana_gauss),   xmin=tspecplt[0], xmax=tspecplt[-1], color='C1', linestyle=':', label='Median KaVA z - f ana smoothed')
    axkavac.hlines(y=medianfactor*np.median(kava_ana_2_z),       xmin=tspecplt[0], xmax=tspecplt[-1], color='C2', linestyle=':', label='Median KaVA z - f ana 2nd vers')
    axkavac.hlines(y=medianfactor*np.median(kava_ana_2_z_gauss), xmin=tspecplt[0], xmax=tspecplt[-1], color='C3', linestyle=':', label='Median KaVA z - f ana 2nd vers smoothed')

    axkavar.plot(tspecplt, kava3plt_ana,        linestyle='--', alpha=0.7, label='KaVA - f ana')
    axkavar.plot(tspecplt, kava3plt_ana_gauss,  linestyle='--', alpha=0.7, label='KaVA - f ana smoothed')
    axkavar.plot(tspecplt, kava3plt_ana_2,      linestyle='--', alpha=0.7, label='KaVA - f ana 2nd vers')
    axkavar.plot(tspecplt, kava3plt_ana_2_gauss,linestyle='--', alpha=0.7, label='KaVA - f ana 2nd vers smoothed')

    # Plot median values with the same color but linestyle ':'
    axkavar.hlines(y=medianfactor*np.median(kava_ana),          xmin=tspecplt[0], xmax=tspecplt[-1], color='C0', linestyle=':', label='Median KaVA - f ana')
    axkavar.hlines(y=medianfactor*np.median(kava_ana_gauss),    xmin=tspecplt[0], xmax=tspecplt[-1], color='C1', linestyle=':', label='Median KaVA - f ana smoothed')
    axkavar.hlines(y=medianfactor*np.median(kava_ana_2),        xmin=tspecplt[0], xmax=tspecplt[-1], color='C2', linestyle=':', label='Median KaVA - f ana 2nd vers')
    axkavar.hlines(y=medianfactor*np.median(kava_ana_2_gauss),  xmin=tspecplt[0], xmax=tspecplt[-1], color='C3', linestyle=':', label='Median KaVA - f ana 2nd vers smoothed')


    axkavac.legend(loc='upper right')
    axkavar.legend(loc='upper right')

    plt.show()




    # ------------------------------------------------------------------------------------------

    axkavactwin  = axkavac.twinx()
    caxkavac = axkavac.pcolor(tspecplt, f2, np.log10(spec2plt), shading='nearest', cmap=cmap)
    # axkavactwin.plot(tspecplt, kavazplt, alpha=.7, linestyle='--', label='KaVA z')
    # axkavactwin.plot(tspecplt, kavazpltgauss, alpha=.7, linestyle='--',label='KaVA z - smoothed')
    axkavactwin.plot(tspecplt, kavazplt_ana,            linestyle='--', alpha=0.7, label='KaVA z - f ana')
    axkavactwin.plot(tspecplt, kavazplt_ana_gauss,      linestyle='--', alpha=0.7, label='KaVA z - f ana smoothed')
    axkavactwin.plot(tspecplt, kavazplt_ana_2,          linestyle='--', alpha=0.7, label='KaVA z - f ana 2nd vers')
    axkavactwin.plot(tspecplt, kavazplt_ana_2_gauss,    linestyle='--', alpha=0.7, label='KaVA z - f ana 2nd vers smoothed')

    # axkavactwin.hlines(y=np.median(kavazplt)*medianfactor, xmin=tspecplt[0], xmax=tspecplt[-1], color='k', linestyle=':', label='Median KaVA Index vertical only')
    # axkavactwin.hlines(y=np.median(kavazpltgauss)*medianfactor, xmin=tspecplt[0], xmax=tspecplt[-1], color='r', linestyle=':', label='Median KaVA Index vertical only smoothed')
    axkavac.set_ylabel('[Hz]',  rotation=90, labelpad=10)
    axkavac.set_ylim(0, fmax+8)
    axkavac.tick_params(axis='y', rotation=45)
    axkavactwin.set_ylabel(' KaVA Index\n Station ',  rotation=90, labelpad=10)
    axkavactwin.tick_params(axis='y', rotation=45)
    # add_colorbar_outside(caxkavac, axkavac) # test
    axkavactwin.legend(loc='upper right')

    # --- Plot spectrogram remote station ------------------------------------------------------------------------------
    axkavartwin = axkavar.twinx()
    axkavar.pcolor(tspecplt, f1, np.log10(spec1plt), shading='nearest', cmap=cmap)
    # axkavartwin.plot(tspecplt, kava3plt, alpha=.7, linestyle='--', label='KaVA')
    # axkavartwin.plot(tspecplt, kava3pltgauss, alpha=.7, linestyle='--', label='KaVA - smoothed')
    axkavartwin.plot(tspecplt, kava3plt_ana, linestyle='--', alpha=0.7, label='KaVA - f ana')
    axkavartwin.plot(tspecplt, kava3plt_ana_gauss, linestyle='--', alpha=0.7, label='KaVA - f ana smoothed')
    axkavartwin.plot(tspecplt, kava3plt_ana_2, linestyle='--', alpha=0.7, label='KaVA - f ana 2nd vers')
    axkavartwin.plot(tspecplt, kava3plt_ana_2_gauss, linestyle='--', alpha=0.7, label='KaVA - f ana 2nd vers smoothed')

    # axkavartwin.hlines(y=np.median(kava3plt)*medianfactor, xmin=tspecplt[0], xmax=tspecplt[-1], color='k', linestyle=':', label='Median KaVA Index three component')
    # axkavartwin.hlines(y=np.median(kava3pltgauss)*medianfactor, xmin=tspecplt[0], xmax=tspecplt[-1], color='r', linestyle=':', label='Median KaVA Index three component smoothed')
    axkavar.set_ylabel('[Hz]', rotation=90, labelpad=10)
    axkavar.set_ylim(0, fmax+8)
    axkavar.tick_params(axis='y', rotation=45)
    axkavartwin.set_ylabel('KaVA Index\n Station ', rotation=90, labelpad=10)
    axkavartwin.tick_params(axis='y', rotation=45)
    axkavartwin.legend(loc='upper right')

    plt.show()

    return kava, dtime, dt, dspec, dfreq, dcax



#-----------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from obspy import read
# from obspy.signal.trigger import classic_sta_lta, trigger_onset

# # Load sample data from obspy
# st = read()  # default demo file included with obspy
# tr = st[0]
# tr.detrend("demean")
# tr.filter("bandpass", freqmin=1.0, freqmax=20.0)

# # Parameters
# df = tr.stats.sampling_rate
# sta = int(1 * df)  # 1 second STA
# lta = int(10 * df)  # 10 second LTA
# cft = classic_sta_lta(tr.data, sta, lta)

# # Trigger thresholds
# on = 1.5
# off = 0.5
# on_off = trigger_onset(cft, on, off)

# # Optional enhancement: Dynamic tbreak based on rolling activity
# def dynamic_tbreak_filter(on_off, cft, sampling_rate, base_tbreak=10, min_tbreak=3, max_tbreak=20, win_sec=30):
#     """
#     Filters trigger onsets using a rolling window of STA/LTA trigger rates to adjust tbreak dynamically.
#     Suppresses triggers that are within tbreak and those at the beginning of bursty phases.
#     """
#     win_samples = int(win_sec * sampling_rate)
#     cft_activity = np.convolve((cft > on).astype(int), np.ones(win_samples), mode='same')
#     cft_activity = cft_activity / win_samples  # normalize activity

#     tbreaks = np.interp(on_off[:, 0], np.arange(len(cft_activity)), 
#                         np.interp(cft_activity, [0, 0.1], [max_tbreak, min_tbreak]))
#     tbreaks = np.clip(tbreaks, min_tbreak, max_tbreak) * sampling_rate

#     filtered = []
#     last_accepted = -np.inf
#     onsets = on_off[:, 0]

#     for i, (onset, offset) in enumerate(on_off):
#         if onset < last_accepted + tbreaks[i]:
#             continue

#         # Check for bursty region ahead
#         future = onsets[(onsets > onset) & (onsets <= onset + 5 * df)]
#         if len(future) > 2:
#             continue  # too bursty

#         filtered.append((onset, offset))
#         last_accepted = onset

#     return np.array(filtered)

# filtered_on_off = dynamic_tbreak_filter(on_off, cft, df)

# # Plotting
# times = np.arange(tr.stats.npts) / df

# fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
# axs[0].plot(times, tr.data, label="Waveform")
# axs[0].legend()
# axs[1].plot(times, cft, label="STA/LTA")
# axs[1].axhline(on, color='r', linestyle='--', label='Trigger ON')
# axs[1].axhline(off, color='b', linestyle='--', label='Trigger OFF')
# axs[1].legend()
# axs[2].plot(times, (cft > on).astype(float), alpha=0.5, label="STA/LTA > ON")

# for onset, offset in filtered_on_off:
#     axs[0].axvspan(onset / df, offset / df, color="red", alpha=0.3)

# axs[2].legend()
# axs[2].set_xlabel("Time [s]")
# plt.tight_layout()
# plt.show()
#--------------------------------------------------

# import numpy as np
# import pandas as pd

# def compute_sta_lta(signal, sta_samples, lta_samples):
#     """Compute the classic STA/LTA characteristic function."""
#     s = pd.Series(signal)
#     sta = s.rolling(sta_samples, min_periods=1).mean().to_numpy()
#     lta = s.rolling(lta_samples, min_periods=1).mean().to_numpy()
#     with np.errstate(divide='ignore', invalid='ignore'):
#         cft = sta / lta
#         cft[np.isnan(cft)] = 0
#         cft[np.isinf(cft)] = 0
#     return cft

# def detect_triggers_static_tbreak(signal, dt,
#                                   sta_win=1.0, lta_win=5.0,
#                                   trig_on=2.5, trig_off=1.5,
#                                   tbreak=20.0):
#     """
#     1) STA/LTA on/off detection
#     2) Enforce a minimum rest time (tbreak) between accepted onsets.
#     3) Drop the first of any packed burst of >=3 triggers within 3*tbreak.
#     """
#     n = len(signal)
#     sta_n = int(sta_win  / dt)
#     lta_n = int(lta_win  / dt)
#     tbreak_n = int(tbreak / dt)

#     # --- 1. Characteristic function
#     cft = compute_sta_lta(signal, sta_n, lta_n)

#     # --- 2. Basic on/off thresholding
#     triggers = []
#     armed = False
#     for i in range(n):
#         if not armed and cft[i] >= trig_on:
#             triggers.append(i)
#             armed = True
#         elif armed and cft[i] < trig_off:
#             armed = False
#     triggers = np.array(triggers, dtype=int)

#     # --- 3. Enforce static tbreak
#     accepted = []
#     last = -np.inf
#     for t in triggers:
#         if t - last > tbreak_n:
#             accepted.append(t)
#             last = t
#     accepted = np.array(accepted, dtype=int)

#     # --- 4. Drop the first of any packed burst (>=3 triggers within 3*tbreak)
#     if len(accepted) >= 3:
#         to_drop = set()
#         M = len(accepted)
#         for idx in range(M - 2):
#             t0 = accepted[idx]
#             window_end = t0 + 3 * tbreak_n
#             # count how many in accepted[idx:] fall ≤ window_end
#             # (we only need to know if >=3 exist)
#             # since accepted is sorted, just look at idx+2:
#             if accepted[idx + 2] <= window_end:
#                 to_drop.add(t0)
#         final = np.array([t for t in accepted if t not in to_drop], dtype=int)
#     else:
#         final = accepted

#     return final, cft

import numpy as np
import pandas as pd

def compute_sta_lta(signal, sta_samples, lta_samples):
    """Classic STA/LTA characteristic function (no abs() needed if signal ≥ 0)."""
    s = pd.Series(signal)
    sta = s.rolling(sta_samples, min_periods=1).mean().to_numpy()
    lta = s.rolling(lta_samples, min_periods=1).mean().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        cft = sta / lta
        cft[np.isnan(cft)] = 0
        cft[np.isinf(cft)] = 0
    return cft

def detect_sta_lta_peaks(signal, dt,
                         sta_win=2.0, lta_win=60.0,
                         trig_on=2.5, trig_off=1.5,
                         tbreak=20.0):
    """
    1) Compute STA/LTA characteristic function.
    2) Detect on/off pairs.
    3) For each pair, pick the index of max(cft) between onset and offset.
    4) Enforce min rest time (tbreak) on these peak indices.
    5) Drop the first of any packed burst of >=3 peaks within 3*tbreak.
    """
    n = len(signal)
    sta_n     = int(sta_win  / dt)
    lta_n     = int(lta_win  / dt)
    tbreak_n  = int(tbreak   / dt)

    # 1. STA/LTA
    cft = compute_sta_lta(signal, sta_n, lta_n)

    # 2. on/off detection
    on_off = []
    armed = False
    on_i = None
    for i in range(n):
        if not armed and cft[i] >= trig_on:
            armed = True
            on_i = i
        elif armed and cft[i] < trig_off:
            armed = False
            off_i = i
            on_off.append((on_i, off_i))
    # (ignore trailing armed without reset)

    # 3. find peaks in each window
    peak_idxs = []
    for on_i, off_i in on_off:
        window = cft[on_i:off_i] if off_i>on_i else cft[on_i:on_i+1]
        rel_idx = np.argmax(window)
        peak_idxs.append(on_i + int(rel_idx))
    peaks = np.array(peak_idxs, dtype=int)

    # 4. enforce static tbreak on peaks
    accepted = []
    last = -np.inf
    for p in peaks:
        if p - last > tbreak_n:
            accepted.append(p)
            last = p
    accepted = np.array(accepted, dtype=int)

    # 5. drop first of any packed burst of >=3 within 3*tbreak
    final = []
    if len(accepted) >= 3:
        to_drop = set()
        for i in range(len(accepted) - 2):
            t0 = accepted[i]
            # if the 3rd peak lies within t0 + 3*tbreak_n, mark t0
            if accepted[i+2] <= t0 + 3*tbreak_n:
                to_drop.add(t0)
        final = [p for p in accepted if p not in to_drop]
    else:
        final = accepted

    return np.array(final, dtype=int), cft

def sta_lta_simple(signal, dt,
                   sta_win=2.0, lta_win=60.0,
                   trig_on=2.5, trig_off=1.5):
    """
    Simple STA/LTA detection: for each on/off pair, pick the index of max(cft) in the window.
    Returns indices of detected peaks and the characteristic function.
    """
    n = len(signal)
    sta_n = int(sta_win / dt)
    lta_n = int(lta_win / dt)

    # Compute STA/LTA characteristic function
    cft = compute_sta_lta(signal, sta_n, lta_n)

    # Detect on/off pairs
    on_off = []
    armed = False
    on_i = None
    for i in range(n):
        if not armed and cft[i] >= trig_on:
            armed = True
            on_i = i
        elif armed and cft[i] < trig_off and armed:
            armed = False
            off_i = i
            on_off.append((on_i, off_i))

    # For each window, pick the index of max(cft)
    peak_idxs = []
    for on_i, off_i in on_off:
        window = cft[on_i:off_i] if off_i > on_i else cft[on_i:on_i+1]
        rel_idx = np.argmax(window)
        peak_idxs.append(on_i + int(rel_idx))

    return np.array(peak_idxs, dtype=int), cft

def sta_lta_recursive_padded(signal, dt,
                              sta_win=2.0, lta_win=60.0,
                              trig_on=2.5, trig_off=1.5):
    """
    Recursive (causal) STA/LTA using Obspy's classic_sta_lta with zero-padding.
    
    Ensures valid output from start and index alignment with the original signal.

    Parameters:
        signal : np.ndarray
        dt : float
        sta_win, lta_win : float
        trig_on, trig_off : float

    Returns:
        peaks : np.ndarray
        onsets : np.ndarray
        offsets : np.ndarray
        cft : np.ndarray (aligned to input signal)
    """
    n = len(signal)
    sta_n = int(sta_win / dt)
    lta_n = int(lta_win / dt)
    pad_n = lta_n  # Pad length equal to LTA window

    # Zero pad signal at start
    # padded_signal = np.pad(signal.astype(np.float32), (pad_n, 0), mode='constant')
    padded_signal = np.pad(signal.astype(np.float32), (pad_n, 0), mode='edge')

    # Compute characteristic function on padded signal
    cft_padded = classic_sta_lta(padded_signal, sta_n, lta_n)

    # Trim to original signal length
    cft = cft_padded[pad_n:]

    # Trigger detection
    armed, on_i = False, None
    on_off = []
    for i in range(n):
        if not armed and cft[i] >= trig_on:
            armed = True
            on_i = i
        elif armed and cft[i] < trig_off:
            armed = False
            on_off.append((on_i, i))

    # Find peaks in each trigger window
    # peaks = [on + np.argmax(cft[on:off]) for on, off in on_off if off > on]
    # Look a bit before the max of CFT to find the true signal rise
    search_back = int(0.5 * sta_n)
    peaks       = [max(on, on + np.argmax(cft[on:off]) - search_back) for on, off in on_off if off > on]
    onsets      = [on for on, off in on_off if off > on]
    offsets     = [off for on, off in on_off if off > on]

    return np.array(peaks, dtype=int), np.array(onsets, dtype=int), np.array(offsets, dtype=int), cft

def sta_lta_recursive_padded_delayed(signal, dt,
                              sta_win=2.0, lta_win=60.0,
                              trig_on=2.5, trig_off=1.5):
    """
    Recursive (causal) STA/LTA using Obspy's classic_sta_lta with zero-padding.
    
    Ensures valid output from start and index alignment with the original signal.

    Parameters:
        signal : np.ndarray
        dt : float
        sta_win, lta_win : float
        trig_on, trig_off : float

    Returns:
        peaks : np.ndarray
        onsets : np.ndarray
        offsets : np.ndarray
        cft : np.ndarray (aligned to input signal)
    """
    n = len(signal)
    sta_n = int(sta_win / dt)
    lta_n = int(lta_win / dt)
    pad_n = lta_n  # Pad length equal to LTA window

    # Zero pad signal at start
    # padded_signal = np.pad(signal.astype(np.float32), (pad_n, 0), mode='constant')
    padded_signal = np.pad(signal.astype(np.float32), (pad_n, 0), mode='edge')

    # Compute characteristic function on padded signal
    from obspy.signal.trigger import delayed_sta_lta
    cft_padded = delayed_sta_lta(padded_signal, sta_n, lta_n)

    # Trim to original signal length
    cft = cft_padded[pad_n:]

    # Trigger detection
    armed, on_i = False, None
    on_off = []
    for i in range(n):
        if not armed and cft[i] >= trig_on:
            armed = True
            on_i = i
        elif armed and cft[i] < trig_off:
            armed = False
            on_off.append((on_i, i))

    # Find peaks in each trigger window
    # peaks = [on + np.argmax(cft[on:off]) for on, off in on_off if off > on]
    # Look a bit before the max of CFT to find the true signal rise
    search_back = int(0.5 * sta_n)
    peaks       = [max(on, on + np.argmax(cft[on:off]) - search_back) for on, off in on_off if off > on]
    onsets      = [on for on, off in on_off if off > on]
    offsets     = [off for on, off in on_off if off > on]

    return np.array(peaks, dtype=int), np.array(onsets, dtype=int), np.array(offsets, dtype=int), cft



# # Simulate a signal: quiet + two bursts
# dt = 0.1
# t = np.arange(0, 200, dt)
# signal = 0.2 * np.random.randn(len(t))
# signal[500:600] += 3 * np.sin(2*np.pi*0.5*np.arange(100)*dt)  # burst
# signal[800:810] += 5                                           # spike

# t = ini.kavatime_slim[:3334]
# signal = ini.kavaprod2_slim[:3334]

# trig_on, trig_off = 1.5, 1.
# lta_win = 60
# # Detect with default tbreak=20 s
# triggers_3_60, cft_3_60 = detect_sta_lta_peaks(signal, dt,
#                                               sta_win=3., lta_win=lta_win,
#                                               trig_on=trig_on, trig_off=trig_off,
#                                               tbreak=20.)

# triggers_3_60_simple, cft_simple_3_60 = sta_lta_simple(signal, dt,
#                                                         sta_win=3., lta_win=lta_win,
#                                                         trig_on=trig_on, trig_off=trig_off)

# trigger_3_60_recursive, onsets_3_60, offsets_3_60, cft_rec_3_60 = sta_lta_recursive_padded(signal, dt,
#                                                                                   sta_win=3., lta_win=lta_win,
#                                                                                   trig_on=trig_on, trig_off=trig_off)

# trigger_3_60_delayed, onsets_3_60_delayed, offsets_3_60_delayed, cft_rec_3_60_delayed = sta_lta_recursive_padded_delayed(signal, dt,
#                                                                                   sta_win=3., lta_win=lta_win,
#                                                                                   trig_on=trig_on, trig_off=trig_off)

# # Plot
# peaksfound = ini.peaks_idx[ini.peaks_idx < len(signal)]
# tpeaksfound= ini.kavatime_slim[peaksfound]
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
# ax1.plot(t, signal, 'k', label='Signal')
# ax1.scatter(tpeaksfound, signal[peaksfound], c='y', s=60, label='Detected Peaks')
# ax1.scatter(t[triggers_3_60], signal[triggers_3_60], c='r', s=60, label='Detected Peaks (3s STA static trigbreak)')
# ax1.scatter(t[triggers_3_60_simple], signal[triggers_3_60_simple], c='g', s=60, label='Detected Peaks (3s STA Simple)')
# ax1.scatter(t[trigger_3_60_recursive], signal[trigger_3_60_recursive], c='b', s=60, label='Detected Peaks (3s STA Recursive)')
# ax1.scatter(t[trigger_3_60_delayed], signal[trigger_3_60_delayed], c='purple', s=60, label='Detected Peaks (3s STA Delayed)')
# ax1.legend()
# ax2.plot(t, cft_3_60, 'tab:blue', label='STA/LTA (3s STA static tbreak)')
# ax2.plot(t, cft_simple_3_60, 'tab:green', label='STA/LTA (3s STA Simple)')
# ax2.plot(t, cft_rec_3_60, 'tab:orange', label='STA/LTA (3s STA Recursive)')
# ax2.plot(t, cft_rec_3_60_delayed, 'tab:purple', label='STA/LTA (3s STA Delayed)')
# ax2.axhline(trig_on, ls='--', color='r', label='on threshold')
# ax2.axhline(trig_off, ls='--', color='b', label='off threshold')
# ax2.legend()
# ax2.set_xlabel("Time [s]")
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_hourly_variance(dataframes):
    # Standardize datetime column names and bin counts per hour
    hourly_counts = []
    for df in dataframes:
        timevar = 'Eventtime(UTC)' if 'Eventtime(UTC)' in df.columns else 'App_event_time(UTC)'
        df = df.copy()
        df['hour'] = df[timevar].dt.floor('H')
        counts = df.groupby('hour').size()
        hourly_counts.append(counts)

    # Combine into one DataFrame, align all times, and fill missing hours with 0
    all_counts = pd.concat(hourly_counts, axis=1).fillna(0)
    all_counts.columns = [f'df_{i}' for i in range(len(dataframes))]

    # Compute variance across rows (i.e., across DataFrames for each hour)
    all_counts['variance'] = all_counts.var(axis=1)

    # Plot individual time series
    all_counts.drop(columns='variance').plot(legend=False, alpha=0.6)
    plt.title('Hourly Event Counts per DataFrame')
    plt.show()

    # Plot variance
    all_counts['variance'].plot(color='red', linewidth=2, title='Variance in Hourly Activity Across DataFrames')
    plt.ylabel('Variance')
    plt.show()

    return all_counts  # for further analysis if needed

result_df = plot_hourly_variance(slimdf)
