import numpy as np
import pandas as pd
from obspy.core import read, Stream, Trace
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import os as os
import datetime
from kav_init import rootdata, rootcode, kav_angle, flag_matching, stationdir, stationid, flag_bandedratio, shifttime, freqbands, triggerresttime, sectionlen, seasons, textwidth, fontsize, pltcolors, fontsizelegend
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
from kavutil import *
from plutil import set_size
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import kav_init as ini
from tqdm import tqdm


def read_patrick(filename='Cluster_df.pkl',
                 path='D:/data_kavachi_both/'):
    if filename.endswith('.pkl'):
        type = 'pkl'
    elif filename.endswith('.txt'):
        type = 'txt'
    else:
        print('File type not supported.')
        return None
    # df = pd.read_pickle(path + filename) if type == 'pkl' else pd.read_csv(path + filename, sep='\t')
    if type == 'pkl':
        df = pd.read_pickle(os.path.join(path, filename))
    else:
        df = pd.read_csv(path + filename, sep='\t')
    print(f'-> File {filename} read. Catalog dataframe created. {len(df)} events found.')
    return df

def kava_events_patrick(df, kav_angle = [-145,-100], var = 'BAZ',time='Starttime', plot=True):
    """
    Function to extract kava events from dataframe.
    Input:
    --------
    df: dataframe with kava events
    kav_angle: list of angles to filter kava events
    var: variable to use for filtering
    time: variable to use for time axis
    plot: boolean to plot the filtered events
    Output:
    --------
    df: filtered dataframe with kava events
    """
    len_before = len(df)
    df = df[(df[var] >= kav_angle[0]) & (df[var] <= kav_angle[1])]
    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[time], df[var], 'o')
        ax.set_xlabel('Date')
        ax.set_ylabel('$\\beta$ [°] ')
        ax.set_title('Seismic events\n from ' + str(kav_angle[0]) + '° to ' + str(kav_angle[1]) + '°')
        plt.close()#show()
    print(f'-> Kavachi events filtered from catalog. Found {len(df)} out of {len_before} events.')
    return df

def random_pick(df_input, n=10, timevar='Starttime', station='KAV11', stationid='c0941'):
    """
    Function to randomly pick n events from dataframe.
    Input:
    --------
    df_input: dataframe with kava events
    n: number of events to pick
    Output:
    --------
    df: filtered dataframe with kava events
    """
    df                  = df_input.sample(n=n, random_state=1)

    # Check if corresponding data files exist
    valid_indices       = []
    invalid_indices     = []
    missing_count       = len(df)
    additional_indices  = df.index.copy()

    while missing_count != 0:
        for idx in additional_indices:
            try:
                # Construct file path
                file_path = os.path.join(
                    'D:/data_kavachi_both/',
                    station,
                    stationid + pd.to_datetime(df_input[timevar].loc[idx]).strftime('%y%m%d') + '000000.pri0'
                )
                # Check if file exists
                if os.path.exists(file_path):
                    valid_indices.append(idx)
                else:
                    invalid_indices.append(idx)
            except Exception as e:
                print(f"Error processing index {idx}: {e}")

        # If some files are missing, replace them with other randomly picked files
        missing_count = len(df) - len(valid_indices)
        if missing_count != 0:
            if missing_count == 1:
                print(f"-> Missing {missing_count} file. Replacing with random pick.")
            else:
                print(f"-> Missing {missing_count} files. Replacing with random picks.")
            try:
                additional_indices = df_input.drop(index=valid_indices + invalid_indices).sample(n=missing_count, random_state=1).index
            except ValueError as e:
                print(f"Error during replacement sampling: {e}")
                break
        else:
            print(f"-> All files found for {len(valid_indices)} events. {missing_count} files missing.")
    
    # Ensure unique indices
    valid_indices = list(set(valid_indices))

    # Filter the dataframe to include only valid or replaced indices
    df = df_input.loc[valid_indices]
    return df

def stacked_spectrum(df, timevar='Starttime', station='KAV11', stationid='c0941', prephase=10, postphase=30, pathdata='D:/data_kavachi_both/', pathout='D:/data_kavachi_both/stacked_spectrum/'):
    """
    Function to compute stacked spectrum of kava events.
    Input:
    --------
    df: dataframe with kava events
    timevar: variable to use for time axis
    station: station to use for data
    prephase: pre-phase time in seconds
    postphase: post-phase time in seconds
    pathdata: path to data
    pathout: path to output
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from obspy.core import read, UTCDateTime
    from obspy.signal.invsim import cosine_taper

    fsizelegend = 8
    pltcolor    = 'black'
    linewidth   = 1.
    plotfreqmax = 65

    if not os.path.exists(pathout):
        os.makedirs(pathout)
        print(f"-> Directory {pathout} created.")

    # Read in data
    # datastream = retrieve_trace_data(selected_events=df, stationdir=stationdir, stationid=stationid)
    stream = Stream()
    for i in range(len(df)):
        # read vertical component
        st = read(os.path.join(pathdata,station, stationid + pd.to_datetime(df[timevar].iloc[i]).strftime('%y%m%d')  + '000000.pri0'))
        # read horizontal components
        # st += read(os.path.join(pathdata, station, df[timevar][i].strftime('%y%m%d') + stationid + '000000.pri1'))
        # st += read(os.path.join(pathdata, station, df[timevar][i].strftime('%y%m%d') + stationid + '000000.pri2'))
        st = st.slice(starttime=UTCDateTime(df[timevar].iloc[i]) - prephase, endtime=UTCDateTime(df[timevar].iloc[i]) + postphase)        
        stream += st
    print(f"-> Data read for {len(df)} events.")


    for i in range(len(df)):
        tr = stream[i].copy()
        # --- Apply bandpass filter
        # tr.filter('bandpass', freqmin=4.5, freqmax=60, corners=4, zerophase=True)
        tr.detrend(type='demean')

        # --- Compute spectrum
        # f, Pxx = signal.welch(st[0].data, fs=st[0].stats.sampling_rate, nperseg=1024)
        # --- Compute spectrum using FFT
        n               = len(tr.data)
        data_tapered    = tr.taper(max_percentage=0.1, type='hann')
        data_tapered    = data_tapered.detrend('demean')
        fft_result      = np.fft.rfft(data_tapered)
        f               = np.fft.rfftfreq(n, d=1.0 / st[0].stats.sampling_rate)
        Pxx             = np.abs(fft_result) ** 2

        # --- Store spectrum in a DataFrame
        if i == 0:
            df_spectra = pd.DataFrame({'Frequency': f})
        df_spectra = pd.concat([df_spectra, pd.DataFrame({f'Spectrum_{pd.to_datetime(df[timevar].iloc[i]).strftime('%d.%m.%Y - %H:%M:%S')}': fft_result})], axis=1)#, ignore_index=True)
        # df_spectra[f'Spectrum_{i}'] = fft_result #10 * np.log10(Pxx)

        # --- Plot spectrum
        fig, ax = plt.subplots(figsize=set_size(textwidth))
        ax.plot(f, np.abs(fft_result), label='Spectrum')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        ax.set_title('Stacked Spectrum of Kava Events\n ' + str(i+1)+' of ' + str(len(df)) + ' - ' + pd.to_datetime(df[timevar].iloc[i]).strftime('%d.%m.%Y - %H:%M:%S'))
        ax.legend()
        plt.savefig(os.path.join(pathout, f'spectrum_{station}_{i}.png'))
        plt.close()
        # plt.show()
        # breakpoint()

    # --- Stack (summing up) and normalize spectra
    df_stacked                          = pd.DataFrame({'Frequency': f})
    # df_stacked['Stacked_Spectrum']      = df_spectra.iloc[:, 1:].mean(axis=1)
    df_stacked['Stacked_Spectrum']      = df_spectra.iloc[:, 1:].sum(axis=1)

    # --- Normalize the stacked spectrum by dividing by its maximum value
    df_stacked['Normalized_Spectrum']   = np.abs(df_stacked['Stacked_Spectrum']) / np.abs(df_stacked['Stacked_Spectrum']).max()

    print(f'-> Stacked spectrum computed and normalized for random sample of {len(df)} events.')

    fig, axes = plt.subplots(2,1,figsize=set_size(textwidth, 1.2), sharex=True)
    # fig2, axes = plt.subplots(2,1,figsize=(10, 6), sharex=True)
    # --- Top subplot: Plot all single spectra
    axes_top = axes[0]
    for i, col in enumerate(df_spectra.columns[1:]):
        if i == 0:
            axes_top.plot(df_spectra['Frequency'], np.abs(df_spectra[col]), label=f'{station}\nSingle spectra', alpha=.5, linewidth=.5)
        else:
            axes_top.plot(df_spectra['Frequency'], np.abs(df_spectra[col]), alpha=0.5, linewidth=linewidth) # label=pd.to_datetime(df[timevar].iloc[i]).strftime('%d.%m.%Y - %H:%M:%S'), )
    axes_top.set_ylabel('Amplitude')
    axes_top.grid(True)
    axes_top.tick_params(axis='x', labelbottom=False)  # Hide x-ticks on top subplot
    axes_top.legend(fontsize=fontsizelegend, loc='upper right')

    # --- Bottom subplot: Plot stacked and normalized spectrum
    axes_bottom = axes[1]
    axes_bottom.plot(df_stacked['Frequency'], np.abs(df_stacked['Normalized_Spectrum']), label='Stacked and normalized',linewidth=.5, color='tab:blue')
    axes_bottom.set_xlabel('Frequency [Hz]')
    axes_bottom.set_ylabel('Amplitude')
    axes_bottom.legend(fontsize=fontsizelegend, loc='upper right')
    axes_bottom.grid(True)
    axes_bottom.set_xlim(0, plotfreqmax)  # Set x-axis limits to 0 to 65 Hz

    # Align y labels and adjust layout
    fig.align_ylabels(axes)
    plt.tight_layout()
    plt.savefig(os.path.join(pathout, f'stacked_spectrum_{station}.png'))
    if ini.showplots == True:
        plt.show()
    else:
        plt.close()

    return stream, df_stacked, df_spectra


def multimodal_analysis_2(df_stacked, station=None, stationid=None, bw_factor=0.2, heightpeaks=0.3):
    """
    Function to perform multimodal analysis on kava events.
    Input:
    df: dataframe with kava events
    df_stacked: dataframe with stacked spectrum
    timevar: variable to use for time axis
    station: station to use for data
    prephase: pre-phase time in seconds
    postphase: post-phase time in seconds
    pathdata: path to data
    pathout: path to output
    """
    # Import necessary libraries
    from numpy.polynomial.polynomial import Polynomial
    from scipy.signal import savgol_filter
    from scipy.ndimage import maximum_filter1d
    from scipy.stats import gaussian_kde
    

    # # --- Simulated spectrum (replace with your real data)
    # np.random.seed(0)
    # freq = np.linspace(0, 100, 1000)
    # spectrum = (np.exp(-(freq - 30)**2 / 50) +
    #             0.7 * np.exp(-(freq - 70)**2 / 20) +
    #             0.3 * np.sin(0.3 * freq)) + 0.05 * np.random.randn(1000)
    
    # --- Load your data
    freq        = df_stacked['Frequency']
    spectrum    = df_stacked['Normalized_Spectrum']

    dfrequency  = freq[1] - freq[0]
    min_width   = int(2//dfrequency) # 2 Hz minimum width for the peaks)
    
    # --- 1. Preprocessing
    # --- 1.1.1: Moving maximum filter (for upper envelope)
    # envelope        = maximum_filter1d(np.abs(spectrum), size=20) #  40
    # spectrum_smooth = envelope
    # 1.1.2 Alternative smoothing methods
    # spectrum_smooth = savgol_filter(spectrum_smooth, window_length=21, polyorder=2)
    # spectrum_smooth = rolling_stats(envelope, func=np.mean, window=21)
    
    # --- 1.2: Polynomial fitting
    # degree          = 50
    # coefs           = Polynomial.fit(freq, spectrum_smooth, deg=degree)
    # spectrum_smooth = coefs(freq)
    # print(f'Polynomial fit of degree {degree} applied to spectrum.')

    # --- 1.0 Apply KDE (Kernel Density Estimation) to smooth the spectrum (instead of polynomial fitting and envelope)
    kde_scott       = gaussian_kde(freq, weights=np.abs(spectrum), bw_method='scott')
    kde_scott.set_bandwidth(bw_method=kde_scott.factor * bw_factor)  # Adjust bandwidth for KDE; alternatively use "kde_scott.factor * 0.6"
    freq_grid       = np.linspace(freq.min(), freq.max(), len(freq))
    spec_scott      = kde_scott(freq_grid)
    spec_scott_norm = spec_scott / np.max(spec_scott)  # Normalize the KDE result
    spectrum_smooth = spec_scott_norm

    # plt.plot(freq, np.abs(spectrum_smooth), label='Polynomial fit', color='tab:green', linewidth=.5)
    # plt.plot(freq_grid, spec_silver_norm, label='KDE Silverman', color='tab:blue', linewidth=.5)
    # plt.plot(freq_grid, spec_scott_norm, label='KDE Scott', color='tab:orange', linewidth=.5)
    # plt.plot(freq, np.abs(spectrum), alpha=.7, label='Spectrum', color='k', linewidth=.5)
    # plt.legend();plt.show()

    # set_silver, set_scott, specs_silver, specs_scott = {}, {}, {}, {}
    # for i,arg in enumerate(np.arange(.4,.9,.1)):
    #     kde_silver_comp = gaussian_kde(freq, weights=np.abs(spectrum), bw_method='silverman')
    #     kde_scott_comp  = gaussian_kde(freq, weights=np.abs(spectrum), bw_method='scott')
    #     kde_silver_comp.set_bandwidth(bw_method=kde_silver_comp.factor * arg)  # Adjust bandwidth for KDE
    #     kde_scott_comp.set_bandwidth(bw_method=kde_silver_comp.factor * arg)  # Adjust bandwidth for KDE
    #     freq_grid = np.linspace(freq.min(), freq.max(), 1000)
    #     spec_silver_comp = kde_silver_comp(freq_grid)
    #     spec_scott_comp  = kde_scott_comp(freq_grid)

    #     plt.plot(freq_grid, spec_silver_comp/spec_silver_comp.max(), label=f'Silverman, bw={arg}',linestyle='-', linewidth=.5)
    #     plt.plot(freq_grid, spec_scott_comp/spec_scott_comp.max(), label=f'Scott, bw={arg}',linestyle='--', linewidth=.5)

    # plt.plot(freq, np.abs(spectrum), alpha=.7, label='Spectrum', color='k', linewidth=.5)
    # plt.legend(); plt.grid(); plt.show()

    # 2. Detect peaks in the smoothed spectrum
    # peaks, peakproperties        = find_peaks(spectrum_smooth, prominence=0.3,height=heightpeaks)#,width=min_width)
    peaks, peakproperties        = find_peaks(spectrum_smooth, prominence=0.25,height=heightpeaks)#,width=min_width)

    # peaks, peakproperties        = find_peaks(spectrum_smooth, width=min_width,height=.2,prominence=0.3)

    # 3.0 Check results
    plt.plot(freq, np.abs(spectrum),alpha=.7,label='Spectrum', color='k', linewidth=.5)
    # plt.plot(freq, spectrum_smooth, alpha=.8,label='KDE', color='tab:blue')
    plt.plot(freq_grid, spectrum_smooth, alpha=.8,label='KDE', color='tab:blue')
    plt.scatter(freq_grid[peaks], spectrum_smooth[peaks], label='peaks', marker='o',color='tab:blue')
    plt.legend()
    if ini.showplots == True:
        plt.show()
    else:
        plt.close()

    # breakpoint()
    # 3. Find boundaries for each peak
    peak_info = []
    for peak in peaks:
        # peak_freq = freq[peak]
        peak_freq = freq_grid[peak]
        peak_amp = spectrum_smooth[peak]

        # Left boundary
        left = peak
        while left > 0 and spectrum_smooth[left] > heightpeaks:
            left -= 1
        left_freq_idx = left
        # left_freq = freq[left]
        left_freq = freq_grid[left]

        # Right boundary
        right = peak
        while right < len(freq) - 1 and spectrum_smooth[right] > heightpeaks:
            right += 1
        right_freq_idx = right
        # right_freq = freq[right]
        right_freq = freq_grid[right]
        # Append peak information
        peak_info.append((peak_freq, left_freq, right_freq, left_freq_idx, right_freq_idx))

    print(f'-> Local extrema in spectrum identified. \n{len(peaks)} dominant frequency band(s) found.')

    # 4. Plot the spectrum
    fig, ax = plt.subplots(figsize=set_size(textwidth))

    ax.plot(freq, np.abs(spectrum), label=f'{station}\nSpectrum',      alpha=.8, linewidth=.5, color='tab:blue')
    # ax.plot(freq, spectrum_smooth,  label='Fit',  alpha=.8, linewidth=.8, color='tab:orange')
    ax.plot(freq_grid, spectrum_smooth,  label='KDE \'Scott\'',  alpha=.8, linewidth=.8, color='tab:orange')
    # Mark peaks and boundaries
    for peak_freq, left_freq, right_freq, _, _ in peak_info:
        # ax.axvline(peak_freq, color='red', linestyle='--', label='Peak' if peak_freq == peak_info[0][0] else "")
        # ax.scatter(freq[peaks], spectrum_smooth[peaks], facecolors='none', edgecolors='red', marker='o')#, label='Peak')
        ax.scatter(freq_grid[peaks], spectrum_smooth[peaks], facecolors='none', edgecolors='red', marker='o')#, label='Peak')
        ax.axvspan(left_freq, right_freq, color='orange', alpha=0.4)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    # ax.set_title('Multimodal Spectral Analysis')
    ax.set_xlim(0, 65)  # Set x-axis limits to 0 to 65 Hz
    ax.set_ylim(-.05, 1.05)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'stacked_spectrum',f'multimodal_analysis_KDE_bw_{bw_factor}_height_{heightpeaks}_{station}.png'), dpi=300, bbox_inches='tight')
    if ini.showplots == True:
        plt.show()
    else:
        plt.close()

    # 5. Print results
    for i, (peak_freq, left_freq, right_freq,_,_) in enumerate(peak_info):
        print(f"-> Peak {i+1}: Frequency = {peak_freq:.2f}, Left = {left_freq:.2f}, Right = {right_freq:.2f}")

    # Store peak information in a dictionary
    peak_info_dict = {
        'peak_freq': [info[0] for info in peak_info],
        'left_freq': [info[1] for info in peak_info],
        'right_freq': [info[2] for info in peak_info],
        'left_freq_idx': [info[3] for info in peak_info],
        'right_freq_idx': [info[4] for info in peak_info]
    }
    
    if station is not None and stationid is not None:
        # Save the peak information to a file
        print(f'-> Dominant multimodal frequency bands computed for seismometer {station} ({stationid})')
    else:
        print('-> Dominant multimodal frequency band analysis completed.')
    return freq, spectrum_smooth, peaks, peak_info_dict
    
def freq_band_analysis_process(catafile,
                                   path,
                                   timevar,
                                   var,
                                   kav_angle,
                                   nrand,
                                   linewidth,
                                   stationidlist,
                                   stationlist,
                                   pathout,
                                   bw_factor=0.2,
                                   heightpeaks=0.3,
                                   showplots=True):
        
        # --- 0.0 Add initial class entries
        if showplots == True:
            ini.showplots = True
        else:
            ini.showplots = False

        # --- Print information
        print(
            "Start frequency band analysis with parameters:\n"
            f"{'Parameter':<15} | {'Value'}\n"
            f"{'-'*15}-+-{'-'*30}\n"
            f"{'catafile':<15} | {catafile}\n"
            f"{'path':<15} | {path}\n"
            f"{'timevar':<15} | {timevar}\n"
            f"{'var':<15} | {var}\n"
            f"{'kav_angle':<15} | {kav_angle}\n"
            f"{'nrand':<15} | {nrand}\n"
            f"{'linewidth':<15} | {linewidth}\n"
            f"{'stationidlist':<15} | {stationidlist}\n"
            f"{'stationlist':<15} | {stationlist}\n"
            f"{'pathout':<15} | {pathout}\n"
            f"{'bw_factor':<15} | {bw_factor}\n"
            f"{'heightpeaks':<15} | {heightpeaks}\n"
            f"{'showplots':<15} | {showplots}\n"
        )
        breakpoint()
        # 2.0 Process Part I: Read in data and filter kava events
        df = read_patrick(filename=catafile, path=path)
        df_kava     = kava_events_patrick(df, kav_angle=kav_angle, var=var, time=timevar, plot=True)
    
         # 2.1 Process Part II: Analysis
        n_kava_11                                           = random_pick(df_kava, n=nrand, station=stationlist[0], stationid=stationidlist[0])
        datastream_11, df_stacked_11, df_spectra_11         = stacked_spectrum(n_kava_11, prephase=10, postphase=30)
        # breakpoint()
        freq, spec_smooth_11, peaks_11, peak_info_dict_11   = multimodal_analysis_2(df_stacked_11, station=stationlist[0],bw_factor=bw_factor, heightpeaks=heightpeaks)
        
        n_kava_04                                           = random_pick(df_kava, n=nrand, station=stationlist[1], stationid=stationidlist[1])
        datastream_04, df_stacked_04, df_spectra_04         = stacked_spectrum(n_kava_04, station=stationlist[1], stationid=stationidlist[1])
        freq, spec_smooth_04, peaks_04, peak_info_dict_04   = multimodal_analysis_2(df_stacked_04, station=stationlist[1], bw_factor=bw_factor, heightpeaks=heightpeaks)
        
        n_kava_00                                           = random_pick(df_kava, n=nrand, station=stationlist[2], stationid=stationidlist[2])
        datastream_00, df_stacked_00, df_spectra_00         = stacked_spectrum(n_kava_00, station=stationlist[2], stationid=stationidlist[2])
        freq, spec_smooth_00, peaks_00, peak_info_dict_00   = multimodal_analysis_2(df_stacked_00, station=stationlist[2], bw_factor=bw_factor, heightpeaks=heightpeaks)
        
        # 3.0 Visualizing the results
        #---------------
        # Create figure and subplots via GridSpec
        fig             = plt.figure(figsize=set_size(textwidth, 1.5))
        gs              = GridSpec(5, 1, height_ratios=[1, 0.3, 1, 0.05, 1],hspace=0)  # Custom spacings
        ax0, ax1, ax2   = fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4])
        
        xlimits, ylimits= [[0, 60], [0, 20]], [-0.1, 1.15]
        yticks          = [0.0, 1.0]
        markersize      = 50

        # --- UPPER plot (with top x-axis) ---
        ax0.plot(   freq,           np.abs(df_stacked_11['Normalized_Spectrum']), alpha=.8, linewidth=.5, color='tab:blue', label='KAV 11') #, label='Normalised spectrum'
        ax0.plot(   freq,           spec_smooth_11,                               alpha=.8, linewidth=.8, color='tab:orange', label='KDE')
        ax0.scatter(freq[peaks_11], spec_smooth_11[peaks_11],                     facecolors='none', edgecolors='red', marker='o')
        [ax0.axvspan(
            peak_info_dict_11['left_freq'][i], 
            peak_info_dict_11['right_freq'][i], 
            color='orange', 
            alpha=0.4, 
            label='Frequency\nbands' if i == 0 else None
        ) for i in range(len(peaks_11))]

        # --- MIDDLE plot (shared x, no bottom ticks) ---
        ax1.plot(   freq,           np.abs(df_stacked_04['Normalized_Spectrum']), alpha=.8, linewidth=.5, color='tab:blue', label='KAV 04')#, label='Normalised spectrum'
        ax1.plot(   freq,           spec_smooth_04,                               alpha=.8, linewidth=.8, color='tab:orange', label='KDE')
        ax1.scatter(freq[peaks_04], spec_smooth_04[peaks_04],                     facecolors='none', edgecolors='blue', marker='o')
        [ax1.axvspan(peak_info_dict_04['left_freq'][i], peak_info_dict_04['right_freq'][i], color='tab:blue', alpha=0.4, label='Frequency\nbands') for i in range(len(peaks_04))] #  label='KAV 04'

        # --- BOTTOM plot (shared x, with labels) ---
        ax2.plot(   freq,           np.abs(df_stacked_00['Normalized_Spectrum']), alpha=.8, linewidth=.5, color='tab:blue', label='KAV 00') #, label='Normalised spectrum'
        ax2.plot(   freq,           spec_smooth_00,                           alpha=.8, linewidth=.8, color='tab:orange', label='KDE')
        ax2.scatter(freq[peaks_00], spec_smooth_00[peaks_00],                 facecolors='none', alpha=1., edgecolors='black', marker='o')
        [ax2.axvspan(peak_info_dict_00['left_freq'][i], peak_info_dict_00['right_freq'][i], color='tab:green', alpha=0.4, label='Frequency\nbands') for i in range(len(peaks_00))] # label='KAV 00'

        # --- Ax and tick parameters
        # --- Remove space between ax1 and ax2 ---
        ax0.spines['top'].set_visible(False); ax1.spines['top'].set_visible(False); ax1.spines['bottom'].set_visible(False); ax2.spines['top'].set_visible(False)
        ax0.xaxis.set_label_position('bottom')
        ax0.xaxis.set_tick_params(labeltop=False, labelbottom=True)  # Show top ticks only
        ax0.set_xlim(xlimits[0]); ax1.tick_params(labelbottom=False); ax1.set_xticks([])
        ax2.set_xlabel('Frequency [Hz]',fontsize=fontsize)
        ax1.set_ylabel('Amplitude', fontsize=fontsize, labelpad=1)
        ax2.set_xticks(np.arange(xlimits[1][0], xlimits[1][1] + 1, 5))
        [axitem.set_yticks(yticks) for axitem in [ax0, ax1, ax2]]
        [axitem.set_ylim(ylimits) for axitem in [ax0, ax1, ax2]]
        [axitem.set_xlim(xlimits[1]) for axitem in [ax1, ax2]]
        [axitem.legend(loc='upper right', fontsize=fontsizelegend) for axitem in [ax0, ax1, ax2]]

        [a.spines['right'].set_visible(False) for a in [ax0, ax1, ax2]]  # Hide right spine

        plt.tight_layout()   # Apply tight layout for minimal padding
        plt.savefig(os.path.join(pathout, f'multimodal_analysis_KDE_bw_{bw_factor}_hmin_{heightpeaks}.png'), dpi=300, bbox_inches='tight')
        if ini.showplots == True:
            plt.show()
        else:
            plt.close()
        
        df_output_multimodal = pd.DataFrame({'KAV11': peak_info_dict_11,
                                            'KAV04':peak_info_dict_04, 
                                            'KAV00':peak_info_dict_00})
        df_output_multimodal.to_pickle(os.path.join(path,f'multimodal_freqanalysis_output_KDE_bw_{bw_factor}_hmin_{heightpeaks}.pkl'))

        return df_output_multimodal

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

        


if __name__ == '__main__':

    # 1. Preallocations
    # catafile    = 'Cluster_df.pkl'
    catafile    = 'catalog_pl.txt'
    path        = 'D:/data_kavachi_both/'
    timevar     = 'Starttime'
    var         = 'BAZ'
    kav_angle   = [-145,-100] # 0 kav_angle # from init
    nrand       = 20
    linewidth   = 1.
    stationidlist = ['c0941','c0939','c0bdd']
    stationlist   = ['KAV11','KAV04','KAV00']
    pathout       = 'D:/data_kavachi_both/stacked_spectrum/'


    bw_factor    = 0.2
    heightpeaks  = 0.3
    df_output_multimodal = freq_band_analysis_process(catafile,
                                                        path,
                                                        timevar,
                                                        var,
                                                        kav_angle,
                                                        nrand,
                                                        linewidth,
                                                        stationidlist,
                                                        stationlist,
                                                        pathout,
                                                        bw_factor=bw_factor,
                                                        heightpeaks=heightpeaks,
                                                        showplots=False)


    # for bw_factor in tqdm([0.2, 0.3, .4,.5, .6]):
    #     for heightpeaks in [0.2, 0.3, 0.4, 0.5]:
    #         print(f'Running analysis with bw_factor={bw_factor} and heightpeaks={heightpeaks}\n -----')
    #         df_output_multimodal = freq_band_analysis_process(catafile,
    #                                                           path,
    #                                                           timevar,
    #                                                           var,
    #                                                           kav_angle,
    #                                                           nrand,
    #                                                           linewidth,
    #                                                           stationidlist,
    #                                                           stationlist,
    #                                                           pathout,
    #                                                           bw_factor=bw_factor,
    #                                                           heightpeaks=heightpeaks,
    #                                                           showplots=False)
    
    # df_output_multimodal = freq_band_analysis_process(catafile,
    #                                                   path,
    #                                                   timevar,
    #                                                   var,
    #                                                   kav_angle,
    #                                                   nrand,
    #                                                   linewidth,
    #                                                   stationidlist,
    #                                                   stationlist,
    #                                                   pathout,
    #                                                   bw_factor=0.2,
    #                                                   heightpeaks=0.3)
    

    breakpoint()
    