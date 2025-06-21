import os
# from    obspy.core           import  read, UTCDateTime, Stream, Trace
import  numpy                as      np
import  datetime
import  matplotlib.pyplot    as      plt
import  pandas               as      pd
from    scipy                import  signal
import  time                 as      tm
import  plutil               as      plutil
from    datetime             import  timedelta
from    colorbar_fct         import  add_colorbar_outside
from    tqdm import tqdm 
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
        

# -----------------------------------------------------------------------------------------------------------------------
'''
>>> runPlotting.py <<<
Script to evaluate data from Kavachi volcano, Solomon Islands.
Further parameters are computated and plotted within this script and attached routines.
This script uses routines from the KavachiProject/KavScripts directory (e.g. kavutil.py, colorbar_fct.py,...)

Author:     Ludwig Bitzan, 2023
created:    2024-01-30
'''

# RUN SCRIPT as __MAIN__ and import initials
# -> end of file
import kav_init as ini
import kavutil_2 as kt

def runPlotCatalog(datetime_start: str | np.datetime64,
                   datetime_end: str | np.datetime64,
                   rrange = None, trigbreak = None):
    '''
    This scripts evaluates data from the Kavachi Survey, from 2023 ongoing. Particular dates and stations should be specified in the initial file kav_init.py.
    Depending on the flags set in the in the inital file, the output covers an overview plot as well as an event catalog.
    Advanced parameters can be set within in the functions syntax, the initial file or the associated modules.
    '''
    ini.start_time = tm.time()
    # --- Ensure working directory is set to 'KavScripts'
    if 'KavScripts' not in os.getcwd():
        if 'KavScripts' not in os.listdir():
            raise FileNotFoundError('KavScripts directory not found. Please check the path to the KavachiProject repository.')
        else:
            os.chdir('KavScripts')
            print('Changed directory to: ', os.getcwd())
    elif 'KavScripts' in os.getcwd():
        print('Script already started from working repository: ', os.getcwd())

    # --- Optional: Replace initial class values with user input vars
    if rrange is not None:
        ini.ratiorange      = rrange
    if trigbreak is not None:
        ini.triggerresttime = trigbreak

    # --- Set time frame according to input
    datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)
    dayslist                     = kt.get_days_list(datetime_start, datetime_end, ini.stationdir, ini.stationid)

    # ini.outputcatalog            = ini.rootouts+ ini.cataloguefile+str(ini.triggerresttime)+'s.txt'
    ini.outputcatalog            = ini.rootouts+ ini.cataloguefile+'.txt'

    df_chunks_slim, df_chunks_ratio = [], []
    results         = {tb: [] for tb in ini.triggerresttime}  # Dictionary to hold results for each trigger rest time
    results_ratio   = {tb: [] for tb in ini.triggerresttime}  # Dictionary to hold results for amplitude ratio

    # --- Output process information
    print(
        f"{'-'*40}\n"
        f"{'Script setup':<20} | Complete after {np.round(tm.time() - ini.start_time, 1)} sec\n"
        f"{'Investigate time':<20} | {datetime_start.strftime('%Y-%m-%d')} to {datetime_end.strftime('%Y-%m-%d')}\n"
        f"{'-'*40}\n"
        f"{'rootcode':<20} | {ini.rootcode}\n"
        f"{'rootproject':<20} | {ini.rootproject}\n"
        f"{'rootdata':<20} | {ini.rootdata}\n"
        f"{'Triggerresttime':<20} | {ini.triggerresttime} sec\n"
        f"{'Shifttime':<20} | {ini.shifttime} sec - {8.5/ini.shifttime} v_app\n"
        f"{'Ampl. ratio range':<20} | {ini.ratiorange[0]} to {ini.ratiorange[1]}\n"
        f"{'Threshold Shark':<20} | {ini.kavaprodemp}\n"
        f"{'Stationdir':<20} | {ini.stationdir}\n"
        f"{'Stationid':<20} | {ini.stationid}\n"
        f"{'Bandpass corners':<20} | {ini.bandpasscorners}\n"
        f"{'Use three components':<20} | {ini.use_three_components}\n"
        f"{'-'*40}\n"
        f"{'Dayslist':<20} |\n"
        f"{dayslist}\n"
        f"{'-'*40}\n"
        f"{'Output catalogue':<20} | {ini.outputcatalog}\n"
    )

    
    # --- Run loop over time ---------------------------------------------------------------------------------------------------
    skipped_dates = pd.DataFrame(columns=['skipped_dates'])
    for tidx, idate in enumerate(tqdm(dayslist, desc="Processing days")):
        ini.idate = idate[0]
        
        print('\n --> Computation started for ' + idate[0].strftime('%Y-%m-%d')+' \n')
        # --- Read in data -----------------------------------------------------------------------------------------------------
        st1, st2 = [kt.get_data4(idate[0], rootdata=ini.rootdata, stationdir=ini.stationdir[i], stationid=ini.stationid[i]) for i in range(len(ini.stationdir))]
        # st1 = kt.get_data3(idate[0], rootdata=ini.rootdata, stationdir=ini.stationdir[0], stationid=ini.stationid[0])
        # st2 = kt.get_data3(idate[0], rootdata=ini.rootdata, stationdir=ini.stationdir[1], stationid=ini.stationid[1])
        # Remove days with missing data
        if len(st1[0]) != len(st2[0]):
            print('--> Length of traces is not equal: Skip date.')
            skipped_dates.loc[len(skipped_dates)] = idate[0]
            continue

        # --- Create output directory corresponding to date
        # outputlabel  = outputlabel + '_'+str(triggerresttime)
        ini.outputdir    = ini.rootouts+ini.outputlabel+'.'+str(ini.triggerresttime)+'/'+ini.outputlabel+ idate[0].strftime('%Y%m')+'/' + ini.outputlabel+ idate[0].strftime('%Y%m%d')+'/' # <--- set path outputdir for  larger data set on external hard drive
        print(f"{'Output directory':<20} | {ini.outputdir}")

        if ini.plot_flag:
            if not os.path.exists(ini.outputdir):
                os.makedirs(ini.outputdir)
        
        # --- Computations ------------------------------------------------------------------------------------------------- 
        ''' Assign traces to variables for computations in time domain. '''
        ini.tr1, ini.tr2    = st1[0].copy(), st2[0].copy()
        ''' Assign and prefilter traces for computations in frequency domain. '''
        ini.d1, ini.d2      = st1.copy(), st2.copy()
        
        # --- Create time array relative to starttime
        ini.time        = ini.tr1.times(reftime = ini.tr1.stats.starttime)
        ini.shift       = ini.shifttime / ini.tr1.stats.delta                                       # time delay between stations as no of indices
        print(f"{'Data initialised for':<20} | {ini.stationdir[0]} and {ini.stationdir[1]} on {idate[0].strftime('%Y-%m-%d')}.")

        # --- Compute amplitude ratio from (3D-) traces
        if ini.compare_different_triggers is True or ini.compare_different_triggers == 'amplitude':
            if ini.flag_bandedratio == False:
                ini.ratio2_1, ini.tampl = kt.compute_simple_ratio(ini.tr1, ini.tr2, stats= ini.stationdir)
            else:
                ini.ratio2_1, ini.tampl = kt.compute_banded_ratio(ini.tr1, ini.tr2, freqbands=ini.freqbands, stats= ini.stationdir, corners=ini.bandpasscorners)
            print(f"{'Amplitude ratio computed':<20} | downsampled from {len(ini.tr1)} to {len(ini.tampl)} samples.")


        # --- Spectrograms
        ini.kava1, ini.d1t, ini.dtspec, ini.d1spec, ini.d1f, _   = kt.compute_kava_3(ini.d1, station=ini.stationdir[0], filter=ini.filter_kava,freq_band_pkl=ini.freqbandpklfile)
        print(f"{'--> KaVA Index computed':<20} | remote array: station {ini.stationdir[0]} with {len(ini.d1t)} samples.")
        ini.kava2, ini.d2t, dtspec,     ini.d2spec, ini.d2f, _   = kt.compute_kava_3(ini.d2, station=ini.stationdir[1], filter=ini.filter_kava,freq_band_pkl=ini.freqbandpklfile)
        print(f"{'--> KaVA Index computed':<20} | near-field array: station {ini.stationdir[1]} with {len(ini.d2t)} samples.")

        # --- Define time step between sample points
        dtsp            = ini.d1t[1]-ini.d1t[0]
        ini.shiftsp     = ini.shifttime/ini.dtspec
        # breakpoint()
        
        # --- Compute KaVA Index product according to time-shift
        ini.kavaprod    = ini.kava2[:-np.int64(ini.shiftsp)] * ini.kava1[np.int64(ini.shiftsp):]
        ini.kavaprod2   = kt.rolling_stats(ini.kavaprod, np.amax, window=10)    # max envelope of kavaprod 6
        # breakpoint()


        # Plot traces and kava
        plot_traces_and_kava = False
        if plot_traces_and_kava:
            start, end = 0 * 60 + 30, 0 * 60 + 45
            wspace, hspace = 0.1, 0.15

            fig, ax = plt.subplots(2, 2, figsize=plutil.set_size(ini.textwidth,1.4))
            # fig, ax = plt.subplots(2, 2, figsize=(6.85, 3.6))
            fig.subplots_adjust(wspace=wspace, hspace=hspace)

            sequt = np.arange(start * 60 / ini.tr2.stats.delta, end * 60 / ini.tr2.stats.delta + 1, dtype=int)
            sequf = np.arange(start * 60 / ini.dtspec, end * 60 / ini.dtspec + 1, dtype=int)
            timef = pd.to_datetime([pd.Timestamp(idate[0]) + pd.Timedelta(seconds=float(s)) for s in ini.d2t[sequf]- ini.d2t[sequf[0]]])
            timet = pd.to_datetime([pd.Timestamp(idate[0]) + pd.Timedelta(seconds=float(s)) for s in ini.time[sequt]-ini.time[sequt[0]]])
            timestart = pd.to_datetime(idate[0]) + pd.Timedelta(seconds=float(ini.time[sequt[0]]))
            # Plotting
            _ = [
                ax[0, 0].plot(timet, ini.tr2.detrend('demean').data[sequt], label='waveform', color='black', alpha=0.5, linewidth=0.5),
                ax[1, 0].plot(timef, ini.kava2[sequf],                      label='KaVA Idx', color='blue',  alpha=0.5, linewidth=0.5),
                ax[0, 1].plot(timet, ini.tr1.detrend('demean').data[sequt], label='waveform', color='black', alpha=0.5, linewidth=0.5),
                ax[1, 1].plot(timef, ini.kava1[sequf],                      label='KaVA Idx', color='blue',  alpha=0.5, linewidth=0.5)
            ]

            # Style tweaks (all one-liners)
            [a.spines['top'].set_visible(False) for a in ax.flatten()]
            [a.spines['right'].set_visible(False) for a in ax.flatten()]
            [a.tick_params(labelbottom=False) for a in ax[0, :]]
            [a.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True)) for a in ax.flatten()]
            [a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) for a in ax.flatten()]
            [a.yaxis.offsetText.set_fontsize(8) for a in ax.flatten()]
            [a.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S')) for a in ax[1, :]]
            [a.tick_params(axis='x', labelrotation=0) for a in ax[1, :]]
            [a.legend(loc='upper right', fontsize=ini.fontsizelegend, frameon=False) for a in ax.flatten()]
            fig.text(0.5, 0.01, 'Time [min] \n since '+timestart.strftime('%Y-%m-%d %H:%M:%S'), ha='center', va='center', fontsize=10)
            fig.text(0.05, 0.5, 'Amplitudes', ha='center', va='center', rotation='vertical', fontsize=10)
            ax[0, 0].set_title(f"{ini.stationdir[1]}", fontsize=10)
            ax[0, 1].set_title(f"{ini.stationdir[0]}", fontsize=10)
            
            plt.savefig(os.path.join(ini.rootouts, 'step_kava_development.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.show()

        if ini.compare_different_triggers is True or ini.compare_different_triggers == 'amplitude':

            for tb in ini.triggerresttime:
                # --- Implement event trigger based on amplitude ratio        
                '''ratio trigger:'''
                ratiodelta = ini.tampl[1] - ini.tampl[0]
                ini.ratiotimedelta  = ini.tampl[1] - ini.tampl[0]
                ini.ratiotriggerpeaks,_ = signal.find_peaks(ini.ratio2_1, height=ini.ratiorange, distance=2/ini.ratiotimedelta) # 2/(tampl[1]-tampl[0]) = 2 sec
                # ratiotriggerpeaks,_ = signal.find_peaks(ratio2_1, height=ratiorange, prominence=50) # ---> with prominence

                ini.ratiotrigger2   = np.zeros_like(ini.ratio2_1)
                ini.ratiotrigger2[ini.ratiotriggerpeaks] = 1

                pulselen            = ini.dtspec/ini.ratiotimedelta
                ini.ratiotrigger2   = kt.rolling_stats(ini.ratiotrigger2, np.amax, window=np.int64(pulselen * .75))

                ini.ratiopeaks, ini.tratiopeaks     = ini.ratio2_1[ini.ratiotriggerpeaks], ini.tampl[ini.ratiotriggerpeaks]

                # --- Implement event trigger based on KaVA Index
                ini.kavaprodtrig    = np.zeros_like(ini.kavaprod)
                ini.kavaprodtrig    = np.where(ini.kavaprod2 > ini.kavaprodemp, ini.kavaprodtrig + 1, ini.kavaprodtrig)

                ''' Assign unisized trigger traces in boolean arrays '''
                if len(ini.d2t) > len(ini.tampl):
                    ini.idx4trigtime    = ini.d2t[:-np.int64(ini.shiftsp)].searchsorted(ini.tampl)
                    ini.kavatrigger     = ini.kavaprodtrig
                    ini.ratiotrigger    = np.interp(ini.d2t[:-np.int64(ini.shiftsp)], ini.tampl, ini.ratiotrigger2)
                    ini.ratiotrigger    = np.round(ini.ratiotrigger)
                elif len(ini.d2t) < len(ini.tampl):
                    ini.idx4trigtime    = ini.tampl.searchsorted(ini.d2t[:-np.int64(ini.shiftsp)])
                    ini.ratiotrigger    = ini.ratiotrigger2
                    ini.kavatrigger     = np.interp(ini.tampl, ini.d2t[:-np.int64(ini.shiftsp)], ini.kavaprodtrig)
                    ini.kavatrigger     = np.round(ini.kavatrigger)
                else:
                    print("Number of indices in kava time and ampl time are equal.")
                    ini.idx4trigtime    = np.arange(len(ini.tampl))
                    ini.kavatrigger     = ini.kavaprodtrig
                    ini.ratiotrigger    = ini.ratiotrigger2

                # --- Fuse traces of trigger elements regarding time and frequency domain and group activies (look up at runPlotCatalog.py)
                # ini.eventmarker     = kt.fusetriggers(ini.kavatrigger, ini.ratiotrigger, ini.tampl, ini.triggerresttime)
                ini.eventmarker     = kt.fusetriggers(ini.kavatrigger, ini.ratiotrigger, ini.tampl, tb)


                ''' Assign event variables for printing/plotting. '''
                if len(ini.d2t[:-np.int64(ini.shiftsp)]) > len(ini.tampl):
                    ini.eventtime           = ini.d2t[:-np.int64(ini.shiftsp)][np.where(ini.eventmarker == 1)]
                    ini.eventkava           = ini.kavaprod2[np.where(ini.eventmarker == 1)] # kavaprod2 is the max envelope of kavaprod

                    ratio2_1forevents       = np.interp(ini.d2t[:-np.int64(ini.shiftsp)], ini.tampl, ini.ratio2_1)
                    ini.eventratio          = ratio2_1forevents[  np.where(ini.eventmarker == 1)]

                elif len(ini.d2t[:-np.int64(ini.shiftsp)]) < len(ini.tampl):
                    ini.eventtime           = ini.tampl[np.where(ini.eventmarker == 1)]
                    ini.eventratio          = ini.ratio2_1[np.where(ini.eventmarker == 1)]

                    kavaprod2forevents      = np.interp(ini.tampl, ini.d2t[:-np.int64(ini.shiftsp)], ini.kavaprod2)
                    ini.eventkava           = kavaprod2forevents[np.where(ini.eventmarker == 1)]

                print(f"{'Events found':<20} | {len(ini.eventtime)} events with amplitude ratio and KaVA product trigger.")

                # --- Write output to file ------------------------------------------------------------------------------------------
                '''
                Append line in catalogue for each event triggered by both ratio and kava prod at the same time.
                Note datetime as utc, amplitude ratio and kava product in its respective column. 
                If both triggers stay in phase for more than one indices, just append the first appearence.
                '''
                
                if ini.write_flag:
                    # --- Write output to pkl file ------------------------------------------------------------------------------------------
                    utc_time_pkl = pd.to_datetime([pd.Timestamp(idate[0]) + pd.Timedelta(seconds=float(s)) for s in ini.eventtime])
                    df_chunk_ratio = pd.DataFrame({
                        'App_event_time(UTC)':      utc_time_pkl,
                        'Norm_amplitude_ratio[]':   np.round(ini.eventratio, 2),
                        'Adv_KaVA_idx':             np.round(ini.eventkava, 2)
                    })

                    results_ratio[tb].append(df_chunk_ratio)

                    if tidx == len(dayslist) - 1:
                        for tb, dfr in results_ratio.items():
                            final_df_ratio = pd.concat(dfr, ignore_index=True)
                            final_df_ratio.to_pickle(ini.outputcatalog[:-4] + f'{tb}s_ratio.pkl')
                            if final_df_ratio.empty:
                                print("--> Warning: The concatenated DataFrame final_df_ratio is empty.\n",
                                      "------------------------------------------------------------\n",
                                      "Set breakpoint to check for events and finish script manually.")
                                breakpoint()
                            else:
                                print(f"{'Output catalogue':<20} | {ini.outputcatalog[:-4] + f'{tb}s_ratio.pkl'}\n",
                                      "Head --- tail:\n",
                                      final_df_ratio.head(5),
                                      "----------\n",
                                      final_df_ratio.tail(5))


        if ini.compare_different_triggers is True or ini.compare_different_triggers == 'frequency':
            # --- 0.0 Find and determine kavachi events based on kava index ONLY ---
            # --- 0.1 Compute kava product (shark)
            dict_comp_adv_kava          = kt.comp_adv_kava(ini.kava1, ini.kava2, ini.d1t, window=ini.window_smooth_kavaprod, enhanced_separate=True)
            ini.kavaprod_slim           = dict_comp_adv_kava[0]
            ini.kavaprod2_slim          = dict_comp_adv_kava[1]
            ini.kavatime_slim           = pd.to_datetime([pd.Timestamp(idate[0]) + pd.Timedelta(seconds=float(s)) for s in ini.d2t[:-np.int64(ini.shiftsp)]])
            ini.shark_qualifier_slim    = pd.Series(ini.kavaprod2_slim).rolling(window=10, min_periods=1).max().to_numpy()
            ini.shark_qualifier_slim    = np.where(ini.shark_qualifier_slim > ini.kavaprodemp, 1, 0)
            
            # --- 0.2 Find peaks in shark | Apply trigger rest time
            
            for tb in ini.triggerresttime:
                print(f"--> {tb} sec trigger rest time applied.")
                ini.pre_peaks_idx, peakprops = signal.find_peaks(ini.kavaprod2_slim, height=ini.kavaprodemp, distance=tb//ini.dtspec)
                kt.fusetriggers_v2(pause=tb)

                ini.eventtime_slim          = ini.kavatime_slim[ini.peaks_idx]
                ini.eventshark_slim         = ini.kavaprod2_slim[ini.peaks_idx]
                ini.eventtime_slim_seconds  = ini.d2t[ini.peaks_idx[ini.peaks_idx < len(ini.d2t[:-np.int64(ini.shiftsp)])]]

                # --- 0.3 Assign event variables for output
                if ini.write_flag:
                    df_out = pd.DataFrame({
                        'Eventtime(UTC)': ini.eventtime_slim,
                        'Eventkavaproduct': ini.eventshark_slim
                    })
                    results[tb].append(df_out)
                    print(f"{'Events found':<20} | {len(ini.eventtime_slim)}  | for {tb} resting time & threshold of {ini.kavaprodemp} | {ini.idate.strftime('%Y-%m-%d')}",)
                            # df_out.head(5),
                            # "----------\n",
                            # df_out.tail(5))
                
                # 0.5 Rough overview plot
                # breakpoint()
                # peakheights                 = ini.kavaprod2_slim[ini.peaks_idx]
                # prepeakheights              = ini.kavaprod2_slim[ini.pre_peaks_idx]
                # plt.plot(ini.kavatime_slim, ini.kavaprod2_slim, label='KaVA product', color='black', alpha=0.5)
                # plt.scatter(ini.kavatime_slim[ini.pre_peaks_idx], prepeakheights, marker='o',label='SHARK peaks', color='orange', alpha=0.8)
                # plt.scatter(ini.kavatime_slim[ini.peaks_idx], peakheights, marker='o',label='SHARK detections', color='red', alpha=0.8)
                # # plt.scatter(ini.kavatime_slim, sharktrigger, marker='^', label='Shark trigger', color='blue', alpha=0.5)
                # plt.fill_between(ini.kavatime_slim, 0, np.max(ini.kavaprod2_slim), where=ini.shark_qualifier_slim > 0, color='tab:blue', alpha=0.5)
                # plt.legend()
                # # plt.close() # 
                # # plt.show()
        
            # --- Concatenate and save output
            if (tidx == len(dayslist) - 1 ) and ini.write_flag:
                for tb, dfs in results.items():
                    breakpoint()
                    final_df = pd.concat(dfs, ignore_index=True)
                    final_df.to_pickle(ini.outputcatalog[:-4] + f'{tb}s_shift_{ini.shifttime}s_slim.pkl')
                    if final_df.empty:
                        print("--> Warning: The concatenated DataFrame final_df is empty.\n",
                            "------------------------------------------------------------\n",
                            "Set breakpoint to check for events and finish script manually.")
                        breakpoint()
                    else:
                        print(f"--> {'Output catalogue':<20} | {ini.outputcatalog[:-4] + f'{tb}s_shift_{ini.shifttime}s_slim.pkl'}\n",
                              f"--> In total {len(final_df)} events found from {datetime_start} to {datetime_end}")
                                # "Head --- tail:\n",
                            #   final_df.head(5),
                            #     "----------\n",
                            #   final_df.tail(5))
             

            
            if hasattr(ini, 'visualise_shark_peaks_and_rmv'):
                if ini.visualise_shark_peaks_and_rmv is True and len(ini.triggerresttime) ==1:
                    fig, ax = plt.subplots(3,1,figsize=(8,4), sharex=True)
                    ax0twin, ax1twin = ax[0].twinx(), ax[1].twinx()
                    sequ = np.arange(1667*2.8,1667*4.6+1, dtype=int)
                    # sequ = np.arange(1667*5,1667*8+1, dtype=int)
                    prepeaksinsequ = ini.pre_peaks_idx[(ini.pre_peaks_idx >= sequ[0]) & (ini.pre_peaks_idx <= sequ[-1])]
                    peaksinsequ = ini.peaks_idx[(ini.peaks_idx >= sequ[0]) & (ini.peaks_idx <= sequ[-1])]
                    peaksrmvinsequ = np.setdiff1d(prepeaksinsequ, peaksinsequ)
                    ax[0].set_title(f"{ini.idate} UTC+11 - {ini.stationdir[1]}/{ini.stationdir[0]}")
                    ax[0].pcolor(ini.kavatime_slim[sequ], ini.d2f[:153], np.log10(ini.d2spec[:153,sequ]), cmap='jet', shading='auto')
                    ax0twin.plot(ini.kavatime_slim[sequ], ini.kava2[sequ], 'k', label='KaVA 2')
                    ax0twin.legend(loc='upper right', fontsize=ini.fontsizelegend)

                    ax[1].pcolor(ini.kavatime_slim[sequ], ini.d2f[:153], np.log10(ini.d1spec[:153, sequ+int(ini.shiftsp)]), cmap='jet', shading='auto')
                    ax1twin.plot(ini.kavatime_slim[sequ], ini.kava1[sequ+int(ini.shiftsp)], 'k-', label='KaVA 1')
                    ax1twin.legend(loc='upper right', fontsize=ini.fontsizelegend)
                    [ax.set_ylabel('[Hz]') for ax in [ax[0], ax[1]]]
                    [ax.set_ylabel('Amplitude\nKaVA') for ax in [ax0twin, ax1twin]]

                    ax[2].fill_between(ini.kavatime_slim[sequ], 0, ini.kavaprod2_slim[sequ], where=ini.shark_qualifier_slim[sequ] > 0, color='tab:blue', alpha=0.5)
                    ax[2].plot(  ini.kavatime_slim[sequ], ini.kavaprod2_slim[sequ], label='SHARK', color='black', alpha=0.5)
                    ax[2].scatter(ini.kavatime_slim[peaksinsequ], ini.kavaprod2_slim[peaksinsequ], marker='o', label='Detections', color='red', alpha=0.8)
                    ax[2].scatter(ini.kavatime_slim[peaksrmvinsequ], ini.kavaprod2_slim[peaksrmvinsequ], marker='o', label='Removed\npeaks', color='orange', alpha=0.8)
                    ax[2].legend(loc='upper right', fontsize=ini.fontsizelegend)
                    ax[2].set_ylabel('Amplitude\nSHARK') #\n(norm.)')
                    # ax[2].set_yticks([0,1]); ax[2].set_ylim([-.1,1*1.1])
                    # Format time labels for shared x axis
                    ax[2].set_xlabel('Time')
                    
                    ax[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
                    # ax[2].xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=2))
                    fig.align_ylabels()
                    fig.autofmt_xdate()
                    fig.subplots_adjust(hspace=.1)
                    plt.savefig(os.path.join(ini.rootouts, 'detection_a_removed_peaks_with_indices_bbd_empiric'+ini.idate.strftime('%Y%m%d')+'.png'), dpi=300, bbox_inches='tight')
                    plt.show()

            # --- Plotting - frequency based trigger
            if ini.plot_flag and len(ini.triggerresttime) == 1:
                if datetime_start.date() == datetime_end.date():
                    plotintervals = pd.date_range(start=datetime_start, end=datetime_end, freq=str(ini.sectionlen)+'S')
                else:
                    plotintervals = pd.date_range(start=idate[0], periods=60*60*24/ini.sectionlen, freq=str(ini.sectionlen)+'S')
                print('\n --> Plotting started for ' + idate[0].strftime('%Y-%m-%d') + ' .\n')

                for iplot in tqdm(plotintervals):
                    # breakpoint()
                    plutil.plot_overview_slim(iplot)
                    print('\n --> Plotted ' + iplot.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + (iplot+timedelta(seconds=ini.sectionlen)).strftime('%Y-%m-%d %H:%M:%S') + ' .\n')
                    print('\n --> Plotting finished for ' + idate[0].strftime('%Y-%m-%d') + ' .\n')


        # --- Plotting - freq and amplitude based trigger
        if ini.compare_different_triggers is True or ini.compare_different_triggers == 'amplitude':
            if ini.plot_flag and len(ini.triggerresttime) == 1:

                tr1, tr2        = st1[0].copy(), st2[0].copy()
                if ini.stationdir[0] in ini.bbd_ids:
                    tr1.filter('bandpass', freqmin=0.1, freqmax=99, corners=ini.bandpasscorners, zerophase=True)
                else:
                    tr1.filter('bandpass', freqmin=3, freqmax=99, corners=ini.bandpasscorners, zerophase=True)

                if ini.stationdir[1] in ini.bbd_ids:
                    tr2.filter('bandpass', freqmin=0.1, freqmax=99, corners=ini.bandpasscorners, zerophase=True)
                else:
                    tr2.filter('bandpass', freqmin=3, freqmax=99, corners=ini.bandpasscorners, zerophase=True)

        
                if datetime_start.date() == datetime_end.date():
                    plotintervals = pd.date_range(start=datetime_start, end=datetime_end, freq=str(ini.sectionlen)+'S')
                else:
                    plotintervals = pd.date_range(start=idate[0], periods=60*60*24/ini.sectionlen, freq=str(ini.sectionlen)+'S')
                print('\n --> Plotting started for ' + idate[0].strftime('%Y-%m-%d') + ' .\n')
                print(plotintervals)

                for iplot in tqdm(plotintervals): #np.arange(0, 60*60, sectionlen):

                    plutil.plot_overview(
                        time_waveform       = ini.time.copy(),
                        time_amplitudes     = ini.tampl.copy(),
                        time_spectra        = ini.d1t.copy(),
                        time_ratiopeaks     = ini.tratiopeaks.copy(),
                        time_eventtime      = ini.eventtime.copy(),
                        trace1              = ini.tr1.copy(),
                        trace2              = ini.tr2.copy(),
                        ratioarray          = ini.ratio2_1.copy(),
                        ratiopeaks          = ini.ratiotrigger2.copy(),
                        kavatrigger         = ini.kavatrigger.copy(),
                        spectrogram1        = ini.d1spec.copy(),
                        spectrogram2        = ini.d2spec.copy(),
                        specfrequencies1    = ini.d1f.copy(),
                        specfrequencies2    = ini.d2f.copy(),
                        kavaidx1            = ini.kava1.copy(),
                        kavaidx2            = ini.kava2.copy(),
                        kavaidxproduct      = ini.kavaprod.copy(),
                        kavaidxproduct2     = ini.kavaprod2.copy(),
                        eventkava           = ini.eventkava.copy(),
                        outputdir           = ini.outputdir,
                        h                   = iplot.hour,
                        idoi                = iplot.day,
                        imoi                = iplot.month,
                        s                   = iplot.second + iplot.minute*60,
                        ratiorange          = ini.ratiorange)
                    print('\n --> Plotted ' + iplot.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + (iplot+timedelta(seconds=ini.sectionlen)).strftime('%Y-%m-%d %H:%M:%S') + ' .\n')
                print('\n --> Plotting finished for ' + idate[0].strftime('%Y-%m-%d') + ' .\n')

    print('\n --> Skipped dates: \n', skipped_dates)
    print('\n --> Running script finished after %.02f seconds ---' % (tm.time() - ini.start_time))    
    # breakpoint()
           
if __name__ == '__main__':
    from kav_init import *
    import kav_init as ini
    # ini.use_three_components = False
    # ini.shifttime = 3.4
    runPlotCatalog(datetime_start='2023-04-05', datetime_end='2023-04-06')
    # runPlotCatalog(datetime_start='2023-06-01', datetime_end='2023-06-03')

    # runPlotCatalog(trigbreak=20, datetime_start='2023-02-01', datetime_end='2023-12-01')
    # runPlotCatalog(trigbreak=20, datetime_start='2023-04-05T06:00:00', datetime_end='2023-04-06')
    # runPlotCatalog(trigbreak=20, datetime_start='2023-04-05', datetime_end='2023-04-06')

