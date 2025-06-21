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
from kav_init import *

def runPlotCatalog(datetime_start: str | np.datetime64,
                   datetime_end: str | np.datetime64,
                   rrange = None, trigbreak = None):
    '''
    This scripts evaluates data from the Kavachi Survey, from 2023 ongoing. Particular dates and stations should be specified in the initial file kav_init.py.
    Depending on the flags set in the in the inital file, the output covers an overview plot as well as an event catalog.
    Advanced parameters can be set within in the functions syntax, the initial file or the associated modules.
    '''
    start_time = tm.time()
    if 'KavScripts' not in os.getcwd():
        if 'KavScripts' not in os.listdir():
            print('Working repository not found. Check path.')
            return
        else:
            os.chdir('KavScripts')
            print('Changed directory to: ', os.getcwd())
    elif 'KavScripts' in os.getcwd():
        print('Script already started from working repository: ', os.getcwd())

    from kavutil import org_days, get_data, compute_simple_ratio, compute_banded_ratio, compute_kava, rolling_stats, fusetriggers, add_colorbar_outside, get_days_list, get_data3
    
    from kav_init import rootproject, rootcode, rootdata, rootouts, cataloguefile, outputlabel, month_flag, hour_flag, plot_flag, write_flag
    from kav_init import stationdir, stationid, shifttime, ratiorange, flag_bandedratio, flag_wlvl, yoi, moi, doi, hoi, shifttime, ratiorange, triggerresttime

    if rrange:
        ratiorange = rrange

    if trigbreak:
        triggerresttime = trigbreak

    datetime_start, datetime_end = pd.to_datetime(datetime_start), pd.to_datetime(datetime_end)
    dayslist = get_days_list(datetime_start, datetime_end, stationdir, stationid)

    print('\n rootcode    :', rootcode); print('\n rootproject :', rootproject); print('\n rootdata    :', rootdata)
    print('')
    print('--- Script setup complete after '+ str(np.round(tm.time() - start_time, 1))+' sec ---\n --> Investigate time from ' + datetime_start.strftime('%Y-%m-%d') + ' to '+ datetime_end.strftime('%Y-%m-%d'))

    print('\n Triggerresttime:   ', triggerresttime, 'sec')
    print('\n Ampl. ratio range: ', str(ratiorange[0]),' to ' + str(ratiorange[1])+' sec')
    print('\n Threshold Shark:   ', str(kavaprodemp))
    
    print('Dayslist \n -------- \n', dayslist)
    # --- IMPORT DATA -----------------------------------------------------------------------------------------------------
    '''
    Set seismological stations which are considered in analysis. One station per close and remote array. 
    Set also time period of interest.
    earliest date: 2023-02-07
    latest date:   2023-07-25
    '''

    # --- Run loop over time ---------------------------------------------------------------------------------------------------
    # if month_flag:
    #     doi = org_days(stationdir=stationdir, yoi=yoi, imoi=imoi)      
    # if hour_flag:
    #     hoi = np.arange(0,24,1)
    skipped_dates = pd.DataFrame(columns=['skipped_dates'])
    for idate in dayslist:
        
        print('\n --> Computation started for ' + idate[0].strftime('%Y-%m-%d')+'.')

        # st1, _ = get_data(year=yoi, month=imoi, day=idoi, rootdata=rootdata, stationdir=stationdir[0], stationid=stationid[0])
        # st2, _ = get_data(year=yoi, month=imoi, day=idoi, rootdata=rootdata, stationdir=stationdir[1], stationid=stationid[1])

        st1 = get_data3(idate[0], rootdata=rootdata, stationdir=stationdir[0], stationid=stationid[0])
        st2 = get_data3(idate[0], rootdata=rootdata, stationdir=stationdir[1], stationid=stationid[1])
        if len(st1[0]) != len(st2[0]):
            print('Length of traces is not equal: Skip date.')
            skipped_dates.loc[len(skipped_dates)] = idate[0]
            continue

        # Create output directory corresponding to date
        # outputlabel  = outputlabel + '_'+str(triggerresttime)
        # outputdir    = rootouts+outputlabel+'/'+outputlabel+'20' + '{year}'.format(year=yoi) +'{:02d}'.format(imoi)+'/'+outputlabel+'20'+ '{year}'.format(year=yoi) + '{:02d}'.format(imoi) + '{:02d}'.format(idoi)+'/' # <--- set path outputdir for  larger data set on external hard drive
        outputdir    = rootouts+outputlabel+'.'+str(trigbreak)+'/'+outputlabel+ idate[0].strftime('%Y%m')+'/' + outputlabel+ idate[0].strftime('%Y%m%d')+'/' # <--- set path outputdir for  larger data set on external hard drive
        outputcatalog= rootouts+ cataloguefile+str(trigbreak)+'s.txt'
        # outputcatalog= rootouts+ 'rmax' +str(ratiorange[1]) + cataloguefile    #TODO: set back to original  # <--- set path and name for output catalogue for larger data set on external hard drive
        # outputcatalog= rootouts+ 'evcnt_rrtest_'+ str(ratiorange[1])[-3:]+'_'+cataloguefile       # <--- set path and name for output catalogue for larger data set on external hard drive
        print('outputdir:     ', outputdir)
        print('outputcatalog: ', outputcatalog)
        if plot_flag:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

        # --- Computations ------------------------------------------------------------------------------------------------- 
        ''' Assign traces to variables for computations in time domain. '''
        tr1, tr2    = st1[0].copy(), st2[0].copy()
        ''' Assign and prefilter traces for computations in frequency domain. '''
        d1, d2      = tr1.copy(), tr2.copy()
        # d1.filter('bandpass', freqmin=3, freqmax=60, zerophase=True); d2.filter('bandpass', freqmin=3, freqmax=60, zerophase=True)

        # --- Create time array relative to starttime
        time        = tr1.times(reftime = tr1.stats.starttime)
        shift       = shifttime / tr1.stats.delta                                       # time delay between stations as no of indices
                
        # --- Compute amplitudes
        if flag_bandedratio == False:
            ratio2_1, tampl = compute_simple_ratio(tr1, tr2, stats= stationdir)

        # --- compute amplitude ratio based on particular frequency bands --------------------------------------------------------------
        if flag_bandedratio:
            ratio2_1, tampl = compute_banded_ratio(tr1, tr2, freqbands=freqbands, stats= stationdir)
        
        # --- Spectrograms
        ''' Use d1, d2 as time series for spectrogram computations. Traces are filtered before between freqmin and freqmax.'''
        sr              = tr1.stats.sampling_rate
        nfft, noverlap  = int(sr), int(sr*.75)                              # sampling rate, number of fft points, overlap
        cspec           = plt.get_cmap('jet')                               # colormap for spectrogram 'viridis' 'plasma' 'inferno' 'magma' 'cividis' 'jet' 'gnuplot2'

        kava1, d1t, dtspec, d1spec, d1f, cax =  compute_kava(d1, noverlap=noverlap, station= stationdir[0], frequency_bands=freqbands, cmap=cspec, appl_waterlvl=flag_wlvl)
        kava2, d2t, dtspec, d2spec, d2f, cax =  compute_kava(d2, noverlap=noverlap, station= stationdir[1], frequency_bands=freqbands, cmap=cspec, appl_waterlvl=flag_wlvl)
        
        # --- Define time step between sample points
        dtsp            = d1t[1]-d1t[0]
        shiftsp         = shifttime/dtsp

        # --- Compute KaVA Index product according to time-shift
        # kavaprod      = kava2smooth[:-np.int64(shiftkavasmooth)] * kava1smooth[np.int64(shiftkavasmooth):]
        kavaprod        = kava2[:-np.int64(shiftsp)] * kava1[np.int64(shiftsp):]
        kavaprod2       = rolling_stats(kavaprod, np.amax, window=10)    # max envelope of kavaprod 6

        # --- Implement event trigger based on amplitude ratio        
        '''new ratio trigger:'''
        ratiodelta = tampl[1] - tampl[0]
        ratiotriggerpeaks,_ = signal.find_peaks(ratio2_1, height=ratiorange, distance=2/ratiodelta) # 2/(tampl[1]-tampl[0]) = 2 sec
        # ratiotriggerpeaks,_ = signal.find_peaks(ratio2_1, height=ratiorange, prominence=50) # ---> TODO: Try out

        ratiotrigger2   = np.zeros_like(ratio2_1)
        ratiotrigger2[ratiotriggerpeaks] = 1

        pulselen        = dtsp/ratiodelta
        ratiotrigger2   = rolling_stats(ratiotrigger2, np.amax, window=np.int64(pulselen * .75))

        ratiopeaks, tratiopeaks     = ratio2_1[ratiotriggerpeaks], tampl[ratiotriggerpeaks]

        # --- Implement event trigger based on KaVA Index
        kavaprodtrig    = np.zeros_like(kavaprod)
        kavaprodtrig    = np.where(kavaprod2 > kavaprodemp, kavaprodtrig + 1, kavaprodtrig)

        ''' Assign unisized trigger traces in boolean arrays '''
        if len(d2t) > len(tampl):
            idx4trigtime    = d2t[:-np.int64(shiftsp)].searchsorted(tampl)
            kavatrigger     = kavaprodtrig
            ratiotrigger    = np.interp(d2t[:-np.int64(shiftsp)], tampl, ratiotrigger2)
            ratiotrigger    = np.round(ratiotrigger)
        elif len(d2t) < len(tampl):
            idx4trigtime    = tampl.searchsorted(d2t[:-np.int64(shiftsp)])
            ratiotrigger    = ratiotrigger2
            kavatrigger     = np.interp(tampl, d2t[:-np.int64(shiftsp)], kavaprodtrig)
            kavatrigger     = np.round(kavatrigger)
        else:
            print("Number of indices in kava time and ampl time are equal.")
            idx4trigtime    = np.arange(len(tampl))
            kavatrigger     = kavaprodtrig
            ratiotrigger    = ratiotrigger2
                    
        # --- Fuse traces of trigger elements regarding time and frequency domain and group activies (look up at runPlotCatalog.py)
        # breakpoint()
        eventmarker     = fusetriggers(kavatrigger, ratiotrigger, tampl, triggerresttime)

        ''' Assign event variables for printing/plotting. '''
        if len(d2t[:-np.int64(shiftsp)]) > len(tampl):
            eventtime           = d2t[:-np.int64(shiftsp)][np.where(eventmarker == 1)]
            eventkava           = kavaprod2[np.where(eventmarker == 1)] # kavaprod2 is the max envelope of kavaprod

            ratio2_1forevents= np.interp(d2t[:-np.int64(shiftsp)], tampl, ratio2_1)
            eventratio          = ratio2_1forevents[  np.where(eventmarker == 1)]

        elif len(d2t[:-np.int64(shiftsp)]) < len(tampl):
            eventtime           = tampl[np.where(eventmarker == 1)]
            eventratio          = ratio2_1[np.where(eventmarker == 1)]

            kavaprod2forevents  = np.interp(tampl, d2t[:-np.int64(shiftsp)], kavaprod2)
            eventkava           = kavaprod2forevents[np.where(eventmarker == 1)]

        # --- Write output to file ------------------------------------------------------------------------------------------
        '''
        Append line in catalogue for each event triggered by both ratio and kava prod at the same time.
        Note datetime as utc, amplitude ratio and kava product in its respective column. 
        If both triggers stay in phase for more than one indices, just append the first appearence.
        '''
        if write_flag:
            if not os.path.exists(outputcatalog):
                with open(outputcatalog, 'w') as file:
                    file.write('App_event_time(UTC), Norm_amplitude_ratio[], Adv_KaVA_idx\n')
                
            for i in np.arange(len(eventtime)):
                seconds4catalogue   = eventtime[i]
                min4cata, sec4cata  = divmod(seconds4catalogue, 60)
                hour4cata, min4cata = divmod(min4cata, 60)
            
                time2catalogue      = datetime.datetime(year=idate[0].year, month=idate[0].month , day=idate[0].day, hour=int(hour4cata), minute=int(min4cata), second=int(sec4cata))

                utc_time            = time2catalogue.strftime("%d-%b-%Y %H:%M:%S")
                amplitude_ratio     = np.round(eventratio[i], 2)
                adv_kava_idx        = np.round(eventkava[i], 2)
                with open(outputcatalog, 'a') as file:
                    file.write(f"{utc_time}, {amplitude_ratio}, {adv_kava_idx}\n")

        # --- Plotting ------------------------------------------------------------------------------------------------------
        tr1, tr2        = st1[0].copy(), st2[0].copy()
        if stationdir[0] in bbd_ids:
            tr1.filter('bandpass', freqmin=0.1, freqmax=99, zerophase=True)
        else:
            tr1.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)
        
        if stationdir[1] in bbd_ids:
            tr2.filter('bandpass', freqmin=0.1, freqmax=99, zerophase=True)
        else:
            tr2.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)

        if plot_flag:
            if datetime_start.date() == datetime_end.date():
                plotintervals = pd.date_range(start=datetime_start, end=datetime_end, freq=str(sectionlen)+'S')
            else:
                plotintervals = pd.date_range(start=idate[0], periods=60*60*24/sectionlen, freq=str(sectionlen)+'S')
            print('\n --> Plotting started for ' + idate[0].strftime('%Y-%m-%d') + ' .\n')

            for iplot in tqdm(plotintervals): #np.arange(0, 60*60, sectionlen):

                # time_waveform, time_amplitudes, time_spectra, time_ratiopeaks, time_eventtime = time.copy(), tampl.copy(), d1t.copy(), tratiopeaks.copy(), eventtime.copy()
                # trace1, trace2, ratioarray, ratiotrigger, ratiopeaks          = tr1.copy(), tr2.copy(), ratio2_1.copy(), ratiotrigger2.copy(), ratiopeaks.copy()
                # kavatrigger, spectrogram1, spectrogram2, specfrequencies1, specfrequencies2 = kavatrigger.copy(), d1spec.copy(), d2spec.copy(), d1f.copy(), d2f.copy()
                # kavaidx1 kavaidx2 kavaidxproduct kavaidxproduct2 eventkava    = kava1.copy(), kava2.copy(), kavaprod.copy(), kavaprod2.copy(), eventkava.copy()

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
                    s                   = iplot.second + iplot.minute*60,
                    ratiorange          = ratiorange)
                print('\n --> Plotted ' + iplot.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + (iplot+timedelta(seconds=sectionlen)).strftime('%Y-%m-%d %H:%M:%S') + ' .\n')
            print('\n --> Plotting finished for ' + idate[0].strftime('%Y-%m-%d') + ' .\n')

    print('\n --> Skipped dates: \n', skipped_dates)
    print('\n --> Running script finished after %.02f seconds ---' % (tm.time() - start_time))    
    # breakpoint()
           
if __name__ == '__main__':
    from kav_init import *
    runPlotCatalog(trigbreak=20, datetime_start='2023-02-01', datetime_end='2023-12-01')
    # runPlotCatalog(trigbreak=20, datetime_start='2023-04-05T06:00:00', datetime_end='2023-04-06')
    # runPlotCatalog(trigbreak=20, datetime_start='2023-04-05', datetime_end='2023-04-06')

    # runPlotCatalog(trigbreak=10, datetime_start='2023-04-05T06:00:00', datetime_end='2023-04-06')
    # runPlotCatalog(trigbreak=10, datetime_start='2023-08-01', datetime_end='2023-11-30')
    # runPlotCatalog(trigbreak=10, datetime_start='2023-02-01', datetime_end='2023-11-30')
    

    '''
    --- To Do: ------------------------------------------------------------------------------------------------------------
    - Add loop over different years
    - Improve catalogue writing by considering event duration

    - Enhance trigger by adding a sta-lta trigger for the advanced KaVA Index, so that the event is not only triggered by the peaks.

    - add events in >timeshift_ana.py< to improve the results statistical validity.

    --- Done: ------------------------------------------------------------------------------------------------------------
    + Add feature to write output to file
    + Add feature to plot spectrograms
    + Add feature to plot KaVA Index
    + Add feature to plot amplitude ratio
    + Add feature to plot waveforms
    + Add feature to plot event trigger
    + Add feature to plot event marker
    + Add feature to combine trace snippeds of a single day to one trace to avoid crashing of the script.

    + Add loop over different months
    + Add loop over different days
    + Add loop over different hours and specific time sections

    + Add computation of amplitude ratio
    + Add computation of KaVA Index
    + Add computation of Advanced KaVA Index (KaVA Product)
    + Add computation of event trigger based on amplitude ratio and Advanced KaVA Index

    + Enhance temporal resolution of amplitude ratio by adjusting the window size and shift.

    + NewScript to read catalogue and perform statistic anaylisis on registered events
        - count events per day/week/month
        - compute mean of adv KaVA Idx for days/weeks/months for catalogue events
        - plot results in Histograms daily events/week; daily events/month; weekly events/month; weekly events/recording time; monthly events/recording time;
        - add adv KaVA means to Histograms described above

    + Add second station empiricals to the code to be able to chose a different station for the remote array. (Check 4 data availability first!)
    + Implement "compute_kava()" function
        
    '''
