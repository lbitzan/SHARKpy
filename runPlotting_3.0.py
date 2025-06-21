import os
if not os.getcwd() == 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts':
    os.chdir('C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts')

from    obspy.core           import  read, UTCDateTime, Stream, Trace
from    obspy.imaging        import  spectrogram
import  numpy                as      np
import  datetime
import  matplotlib.pyplot    as      plt
import  matplotlib.colorbar  as      cbar
from    scipy                import  io, signal
from    scipy.fft            import  fftshift
import  pandas               as      pd
import  time                 as      tm
from datetime                import timedelta
from colorbar_fct            import add_colorbar_outside
from kavutil                 import * # rolling_stats, myplot, compute_kava, get_data, compute_banded_ratio
# -----------------------------------------------------------------------------------------------------------------------
'''
>>> runPlotting.py <<<
Script to evaluate data from Kavachi volcano, Solomon Islands.
Further parameters are computated and plotted within this script and attached routines.
This script uses routines from the KavachiProject/KavScripts directory (e.g. kavutil.py, colorbar_fct.py,...)

Author:     Ludwig Bitzan, 2023
created:    2024-01-30
'''

# Get initials
from kav_init import *

print('rootcode    :', rootcode); print('rootproject :', rootproject); print('rootdata    :', rootdata)

start_time = tm.time()

# --- IMPORT DATA -----------------------------------------------------------------------------------------------------
'''
Set seismological stations which are considered in analysis. One station per close and remote array. 
Set also time period of interest.
earliest date: 2023-02-07
latest date:   2023-07-25
'''

# --- Run loop over time ---------------------------------------------------------------------------------------------------
for imoi in moi:

    if month_flag:
       doi = org_days(stationdir=stationdir, yoi=yoi, imoi=imoi)
            
    if hour_flag:
        hoi = np.arange(0,24,1)

    for idoi in doi:
        
        print('--- Plotting started for date ' + '{:02d}'.format(idoi) +'/'+ '{:02d}'.format(imoi)+'/'+'20'+str(yoi)+' ---')

        st1, _ = get_data(year=yoi, month=imoi, day=idoi, rootdata=rootdata, stationdir=stationdir[0], stationid=stationid[0])
        st2, _ = get_data(year=yoi, month=imoi, day=idoi, rootdata=rootdata, stationdir=stationdir[1], stationid=stationid[1])

        # Create output directory corresponding to date
        outputdir    = rootouts+outputlabel+'/'+outputlabel+'20' + '{year}'.format(year=yoi) +'{:02d}'.format(imoi)+'/'+outputlabel+'20'+ '{year}'.format(year=yoi) + '{:02d}'.format(imoi) + '{:02d}'.format(idoi) +'/' # <--- set path outputdir for  larger data set on external hard drive
        outputcatalog= rootouts+cataloguefile                                                         # <--- set path and name for output catalogue for larger data set on external hard drive
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        # --- Computations ------------------------------------------------------------------------------------------------- 
        ''' Assign traces to variables for computations in time domain. '''
        tr1          = st1[0]
        tr2          = st2[0]
        ''' Assign and prefilter traces for computations in frequency domain. '''
        d1          = tr1.copy()
        d2          = tr2.copy()
        # d1.filter('bandpass', freqmin=3, freqmax=60, zerophase=True)
        # d2.filter('bandpass', freqmin=3, freqmax=60, zerophase=True)

        # --- Create time array relative to starttime
        time         = tr1.times(reftime = tr1.stats.starttime)
        
        # --- Compute amplitudes
        shift       = shifttime / tr1.stats.delta                                       # time delay between stations as no of indices

        if flag_bandedratio == False:
            
            ratio2_1, tampl = compute_simple_ratio(tr1, tr2, stats= stationdir)

        # --- compute amplitude ratio based on particular frequency bands --------------------------------------------------------------
        if flag_bandedratio:
           
           ratio2_1, tampl = compute_banded_ratio(tr1, tr2, freqbands=freqbands, stats= stationdir)
           


        # --- Spectrograms
        ''' Use d1, d2 as time series for spectrogram computations. Traces are filtered before between freqmin and freqmax.'''
        sr          = tr1.stats.sampling_rate       # sampling rate
        nfft            = int(sr)  # int(sr)
        noverlap        = int(sr*.75) #= int(sr/2)
        cspec           = plt.get_cmap('jet')                           # colormap for spectrogram 'viridis' 'plasma' 'inferno' 'magma' 'cividis' 'jet' 'gnuplot2'

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

        ratiotrigger2   = np.zeros_like(ratio2_1)
        ratiotrigger2[ratiotriggerpeaks] = 1

        pulselen        = dtsp/ratiodelta
        ratiotrigger2   = rolling_stats(ratiotrigger2, np.amax, window=np.int64(pulselen * .75))

        ratiopeaks      = ratio2_1[ratiotriggerpeaks]
        tratiopeaks     = tampl[ratiotriggerpeaks]

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
        
        # --- Implement combined event trigger based on both KaVA Index and amplitude ratio
        combtrigger     = np.zeros_like(kavatrigger)
        combtrigger     = np.where((ratiotrigger == 1) & (kavatrigger == 1), combtrigger + 1, combtrigger)

        eventmarker     = np.zeros_like(combtrigger)
        eventmarker[1:] = np.where(combtrigger[1:] - combtrigger[:-1] == 1, eventmarker[1:] +1, eventmarker[1:])
        eventmarker[0]  = 0
        
        # --- Group activities to single events
        if flag_group == True:
            trigger = np.zeros_like(eventmarker, dtype='int')
            switch  = True
            activities = np.array(kavatrigger) > 0
            for itrig, A in enumerate(activities):
                if A == False:
                    switch  = True
                    continue
                elif (A == True) & (switch == True):
                    if eventmarker[itrig] == True:
                        trigger[itrig]  = 1
                        switch          = False
                    else:
                        continue
                elif (A == True) & (switch == False):
                    continue
            eventmarker = trigger       
        

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
            
                time2catalogue      = datetime.datetime(year=2023, month=imoi, day=idoi, hour=int(hour4cata), minute=int(min4cata), second=int(sec4cata))

                utc_time            = time2catalogue.strftime("%d-%b-%Y %H:%M:%S")
                amplitude_ratio     = np.round(eventratio[i], 2)
                adv_kava_idx        = np.round(eventkava[i], 2)
                with open(outputcatalog, 'a') as file:
                    file.write(f"{utc_time}, {amplitude_ratio}, {adv_kava_idx}\n")

        # --- Plotting ------------------------------------------------------------------------------------------------------
        tr1         = st1[0].copy()
        tr2         = st2[0].copy()
        tr1.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)
        tr2.filter('bandpass', freqmin=3, freqmax=99, zerophase=True)

        if plot_flag:
            minutes     = np.copy(time)
            minutes     = time/60
            # sectionlen  = 60*3 # length of section that is evaluated [s]

            markeroffset= 20
            yaxlimshark = 2000

            for h in hoi:
                print('--- Plotting started for ' + '{:02d}'.format(h) + ':00 h ' + '{:02d}'.format(idoi) +'/'+ '{:02d}'.format(imoi)+'/'+'20'+str(yoi)+' ---')
                print('--- %.02f seconds ---' % (tm.time() - start_time))
                
                for s in np.arange(0, 60*60, sectionlen):
                    # Create indices:
                    # idxtr   = h*60*60/tr04.stats.delta + s/tr04.stats.delta
                    idxtr       = np.where((time  >= h*60*60 + s) & (time  < h*60*60 + s + sectionlen))
                    idxampl     = np.where((tampl >= h*60*60 + s) & (tampl < h*60*60 + s + sectionlen))
                    idxsp       = np.where((d1t   >= h*60*60 + s) & (d1t   < h*60*60 + s + sectionlen))
                    idxevent    = np.where((eventtime >= h*60*60 + s) & (eventtime < h*60*60 + s + sectionlen))
                    idxratiopeaks = np.where((tratiopeaks >= h*60*60 + s) & (tratiopeaks < h*60*60 + s + sectionlen))
                    
                    if (idxtr[0][-1]+int(shift)) > len(tr1.data):
                        print('--- Plotting finished for date ' + '{:02d}'.format(idoi) +'/'+ '{:02d}'.format(imoi)+'/'+'20'+str(yoi)+' ---')
                    else:
                        # Create timestamps:
                        m = 0
                        if s>59: m = s//60; s = s-m*60
                        customtime     = datetime.datetime(2023, imoi, idoi, h, m, s)

                        # Definfe array slices for plotting
                        timeplt         = time[         idxtr] #- h*60*60 - m*60
                        tr2plt          = tr2[          idxtr]
                        tr1plt          = tr1[          idxtr + np.int64(shift)]

                        tamplplt        = tampl[      idxampl] #- h*60*60 - m*60 
                        ratio2_1plt     = ratio2_1[    idxampl]
                        ratio2_1plt     = ratio2_1[idxampl]
                        ratiotriggerplt = ratiotrigger2[idxampl]

                        tratiopeaksplt  = tratiopeaks[idxratiopeaks] #- h*60*60 - m*60
                        ratiopeaksplt   = ratiopeaks[ idxratiopeaks]
                        
                        tspecplt        = d1t[        idxsp] #- h*60*60 - m*60
                        spec2plt        = d2spec[:fmax+1,     idxsp[0][:]]
                        spec1plt        = d1spec[:fmax+1,     idxsp[0][:] + int(shiftsp)]
                        kava2plt        = kava2[      idxsp]
                        kava1plt        = kava1[ (idxsp + np.int64(shiftsp))].flatten()

                        kavaprodplt     = kavaprod[    idxsp]
                        kavaprod2plt    = kavaprod2[   idxsp]
                        kavatriggerplt  = kavatrigger[idxampl]
                        combtriggerplt  = combtrigger[idxampl]

                        eventtimeplt    = eventtime[idxevent] #- h*60*60 - m*60
                        eventkavaplt    = eventkava[idxevent]
                        eventkavaplt[eventkavaplt >= yaxlimshark] = yaxlimshark - 100 # set max value 2 scale plotting
                        
                        # --- Setup figure ------------------------------------------------------------------------------------------------
                        figkav, (ax1, ax2, ax34, ax3, ax4) = plt.subplots(5, 1, figsize=(18, 10), sharex=True)
                        # figkav.suptitle('Kavachi volcanic activity analysis ' + stationdir[1] + ' / ' + stationdir[0] +' - '+ customtime.strftime("%H:%M:%S %B %d, %Y"))
                        ax1.set_title('Kavachi volcanic activity analysis ' + stationdir[1] + ' / ' + stationdir[0] +' - '+ customtime.strftime("%H:%M:%S %B %d, %Y"))

                        # --- plot waveforms close station --------------------------------------------------------------------------------
                        color   = 'tab:red'
                        hand1   = ax1.plot(timeplt, tr2plt, color=color, alpha=.5, linewidth=0.4, label='WF '+ stationdir[1])
                        ax1.tick_params(axis='x', labelbottom=False, labeltop=False, labelrotation=45)
                        ax1.set_ylabel('WF \n'+ stationdir[1], rotation=90, labelpad=10, color=color)
                        ax1.tick_params(axis='y', labelcolor=color)
                        ax1.set_ylim(-np.amax(np.abs(tr2plt)), np.amax(np.abs(tr2plt)))
                        ax1.grid(alpha=0.3)

                        # --- plot waveforms remote station -------------------------------------------------------------------------------
                        ax1twin = ax1.twinx()
                        color   = 'tab:blue'
                        hand2   = ax1twin.plot(timeplt, tr1plt[0,:], color=color, alpha=.5, linewidth=0.4, label='WF '+ stationdir[0])
                        ax1twin.set_ylabel('WF  '+ stationdir[0]+ '\n'+ str(shifttime) +' s shifted', rotation=90, labelpad=10, color=color)
                        ax1twin.tick_params(axis='y', labelcolor=color)
                        ax1twin.set_ylim(-np.amax(np.abs(tr1plt)), np.amax(np.abs(tr1plt)))
                        myplot(axoi=ax1, handles=[hand1, hand2])

                        # --- plot amplitude ratio ------------------------------------------------------------------------------
                        cax2 = ax2.scatter( tamplplt, ratio2_1plt, c = 'seagreen', edgecolors='darkgreen', alpha=.5, label='ampl. ratio') # cmap = 'RdYlBu' 'RdPu' 'set1' 'nipy_spectral'
                        ax2.fill_between(tamplplt, 0, np.max(ratio2_1plt)+markeroffset+5, where=(kavatriggerplt  ==1), facecolor='tab:red',  alpha=0.1, label='dual-station trigger')
                        ax2.fill_between(tamplplt, 0, np.max(ratio2_1plt)+markeroffset+5, where=(ratiotriggerplt ==1), facecolor='tab:blue', alpha=0.4, label='ratio trigger')
                        ax2.hlines(ratiorange, tamplplt[0], timeplt[-1], colors='dimgray', linestyles='--', alpha=0.8, label='confidence range')

                        ax2.set_ylabel('Amplitude ratio\n' + stationdir[1] + ' / ' + stationdir[0], rotation=90, labelpad=10)
                        ax2.scatter(tratiopeaksplt, ratiopeaksplt+markeroffset, s= 30, c='orange', marker='v', alpha=1)
                        # ax2.set_yticks(np.arange(0, np.amax(ratio2_1plt),1))
                        myplot(axoi=ax2)
                        # add_colorbar_outside(cax2, ax2)
                        
                        # --- Plot SHARK Index (KaVA Product) ----------------------------------------------------------------------
                        ax34.plot(tspecplt, kavaprodplt, color='k', alpha=0.8, linewidth=1, label='KaVA product')
                        ax34.plot(tspecplt, kavaprod2plt, color='b', alpha=0.5, linewidth=1,linestyle='--', label='KaVA product envelope')
                        ax34.fill_between(tamplplt, 0, np.max(kavaprodplt), where=(kavatriggerplt  ==1), facecolor='tab:red',  alpha=0.4, label='dual-station trigger')
                        ax34.fill_between(tamplplt, 0, np.max(kavaprodplt), where=(ratiotriggerplt ==1), facecolor='tab:blue', alpha=0.4, label='ratio trigger')
                        ax34.scatter(eventtimeplt, eventkavaplt, c='r', marker=(5,2), label='events')
                        ax34.set_ylabel('KaVA \n product', rotation=90, labelpad=10)
                        if np.max(kavaprodplt) > 3200:
                            ax34.set_ylim(0, 3200)
                        ax34.legend(loc='upper right'); ax34.grid(alpha=0.3)
                        myplot(axoi=ax34)

                        # --- plot spectrogram close station -------------------------------------------------------------------------------
                        ax3twin = ax3.twinx()
                        cax3 = ax3.pcolor(tspecplt, d2f[:fmax+1], np.log10(spec2plt), shading='nearest', cmap=cspec)
                        ax3twin.plot(tspecplt,       kava2plt,       color='k', alpha=0.8, linewidth=1, label='KaVA')
                        ax3.set_ylabel('Spectrogram \n '+ stationdir[1]+ '    [Hz]',  rotation=90, labelpad=10)
                        ax3.set_ylim(0, fmax+8)
                        ax3twin.set_ylabel(' KaVA Index\n Station '  + stationdir[1],  rotation=90, labelpad=10)
                        # add_colorbar_outside(cax3, ax3) # test
                        ax3twin.legend(loc='upper right')

                        # --- Plot spectrogram remote station ------------------------------------------------------------------------------
                        ax4twin = ax4.twinx()
                        ax4.pcolor(tspecplt, d1f[:fmax+1], np.log10(spec1plt), shading='nearest', cmap=cspec)
                        ax4twin.plot(tspecplt,       kava1plt,       color='k', alpha=0.8, linewidth=1, label='KaVA')
                        ax4.set_ylabel('Spectrogram \n  '+ stationdir[0]+'    [Hz]', rotation=90, labelpad=10)
                        ax4.set_ylim(0, fmax+8)
                        ax4twin.set_ylabel('KaVA Index\n Station '  + stationdir[0], rotation=90, labelpad=10)

                        ax4.set_xlabel('Time')
                        xticks = np.array(np.arange(timeplt[0], timeplt[-1], sectionlen//18), dtype=int)
                        xticklabels = np.array([str(timedelta(seconds=s)) for s in np.array(xticks, dtype="float")])

                        ax4.set_xticks(xticks, xticklabels)

                        ax4.tick_params(axis='x', labeltop=False, labelbottom=True, labelrotation=45)
                        ax4twin.legend(loc='upper right')

                        figkav.align_ylabels()

                        plt.subplots_adjust(hspace=0.)


                        # Save figures
                        if save_flag == True:
                            # plt.savefig(rootproject+'/results/mixed/mixed'+'{:02d}'.format(idoi) + '{:02d}'.format(imoi) +'/mixed'+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.png', dpi=300, bbox_inches='tight') # <--- set path outputdir for smaller data set on local machine
                            plt.savefig(outputdir+'/'+outputlabel+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.png', dpi=1200, bbox_inches='tight')   # <--- set path outputdir for larger data set on external hard drive
                            # plt.savefig(outputdir+'/'+outputlabel+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.svg',format='svg', dpi=300, bbox_inches='tight')   # <--- set path outputdir for larger data set on external hard drive
                            plt.close(figkav)
                        else:
                            plt.show()
                                
                        print('--- %.02f sec elapsed - plotting in progress ---' % (tm.time() - start_time))


print('--- Running script finished after %.02f seconds ---' % (tm.time() - start_time))    
            

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
