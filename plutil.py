import  numpy               as np
import  matplotlib.pyplot   as plt
import  time                as tm
import  datetime
from    datetime            import timedelta
import  pandas              as pd
import  matplotlib.dates    as mdates
from    obspy               import UTCDateTime, Trace, Stream, read
from    tqdm                import tqdm
import  os as os
import  kav_init as ini
import  matplotlib.gridspec as gridspec

"""
Module: 
    plutil.py
Contains function to customize plotting of data and results regarding the Kavachi project.

List of functions:
------------------
    - plot_overview_slim
    - add_colorbar_outside
    - set_size

lb, 2024/06/01
"""

def add_colorbar_outside(im,ax):
    """
    Add a colorbar outside of the plot.
    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The image to which the colorbar is attached.
    ax : matplotlib.axes.Axes
        The axes to which the colorbar is attached.
    """
    fig     = ax.get_figure()
    bbox    = ax.get_position() # bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    width   = 0.01
    eps     = 0.01              # margin between plot and colorbar
    cax     = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar    = fig.colorbar(im, cax=cax)

def plot_overview_slim(iplot):
    '''
    Plot overview of the Kavachi volcanic activity analysis for a given date and time.
    Parameters
    ----------
    iplot : object
        An object containing the month, day, hour, minute, second, and absolute second for the plot.
    ini : object
        Class containing the necessary parameters and data for the plot, such as time arrays, traces, spectra, and indices.
    Returns
    -------
    None
    '''
    # Set time parameters and load time from ini
    imoi, idoi, h, m, s, s_abs = iplot.month, iplot.day, iplot.hour, iplot.minute, iplot.second, iplot.second+iplot.minute*60
    t_wf        = ini.time
    t_spec      = ini.d2t
    t_event     = ini.eventtime_slim_seconds

    # Prepare traces
    pretr1, pretr2 = ini.tr1.copy(), ini.tr2.copy()
    pretr1.detrend('demean')
    pretr2.detrend('demean')
    pretr2.filter('bandpass', freqmin=ini.fminbandpass[ini.stationdir[1]], freqmax=ini.fmaxbandpass[ini.stationdir[1]], corners=ini.bandpasscorners, zerophase=True)
    pretr1.filter('bandpass', freqmin=ini.fminbandpass[ini.stationdir[0]], freqmax=ini.fmaxbandpass[ini.stationdir[0]], corners=ini.bandpasscorners, zerophase=True)

    cmap        = plt.get_cmap(ini.cmapnamespec)

    # --- Organise indices
    idx_wf      = np.where((t_wf    >= h*60*60 + s_abs) & (t_wf     < h*60*60 + s_abs + ini.sectionlen))[0]
    idx_spec    = np.where((t_spec  >= h*60*60 + s_abs) & (t_spec   < h*60*60 + s_abs + ini.sectionlen))[0]
    idx_event   = np.where((t_event >= h*60*60 + s_abs) & (t_event  < h*60*60 + s_abs + ini.sectionlen))[0]
    idx_freq1   = np.where((ini.d1f >= (ini.fminbandpass[ini.stationdir[0]]//1)) & (ini.d1f <= (ini.fmaxbandpass[ini.stationdir[0]]//1)))[0]
    idx_freq2   = np.where((ini.d2f >= (ini.fminbandpass[ini.stationdir[1]]//1)) & (ini.d2f <= (ini.fmaxbandpass[ini.stationdir[1]]//1)))[0]

    if (idx_wf[-1]+np.int64(ini.shift)) > len(pretr1.data):
        print(f"{'-> Plotting finished for':<20} | {ini.idate.strftime('%Y-%m-%d')}")
    else:
        customtime          = datetime.datetime(ini.idate.year, imoi, idoi, h, m, s)
        timeplt             = t_wf[   idx_wf] #- h*60*60 - m*60 
        tspecplt            = t_spec[ idx_spec]
        teventplt           = t_event[idx_event]

        timeplt, tspecplt, teventplt = pd.to_datetime(customtime) + pd.to_timedelta(timeplt, unit='s'), pd.to_datetime(customtime) + pd.to_timedelta(tspecplt, unit='s'), pd.to_datetime(customtime) + pd.to_timedelta(teventplt, unit='s')
        timeplt, tspecplt, teventplt = [timearray - pd.to_timedelta(customtime.hour*60*60 + customtime.minute*60 + customtime.second, unit='s') for timearray in [timeplt, tspecplt, teventplt]]
        
        f2plt               = ini.d2f[idx_freq2]
        f1plt               = ini.d1f[idx_freq1]
        tr2plt              = pretr2[idx_wf].copy()
        tr1plt              = pretr1[idx_wf + int(ini.shift)].copy()
        kava2plt            = ini.kava2[idx_spec]
        kava1plt            = ini.kava1[ idx_spec                + int(ini.shiftsp)]
        spec2plt            = ini.d2spec[idx_freq2,:][:,idx_spec]
        spec1plt            = ini.d1spec[idx_freq1,:][:,idx_spec + int(ini.shiftsp)]
        sharkplt            = ini.kavaprod2_slim[idx_spec]
        eventsharkplt       = ini.eventshark_slim[idx_event]
        sharkqualifierplt   = ini.shark_qualifier_slim[idx_spec]

        # Custom height ratios for each panel (adjust as needed)
        height_ratios       = [1.1, 1, 1, 1.1]  # Example: waveform panel a bit taller
        labelrot            = 0                 # Rotation for x labels

        # fig = plt.figure(figsize=(18, 10))
        fig = plt.figure(figsize=(7, 3.88))
        gs = gridspec.GridSpec(4, 1, height_ratios=height_ratios, hspace=0.05)
        # fig.suptitle('Kavachi volcanic activity analysis ' + ini.stationdir[1] + ' / ' + ini.stationdir[0] +' - '+ customtime.strftime("%H:%M:%S %B %d, %Y"))

        # Panel 0: Waveforms
        ax0 = fig.add_subplot(gs[0])
        ax0.set_title('Kavachi volcanic activity analysis ' + ini.stationdir[1] + ' / ' + ini.stationdir[0] +' - '+ customtime.strftime('%Y-%m-%d %H:%M:%S' + ' UTC+11'), fontsize=ini.fontsizelegend)
        ax00 = ax0.twinx()
        c2, c1 = 'tab:red', 'tab:blue'
        handle0, = ax0.plot(timeplt, tr2plt, color=c2, label=f'WF {ini.stationdir[1]}', alpha=0.6, linewidth=0.5)
        ax0.set_ylabel('Waveforms', labelpad=10, color=c2, fontsize=ini.fontsizelegend)
        handle00, = ax00.plot(timeplt, tr1plt, color=c1, label=f'WF {ini.stationdir[0]}\n {ini.shifttime} s shifted', alpha=0.6, linewidth=0.5)
        ax00.set_ylabel('Waveforms', labelpad=10, color=c1, fontsize=ini.fontsizelegend)
        ax0.tick_params(axis='y', rotation=labelrot, colors=c2, labelsize=ini.fontsizelegend)
        ax00.tick_params(axis='y', rotation=labelrot, colors=c1, labelsize=ini.fontsizelegend)
        ax0.tick_params(axis='x', labelbottom=False, labeltop=False, labelrotation=labelrot)
        ax0.set_ylim(-np.amax(np.abs(tr2plt)), np.amax(np.abs(tr2plt)))
        ax00.set_ylim(-np.amax(np.abs(tr1plt)), np.amax(np.abs(tr1plt)))
        lines = [handle0, handle00]
        linelabels = [line.get_label() for line in lines]
        ax0.legend(lines, linelabels, loc='upper right', fontsize=ini.fontsizelegend)
        ax00.grid(alpha=0.5)

        # Panel 1: kava2 + spectrogram 2
        ax1  = fig.add_subplot(gs[1], sharex=ax0)
        ax11 = ax1.twinx()
        cax1 = ax1.pcolor(tspecplt, f2plt, np.log10(spec2plt), shading='nearest', cmap=cmap)
        ax11.plot(tspecplt, kava2plt, color='k', label=f"KaVA Idx\n{ini.stationdir[1]}", linewidth=0.8)
        ax1.set_ylabel('[Hz]', rotation=90, labelpad=10, fontsize=ini.fontsizelegend)
        ax1.set_ylim(0, 65)
        yticks = np.arange(0, 63, 20)
        ax1.set_yticks(yticks)
        [a.tick_params(axis='y', rotation=labelrot, labelsize=ini.fontsizelegend) for a in [ax1, ax11]]
        ax1.tick_params(axis='x', labelbottom=False, labeltop=False, labelrotation=labelrot, labelsize=ini.fontsizelegend)
        ax11.set_ylabel('KaVA Idx\nClose station', rotation=90, labelpad=10, fontsize=ini.fontsizelegend)

        # Panel 2: kava1 + spectrogram 1
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        ax22 = ax2.twinx()
        cax2 = ax2.pcolor(tspecplt, f1plt, np.log10(spec1plt), shading='nearest', cmap=cmap)
        ax22.plot(tspecplt, kava1plt, color='k', label=f"KaVA Idx\n{ini.stationdir[0]}", linewidth=0.8)
        ax2.set_ylabel('[Hz]', rotation=90, labelpad=10, fontsize=ini.fontsizelegend)
        ax2.set_ylim(0, 65)
        yticks = np.arange(0, 63, 20)
        ax2.set_yticks(yticks)
        [a.tick_params(axis='y', rotation=labelrot, labelsize=ini.fontsizelegend) for a in [ax2, ax22]]
        ax2.tick_params(axis='x', labelbottom=False, labeltop=False, labelrotation=labelrot, labelsize=ini.fontsizelegend)
        ax22.set_ylabel('KaVA Idx\nRemote station', rotation=90, labelpad=10, fontsize=ini.fontsizelegend)

        # Panel 3: SHARK index + detections + fill_between
        ax3 = fig.add_subplot(gs[3], sharex=ax0)
        ax3.fill_between(tspecplt, 0, sharkplt.max()*1.1, where=sharkqualifierplt == 1, facecolor='tab:blue', alpha=0.4)
        ax3.plot(tspecplt, sharkplt, color='tab:blue', label='SHARK Idx')
        ax3.scatter(teventplt, eventsharkplt, color='tab:red', label='Detections')
        ax3.set_ylabel('KaVA 2 x KaVA 1\n SHARK', fontsize=ini.fontsizelegend)
        ax3.tick_params(axis='y', rotation=labelrot, labelsize=ini.fontsizelegend)
        ax3.legend(fontsize=ini.fontsizelegend, loc='upper right')
        ax3.set_ylim(0, np.max(sharkplt)*1.1)  # Set y-limits to 10% above max value

        [a.spines['top'].set_visible(False) for a in [ax1, ax11, ax2, ax22, ax3]]
        [a.legend(loc='upper right', fontsize=ini.fontsizelegend) for a in [ax11, ax22, ax3]]
        [a.grid(alpha=0.5) for a in [ax1, ax2, ax3]]

        # Shared X axis formatting
        ax3.set_xlabel('Time', fontsize=ini.fontsizelegend)
        # if ini.sectionlen == 20*60:  # 20 minutes
        #     xticks = np.array(np.arange(timeplt[0], timeplt[-1]+1, 2*60), dtype=int)
        # else:
        #     xticks = np.array(np.arange(timeplt[0], timeplt[-1], ini.sectionlen//18), dtype=int)
        # xticklabels = np.array([str(timedelta(seconds=s)) for s in np.array(xticks, dtype="float")])
        # ax3.set_xticks(xticks)
        # ax3.set_xticklabels(xticklabels, rotation=labelrot, ha='right')
        
        ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=3))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax3.tick_params(axis='x', labeltop=False, labelbottom=True, labelrotation=labelrot, labelsize=ini.fontsizelegend)

        fig.align_ylabels()
        plt.subplots_adjust(hspace=0.)

        # Save figures
        if ini.save_flag == True:
            plt.savefig(os.path.join(ini.outputdir, ini.outputlabel+'slim_'+customtime.strftime("%Y%m%d_%H%M%S")+'.png'), dpi=400, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            
        if hasattr(ini, 'start_time'):
            print('--- %.02f sec elapsed - plotting in progress ---' % (tm.time() - ini.start_time))
        else:
            print('--- Plotting still in progress ---')

def set_size(width, fraction=1):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt    = width * fraction

    # Convert from pt to inches
    inches_per_pt   = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio    = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in    = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in   = fig_width_in * golden_ratio

    fig_dim         = (fig_width_in, fig_height_in)

    return fig_dim

# def plot_overview(
#         time_waveform,
#         time_amplitudes,
#         time_spectra,
#         time_ratiopeaks,
#         time_eventtime,
#         trace1,
#         trace2,
#         ratioarray,
#         ratiopeaks,
#         kavatrigger,
#         spectrogram1,
#         spectrogram2,
#         specfrequencies1,
#         specfrequencies2,
#         kavaidx1,
#         kavaidx2,
#         kavaidxproduct,
#         kavaidxproduct2,
#         eventkava,
#         imoi, idoi, h, s,
#         outputdir,
#         start_time  = None,
#         cmapname    = ini.cmapnamespec,
#         ratiorange  = None
#         ):

#     from params_plot    import markeroffset, yaxlimshark
#     from kavutil        import myplot
#     from kav_init       import sectionlen, save_flag, shifttime, fmin, fmax, fminbbd, stationdir, outputlabel, bbd_ids, rootproject, yoi
#     if ratiorange == None:
#         from kav_init import ratiorange

#     # var_strings = ['time_waveform','time_amplitudes','time_spectra','time_ratiopeaks','time_eventtime','trace1','trace2','ratioarray','ratiotrigger','ratiopeaks','kavatrigger','spectrogram1','spectrogram2','specfrequencies1','specfrequencies2','kavaidx1','kavaidx2','kavaidxproduct','kavaidxproduct2','eventkava','imoi','idoi','h','s','yoi','start_time','cspec','outputdir']
#     ## --- Check if global_var is defined
#     # for var in var_strings:
#     #     if var in globals():
#     #         locals()[var] = globals()[var]
#     #     else:
#     #         print(var + ' is not defined. Return back to function when ' + var + ' is defined.')
#     #         break

#     # --- Date and time
#     imoi    = imoi
#     idoi    = idoi
#     h       = h
#     s       = s

#     # --- Time arrays
#     t_wf    = time_waveform         # time
#     t_ampl  = time_amplitudes       # tampl
#     t_spec  = time_spectra          # d1t
#     t_peaks = time_ratiopeaks       # tratiopeaks
#     t_event = time_eventtime

#     # --- Data arrays
#     tr1             = trace1.copy()         # tr1
#     tr2             = trace2.copy()         # tr2
#     ratio2_1        = ratioarray.copy()     # ratio2_1
#     # rtrig           = ratiotrigger.copy()   # ratiotrigger2
#     ratiopeaks      = ratiopeaks.copy()     # ratiopeaks
#     spec1           = spectrogram1.copy()   # d1spec
#     spec2           = spectrogram2.copy()   # d2spec
#     kava1           = kavaidx1.copy()       # kava1
#     kava2           = kavaidx2.copy()       # kava2
#     shark           = kavaidxproduct.copy() # kavaprod
#     shark2          = kavaidxproduct2.copy()# kavaprod2
#     ktrig           = kavatrigger.copy()    # kavatrigger
#     eventkava       = eventkava.copy()      # eventkava
#     f1              = specfrequencies1.copy() # d1f
#     f2              = specfrequencies2.copy() # d2f

#     # --- Allocations
#     shift   = shifttime / tr1.stats.delta
#     shiftsp = shifttime / (t_spec[1] - t_spec[0])
#     cspec   = plt.get_cmap(cmapname)

#     # --- Oranise indices
#     idxtr         = np.where((t_wf    >= h*60*60 + s) & (t_wf    < h*60*60 + s + sectionlen))
#     idxampl       = np.where((t_ampl  >= h*60*60 + s) & (t_ampl  < h*60*60 + s + sectionlen))
#     idxsp         = np.where((t_spec  >= h*60*60 + s) & (t_spec  < h*60*60 + s + sectionlen))
#     idxevent      = np.where((t_event >= h*60*60 + s) & (t_event < h*60*60 + s + sectionlen))
#     idxratiopeaks = np.where((t_peaks >= h*60*60 + s) & (t_peaks < h*60*60 + s + sectionlen))


#     if (idxtr[0][-1]+int(shift)) > len(tr1.data):
#         print('--- Plotting finished for date ' + '{:02d}'.format(idoi) +'/'+ '{:02d}'.format(imoi)+'/'+'20'+str(yoi)+' ---')
#     else:
#         # Create timestamps:
#         m = 0
#         if s>59: m = s//60; s = s-m*60
#         customtime     = datetime.datetime(2023, imoi, idoi, h, m, s)

#         # Definfe array slices for plotting
#         timeplt         = t_wf[         idxtr] #- h*60*60 - m*60
#         tr2plt          = tr2[          idxtr]
#         tr1plt          = tr1[          idxtr + np.int64(shift)]

#         t_amplplt       = t_ampl[       idxampl] #- h*60*60 - m*60 
#         ratio2_1plt     = ratio2_1[     idxampl]
#         # rtrigplt        = rtrig[ idxampl]

#         t_peaksplt      = t_peaks[      idxratiopeaks] #- h*60*60 - m*60
#         ratiopeaksplt   = ratiopeaks[   idxratiopeaks]
        
#         tspecplt        = t_spec[       idxsp] #- h*60*60 - m*60

#         if stationdir[1] in bbd_ids:
#             spec2plt    = spec2[fminbbd:fmax+1, idxsp[0][:]]
#             f2          = f2[fminbbd:fmax+1]
#         else:
#             spec2plt    = spec2[fmin:fmax+1,     idxsp[0][:]]
#             f2          = f2[fmin:fmax+1]
#         if stationdir[0] in bbd_ids:
#             spec1plt    = spec1[fminbbd:fmax+1, idxsp[0][:] + int(shiftsp)]
#             f1          = f1[fminbbd:fmax+1]
#         else:
#             spec1plt    = spec1[fmin:fmax+1,     idxsp[0][:] + int(shiftsp)]
#             f1          = f1[fmin:fmax+1]

#         kava2plt        = kava2[        idxsp]
#         kava1plt        = kava1[ (idxsp + np.int64(shiftsp))].flatten()

#         sharkplt        = shark[        idxsp]
#         shark2plt       = shark2[       idxsp]
#         ktrigplt        = ktrig[        idxampl]
#         # ctrigplt        = ctrig[        idxampl]

#         t_eventplt      = t_event[      idxevent] #- h*60*60 - m*60
#         eventkavaplt    = eventkava[    idxevent]
#         eventkavaplt[eventkavaplt >= yaxlimshark] = yaxlimshark - 100 # set max value 2 scale plotting
        
#         # --- Setup figure ------------------------------------------------------------------------------------------------
#         figkav, (axwave, axkavac, axkavar, axratio, axshark) = plt.subplots(5, 1, figsize=(18, 10), sharex=True)
#         figkav.suptitle('Kavachi volcanic activity analysis ' + stationdir[1] + ' / ' + stationdir[0] +' - '+ customtime.strftime("%H:%M:%S %B %d, %Y"))
        
#         # axwave.set_title('Kavachi volcanic activity analysis ' + stationdir[1] + ' / ' + stationdir[0] +' - '+ customtime.strftime("%H:%M:%S %B %d, %Y"))

#         # --- Plot waveform pannel ----------------------------------------------------------------------------------------
#         # --- plot waveforms close station ---
#         color   = 'tab:red' 
#         hand1   = axwave.plot(timeplt, tr2plt, color=color, alpha=.5, linewidth=0.4, label='WF '+ stationdir[1])
#         axwave.tick_params(axis='x', labelbottom=False, labeltop=False, labelrotation=45)
#         axwave.set_ylabel('Waveforms', rotation=90, labelpad=10, color=color)
#         axwave.tick_params(axis='y', rotation=45, labelcolor=color)
#         axwave.set_ylim(-np.amax(np.abs(tr2plt)), np.amax(np.abs(tr2plt)))
#         axwave.grid(alpha=0.3)
#         # --- plot waveforms remote station -------------------------------------------------------------------------------
#         axwavetwin = axwave.twinx()
#         color   = 'tab:blue'
#         hand2   = axwavetwin.plot(timeplt, tr1plt[0,:], color=color, alpha=.5, linewidth=0.4, label='WF '+ stationdir[0])
#         # axwavetwin.set_ylabel('WF  '+ stationdir[0]+ '\n'+ str(shifttime) +' s shifted', rotation=90, labelpad=10, color=color)
#         axwavetwin.tick_params(axis='y', rotation=45, labelcolor=color)
#         axwavetwin.set_ylim(-np.amax(np.abs(tr1plt)), np.amax(np.abs(tr1plt)))
#         myplot(axoi=axwave, handles=[hand1, hand2])


#         # --- plot amplitude ratio ------------------------------------------------------------------------------
#         # caxratio = axratio.scatter( t_amplplt, ratio2_1plt, c = 'seagreen', edgecolors='darkgreen', alpha=.5, label='ampl. ratio') # cmap = 'RdYlBu' 'RdPu' 'set1' 'nipy_spectral'
#         caxratio = axratio.plot(   t_amplplt, ratio2_1plt, color='k', alpha=.5, label='ampl. ratio')
#         axratio.fill_between(t_amplplt, 0, np.max(ratio2_1plt)+markeroffset+5, where=(ktrigplt  ==1), facecolor='tab:red',  alpha=0.3, label='dual-station trigger')
#         # axratio.fill_between(t_amplplt, 0, np.max(ratio2_1plt)+markeroffset+5, where=(rtrigplt ==1), facecolor='tab:blue', alpha=0.4, label='ratio trigger')
#         axratio.hlines(ratiorange, t_amplplt[0], timeplt[-1], colors='dimgray', linestyles='--', alpha=0.8, label='confidence range')

#         axratio.set_ylabel('Amplitude ratio\n' + stationdir[1] + ' / ' + stationdir[0], rotation=90, labelpad=10)
#         axratio.tick_params(axis='y', rotation=45)
#         axratio.scatter(t_peaksplt, ratiopeaksplt+markeroffset, s= 30, c='orange', marker='v', alpha=1)
#         # axratio.set_yticks(np.arange(0, np.amax(ratio2_1plt),1))
#         myplot(axoi=axratio)
#         # add_colorbar_outside(caxratio, axratio)
        
#         # --- Plot SHARK Index (KaVA Product) ----------------------------------------------------------------------
#         axshark.plot(tspecplt, sharkplt, color='k', alpha=0.8, linewidth=1, label='KaVA product')
#         axshark.plot(tspecplt, shark2plt, color='b', alpha=0.5, linewidth=1,linestyle='--', label='KaVA product envelope')
#         axshark.fill_between(t_amplplt, 0, np.max(sharkplt), where=(ktrigplt  ==1), facecolor='tab:red',  alpha=0.4, label='dual-station trigger')
#         # axshark.fill_between(t_amplplt, 0, np.max(sharkplt), where=(rtrigplt ==1), facecolor='tab:blue', alpha=0.4, label='ratio trigger')
#         axshark.scatter(t_eventplt, eventkavaplt, c='r', marker='v', s=100 , label='events')
#         axshark.set_ylabel('KaVA \n product', rotation=90, labelpad=10)
#         if np.max(sharkplt) > 3200:
#             axshark.set_ylim(0, 3200)
#         axshark.tick_params(axis='y', rotation=45)
#         axshark.legend(loc='upper right'); axshark.grid(alpha=0.3)
#         myplot(axoi=axshark)

#         # --- plot spectrogram close station -------------------------------------------------------------------------------
#         axkavactwin = axkavac.twinx()
#         caxkavac = axkavac.pcolor(tspecplt, f2, np.log10(spec2plt), shading='nearest', cmap=cspec)
#         axkavactwin.plot(tspecplt,       kava2plt,       color='k', alpha=0.8, linewidth=1, label='KaVA Index '+ stationdir[1])
#         axkavac.set_ylabel('[Hz]',  rotation=90, labelpad=10)
#         axkavac.set_ylim(0, fmax+8)
#         axkavac.tick_params(axis='y', rotation=45)
#         axkavactwin.set_ylabel(' KaVA Index\n Station '  + stationdir[1],  rotation=90, labelpad=10)
#         axkavactwin.tick_params(axis='y', rotation=45)
#         # add_colorbar_outside(caxkavac, axkavac) # test
#         axkavactwin.legend(loc='upper right')

#         # axkavactwin = axkavac.twinx()
#         # caxkavactwin = axkavactwin.pcolor(tspecplt, f2, np.log10(spec2plt), shading='nearest', cmap=cspec)
#         # axkavactwin.set_ylabel('[Hz]', rotation=90, labelpad=10)
#         # axkavactwin.set_ylim(0, fmax+8)
#         # axkavactwin.tick_params(axis='y', rotation=45)
#         # axkavac.plot(tspecplt,       kava2plt,       color='k', alpha=0.8, linewidth=1, label='KaVA Index '+ stationdir[1])
#         # axkavac.set_ylabel('KaVA Index', rotation=90, labelpad=10)
#         # axkavac.tick_params(axis='y', rotation=45)
#         # axkavac.legend(loc='upper right')

#         # --- Plot spectrogram remote station ------------------------------------------------------------------------------
#         axkavartwin = axkavar.twinx()
#         axkavar.pcolor(tspecplt, f1, np.log10(spec1plt), shading='nearest', cmap=cspec)
#         axkavartwin.plot(tspecplt,       kava1plt,       color='k', alpha=0.8, linewidth=1, label='KaVA Index '+ stationdir[0])
#         axkavar.set_ylabel('[Hz]', rotation=90, labelpad=10)
#         axkavar.set_ylim(0, fmax+8)
#         axkavar.tick_params(axis='y', rotation=45)
#         axkavartwin.set_ylabel('KaVA Index\n Station '  + stationdir[0], rotation=90, labelpad=10)
#         axkavartwin.tick_params(axis='y', rotation=45)
#         axkavartwin.legend(loc='upper right')

#         # axkavartwin = axkavar.twinx()
#         # caxkavartwin = axkavartwin.pcolor(tspecplt, f1, np.log10(spec1plt), shading='nearest', cmap=cspec)
#         # axkavartwin.set_ylabel('[Hz]', rotation=90, labelpad=10)
#         # axkavartwin.set_ylim(0, fmax+8)
#         # axkavartwin.tick_params(axis='y', rotation=45)
#         # axkavar.plot(tspecplt,       kava1plt,       color='k', alpha=0.8, linewidth=1, label='KaVA Index '+ stationdir[0])
#         # axkavar.set_ylabel('KaVA Index', rotation=90, labelpad=10)
#         # axkavar.tick_params(axis='y', rotation=45)
#         # axkavar.legend(loc='upper right')
#         # # ax.set_zorder(axTwin.get_zorder() + 1)
#         # axkavar.set_zorder(axkavartwin.get_zorder() + 1)


#         # --- Set customs for shared bottom x-axis --------------------------------------------------------------------------
#         axshark.set_xlabel('Time')
#         xticks = np.array(np.arange(timeplt[0], timeplt[-1], sectionlen//18), dtype=int)
#         xticklabels = np.array([str(timedelta(seconds=s)) for s in np.array(xticks, dtype="float")])

#         axshark.set_xticks(xticks, xticklabels)

#         axshark.tick_params(axis='x', labeltop=False, labelbottom=True, labelrotation=45)
        

#         figkav.align_ylabels()

#         plt.subplots_adjust(hspace=0.)


#         # Save figures
#         if save_flag == True:
#             # plt.savefig(rootproject+'/results/mixed/mixed'+'{:02d}'.format(idoi) + '{:02d}'.format(imoi) +'/mixed'+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.png', dpi=300, bbox_inches='tight') # <--- set path outputdir for smaller data set on local machine
#             plt.savefig(outputdir+'/'+outputlabel+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.png', dpi=1200, bbox_inches='tight')   # <--- set path outputdir for larger data set on external hard drive
#             # plt.savefig(outputdir+'/'+outputlabel+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.svg',format='svg', dpi=300, bbox_inches='tight')   # <--- set path outputdir for larger data set on external hard drive
#             plt.close(figkav)
#         else:
#             plt.show()
        
#         if start_time == None:
#             print('--- Plotting still in progress ---')
#         else:
#             print('--- %.02f sec elapsed - plotting in progress ---' % (tm.time() - start_time))

# def plot_magnitudes(
#         catalogname: str,
#         ax = None,
#         baz= None,
#         cut_may: bool = True,
#         plot_baz: bool = False):
#     """
#     Plot magnitudes from catalog and a polynomial fit. If choosen, cut Mai from measurements due to lack of data.

#     INPUT:
#     catalog: pd.DataFrame
#         Catalog with magnitudes and dates
#     ax: matplotlib.axes.Axes
#         Axes to plot magnitude
#     cut_may: bool
#         Cut May from measurements. Default is True
#     plot_baz: bool
#         Plot baz as colormappable. Default is False
    
#     OUTPUT:
#     fig: matplotlib.figure.Figure
#         Figure with magnitude plot
#     ax: matplotlib.axes.Axes
#         Axes with magnitude plot

#     """

#     # Read data
#     cname   = catalogname
#     df      = pd.read_pickle(cname)

#     # Separate measuring episodes "cut_may"
#     if cut_may:
#         df1 = df[df['events']  < '2023-05-01'].copy()
#         df2 = df[df['events'] >= '2023-05-31'].copy()

#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     pass
#     # return ax

# def plot_spectra(selected_events:   pd.DataFrame,
#                  stationdir:        str= 'KAV11',
#                  stationid:         str = 'c0941',
#                  upperylim:         int = 60):
#     """
#     Plot spectra of selected events

#     INPUT:
#     selected_events: pd.DataFrame
#         DataFrame with selected events
#     stationdir: str
#         Station directory
#     stationid: str
#         Station ID

#     OUTPUT:
#     ax: matplotlib.axes.Axes
#         Axes with spectra plot
#     """
#     from kav_init   import rootdata, freqbands, kav00bands, kav11bands, kav04bands
#     from kavutil_2  import retrieve_trace_data, define_ylim

#     stream = retrieve_trace_data(selected_events, stationdir, stationid)
    
#     for itrace in tqdm(range(len(stream))):
#         tr              = stream[itrace].copy()
#         tr.detrend('demean')
#         signal          = tr.data.copy()
#         signal_padded   = np.pad(signal, len(signal)//2, 'constant')    # Zero padding of trace
#         nsamples        = len(signal_padded)                            # Number of samples
#         fs              = tr.stats.sampling_rate                        # Sampling rate
#         nfft            = nsamples//2                                   # Number of samples for FFT

#         spectrum_raw    = np.fft.fft(signal_padded)
#         spectrum        = np.abs(spectrum_raw[:nfft])
#         freq            = np.fft.fftfreq(nsamples, 1/fs)[:nfft]

#         ylim_wf         = np.max(np.abs(tr.data))+np.max(np.abs(tr.data))/10
#         ylim_spec       = np.max(spectrum)+np.max(spectrum)/10
#         frequencybands  = freqbands[stationdir]

#         # Plot
#         fig, ax = plt.subplots(2, 1, figsize=(10, 10))
#         # Plot waveform
#         ax[0].plot(tr.times(), tr.data, color='tab:orange', linewidth=.75, label='Waveform')
#         ax[0].tick_params(axis='x', labelbottom=False, labeltop=True, rotation=45)
#         ax[0].set_xlabel('Time [s]') # ax[0].set_ylabel('Amplitude')
#         ax[0].set_title('Waveform and spectrum of event ' + tr.stats.starttime.strftime('%Y-%m-%d %H:%M:%S'))
#         define_ylim(ax[0], tr.data) # ax[0].set_ylim(-ylim_wf, ylim_wf); ax1.grid(); ax1.legend()

#         # Plot spectra
#         ax[1].plot(freq, spectrum, color='tab:blue', linewidth=.75, label='Spectrum')
        
#         for band in frequencybands:
#             ax[1].fill_between(freq, 0, ylim_spec, where=(freq >= band[0]) & (freq <= band[1]), color='tab:red', alpha=0.4, label= '%s-%s Hz' % (band[0], band[1]))
#         ax[1].set_xlabel('Frequency [Hz]')# ; ax[1].set_ylabel('Amplitude')
        
#         # ax2.set_title('Spectrum of event ' + tr.stats.starttime.strftime('%Y-%m-%d %H:%M:%S'))
#         ax[1].set_ylim(0, ylim_spec)
#         ax[1].set_xlim([0, upperylim]) # ; ax2.grid(); ax2.legend()
#         [axitem.set_ylabel('Amplitude') for axitem in ax]
#         [axitem.grid() for axitem in ax]
#         [axitem.legend() for axitem in ax]
#         plt.tight_layout()
        
#         # Show, save plot and return axes
#         if 'spectra.'+stationdir not in os.listdir(rootdata+'/results/spectra'):
#             os.mkdir(rootdata+'/results/spectra/spectra.'+stationdir)
#         plt.savefig(rootdata + '/results/spectra/spectra.'+stationdir+'/spectra.'+stationdir+'.'+tr.stats.starttime.strftime('%Y-%m-%d_%H-%M-%S')+'.png', dpi=300, bbox_inches='tight')
#         print('Spectrum of event ' + tr.stats.starttime.strftime('%Y-%m-%d %H:%M:%S') + 'plotted and saved.')
#         plt.close(fig)

# def plot_logNlogM(axes:     plt.Axes = None,
#                   showplt:  bool     = False):
#     """
#     Plot logN-logM diagram

#     INPUT:
#     catalog: pd.DataFrame
#         Catalog with magnitudes and dates. Is set in kav_init.py
#     ax: matplotlib.axes.Axes
#         Axes to plot logN-logM diagram

#     OUTPUT:
#     fig: matplotlib.figure.Figure
#         Figure with logN-logM diagram
#     ax: matplotlib.axes.Axes
#         Axes with logN-logM diagram
#     """
#     from kav_init import rootdata, catalog_magnitudes
#     from kavutil_2  import myplot
    
#     # assign data to variables magnitudes mlv and dates
#     catalog = pd.read_pickle(catalog_magnitudes)
#     mlv     = catalog['mlv']
#     if 'events' in catalog.columns:
#         events  = catalog['events']
#     elif 'time' in catalog.columns:
#         events  = catalog['time']
#     elif 'puretime' in catalog.columns:
#         events  = catalog['puretime']
#     elif 'date' in catalog.columns:
#         events  = catalog['date']
#     else:
#         raise ValueError('No time column found in catalog. Please check the data for datetime column.')
#     bins    = np.arange(0, np.ceil(np.max(mlv))+0.2, 0.05)
#     n, bins = np.histogram(mlv, bins=bins)
#     nacc    = np.flip(np.cumsum(n))
#     bins    = bins[1:]

#     # Setup figure and axes
#     if axes is None:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 5))

#     # Plot logN-logM diagram while logM is already considered in the data.
#     ax.plot(bins, np.log(nacc), color='k', linewidth=0.75, label='logN-logA distribution')
#     ax.fill_between(bins, 0, np.log(nacc), color='tab:blue', alpha=0.5)
#     ax.set_xlabel('Local magnitude equivalent')
#     ax.set_ylabel('Accumulated number of events of magnitude >= M \n log N')
#     ax.set_title('logN-logA diagram')
#     myplot()
    
#     if axes is not None:
#         return ax
#     else:
#         plt.savefig(rootdata + '/results/logNlogM/logNlogM.png', dpi=300, bbox_inches='tight')
#         if showplt:
#             plt.show()
#         plt.close(fig)

