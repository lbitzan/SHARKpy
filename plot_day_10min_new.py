# --- Import libraries ------------------------------------------------------------------------------------------------
import os
if not os.getcwd() == 'C:/Users/Ludwig/projectsT14/GitHub/KavScripts':
    os.chdir('C:/Users/Ludwig/projectsT14/GitHub/KavScripts')
from    obspy.core           import  read, UTCDateTime, Stream, Trace
from    obspy.imaging        import  spectrogram
import  numpy                as      np
import  os, datetime, glob
import  matplotlib.pyplot    as      plt
import  matplotlib.colorbar  as      cbar
from    scipy                import  io, signal
from    scipy.fft            import  fftshift
import  pandas               as      pd
import  time                 as      tm
from matplotlib.ticker import MaxNLocator

from kavutil import *
from kav_init import *
from plutil import plot_spectra
from comp_rsam import comp_rsam, comp_rsam_v2, rsam_evol, rsam_compute
import glob

# ----------------------------------------------------------------------------------------------------------------------
flag_plot_spectra       = False
flag_plot_wf_spectro    = False
flag_plot_dayplot       = False
flag_plot_rsam          = False
flag_plot_activity_freq = False
flag_plot_rsam_all      = True
# ----------------------------------------------------------------------------------------------------------------------
setdpi  = 300
lw      = 1.
st04 = read(rootdata+'/KAV04/c0939230405000000.pri0')
st11 = read(rootdata+'/KAV11/c0941230405000000.pri0')
st00 = read(rootdata+'/KAV00/c0bdd230405000000.pri0')


if flag_plot_dayplot:
    fig4 = plt.figure(figsize=(10,10))
    st04.plot(fig=fig4, type="dayplot", dpi=setdpi, interval = 30, color="blue",title="KAV04 - 05/04/2023", show_y_UTC_label=False, linewidth=lw)
    plt.savefig(rootdata+"/results/thesis/dayplotKAV04.png",format="png", dpi =setdpi)

    fig11 = plt.figure(figsize=(10,10))
    st11.plot(fig=fig11, type="dayplot", dpi=setdpi, interval = 30, color="blue",title="KAV11 - 05/04/2023", show_y_UTC_label=False, linewidth=lw)
    plt.savefig(rootdata+"/results/thesis/dayplotKAV11.png", format="png", dpi=setdpi)

    fig00 = plt.figure(figsize=(10,10))
    st00.plot(fig=fig00, type="dayplot", dpi=setdpi, interval = 30, color="blue",title="KAV00 - 05/04/2023", show_y_UTC_label=False, linewidth=lw)
    plt.savefig(rootdata+"/results/thesis/dayplotKAV00.png", format="png", dpi=setdpi)


if flag_plot_wf_spectro:
    tr04 = st04[0].copy()
    tr11 = st11[0].copy()
    tr00 = st00[0].copy()
    tr04.detrend('demean'); tr04.stats.station='KAV04'; tr04.stats.channel=''; d04 = tr04.copy()
    tr11.detrend('demean'); tr11.stats.station='KAV11'; tr11.stats.channel=''; d11 = tr11.copy()
    tr00.detrend('demean'); tr00.stats.station='KAV00'; tr00.stats.channel=''; d00 = tr00.copy()

    time = tr04.times(reftime=tr04.stats.starttime)
    sr = tr04.stats.sampling_rate

    dspec04, dfreq04, dtime04, dcax = plt.specgram(d04.data, NFFT=int(sr), Fs=int(sr), noverlap=int(sr*.75), scale_by_freq=True, cmap='viridis'); plt.close()
    dspec11, dfreq11, dtime11, dcax = plt.specgram(d11.data, NFFT=int(sr), Fs=int(sr), noverlap=int(sr*.75), scale_by_freq=True, cmap='viridis'); plt.close()
    dspec00, dfreq00, dtime00, dcax = plt.specgram(d00.data, NFFT=int(sr), Fs=int(sr), noverlap=int(sr*.75), scale_by_freq=True, cmap='viridis'); plt.close()

    itr         = np.where((time    >= 60*45) & (time    <= 60*55))
    isp         = np.where((dtime04 >= 60*45) & (dtime04 <= 60*55))
    xticks      = np.array([45, 47, 49, 51, 53, 55]) * 60
    test        = pd.date_range("00:45:00", "00:55:00", freq="2min")
    xtickstr    = test.strftime('%X') 

    f04, axs = plt.subplots(2,1, figsize=(10,5), sharex=True)
    plt.subplots_adjust(hspace=0.)
    ax0 = axs[0]
    ax0.plot(time[itr]  ,tr04.data[itr], color='tab:blue', linewidth=.5, label='KAV04 WF')
    define_ylim(ax0, tr04.data[itr])
    ax0.grid()
    ax0.legend()
    ax0.set_ylabel('Amplitude')
    ax1 = axs[1]
    ax1.pcolor(dtime11[isp], dfreq11[3:60],np.log10(dspec04[3:60, isp[0][:]]), cmap='jet')
    ax1.set_ylabel('[Hz]')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtickstr)
    f04.align_ylabels(axs[:])
    plt.savefig(rootdata+'/results/thesis/KAV04_10min_jet_png_new.png', format='png',dpi=setdpi)
    plt.show()

    f11, axs = plt.subplots(2,1,figsize=(10,5),sharex=True)
    plt.subplots_adjust(hspace=0.)
    ax0 = axs[0]
    ax0.plot(time[itr]  ,tr11.data[itr], color='tab:blue', linewidth=.5, label='KAV11 WF')
    define_ylim(ax0, tr11.data[itr])
    ax0.grid()
    ax0.legend()
    ax0.set_ylabel('Amplitude')
    ax1 = axs[1]
    ax1.pcolor(dtime11[isp], dfreq11[3:60],np.log10(dspec11[3:60, isp[0][:]]), cmap='jet')
    ax1.set_ylabel('[Hz]')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtickstr)
    f11.align_ylabels(axs[:])
    plt.savefig(rootdata+'/results/thesis/KAV11_10min_jet_png_new.png', format='png',dpi=setdpi)
    plt.show()

    f00, axs = plt.subplots(2,1,figsize=(10,5),sharex=True)
    plt.subplots_adjust(hspace=0.)
    axs[0].plot(time[itr]  ,tr00.data[itr], color='tab:blue', linewidth=.5, label='KAV00 WF')
    define_ylim(axs[0], tr00.data[itr])
    axs[0].set_ylabel('Amplitude')
    axs[0].grid()
    axs[0].legend()
    axs[1].pcolor(dtime00[isp], dfreq00[3:60],np.log10(dspec00[3:60, isp[0][:]]), cmap='jet')
    axs[1].set_ylabel('[Hz]')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xtickstr)
    f00.align_ylabels(axs[:])
    plt.savefig(rootdata+'/results/thesis/KAV00_10min_jet_png_new.png', format='png',dpi=setdpi)
    plt.show()

if flag_plot_spectra:
    # --- Plotting the spectrogram of the three stations
    cata = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.20.0s.mlv.pkl')
    # events = select_events_v2(cata, '2023-04-05T00:45:00', '2023-04-05T00:55:00', asstring=True)
    # events = select_events_v2(cata, '2023-04-24T00:00:00', '2023-04-24T02:00:00', asstring=True)
    # events = select_events_v2(cata, '2023-06-02T00:00:00', '2023-06-02T01:00:00', asstring=True)
    events = select_events_v2(cata, '2023-02-19T00:00:00', '2023-02-19T06:00:00', asstring=True)
    breakpoint()
    print(events)

    stationsname = ['KAV04', 'KAV11', 'KAV00']
    stationsident = ['c0939', 'c0941', 'c0bdd']
    [plot_spectra(events, stationdir=stationsname[i], stationid=stationsident[i], upperylim=50) for i in range(len(stationsname))]

if flag_plot_rsam:
    
    # Season 1:
    # rsam_df = rsam_compute('2023-02-01','2023-08-01','KAV11', 'c0941')
    rsam_df = pd.read_pickle(rootdata+'/results/rsam_KAV11.pkl')
    catalog = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.20.0s.mlv.pkl')

    for ep in range(len(seasons)):
        cata = catalog[(catalog['date'] >= seasons[ep][0]) & (catalog['date'] <= seasons[ep][1])]
        rsam_df = pd.read_pickle(rootdata+'/results/rsam_KAV11.pkl')
        rsam_df = rsam_df[(rsam_df['time'] >= seasons[ep][0]) & (rsam_df['time'] <= seasons[ep][1])]

        fig, ax = plt.subplots(3,1,figsize=(10,10),sharex=True)
        fig.suptitle('From' + str(seasons[ep][0])+' to '+str(seasons[ep][1])+'\n Geophone setup and 20 s triggerrestingtime \nRSAM and KaVA Idx event count')
        # Plot daily event count on ax[0]
        daily_event_count = cata['date'].dt.floor('d').value_counts().sort_index()
        ax[0].grid(True)
        ax[0].bar(daily_event_count.index, daily_event_count.values, color='tab:blue', label='Daily Event Count')
        ax[0].set_ylabel('Counts')
        ax[0].legend()
        

        # Plot RSAM on ax[1]
        ax[1].plot(rsam_df['time'], rsam_df['rsam'], color='tab:orange', label='RSAM')
        ax[1].set_ylabel('RSAM'); ax[1].legend(); ax[1].grid(); ax[1].set_ylim(0, 2*rsam_df['rsam'].sort_values().iloc[-len(rsam_df)//1000]*1.1)
        ax[1].grid()
        ax[1].legend()
        

        # Plot magnitudes on ax[2].
        cata['date'] = cata['date'].dt.floor('d')
        daily_mean_magnitude = cata.groupby('date')['mlv'].mean()
        ax[2].scatter(cata['date'], cata['mlv'], color='#4daf4a', alpha=0.1, label='Magnitude')
        ax[2].plot(daily_mean_magnitude.index, daily_mean_magnitude.values, color='#f781bf', label='Daily Mean Magnitude')
        ax[2].set_ylabel('Magnitude')
        ax[2].grid()
        ax[2].legend()
        ax[2].set_xlabel('Time')

        fig.align_ylabels(ax[:])
        plt.tight_layout()
        plt.savefig(rootdata+'/results/thesis/rsam_3D.20sec.'+str(seasons[ep][0])+'_'+str(seasons[ep][1])+'.png', format='png', dpi=setdpi)
    
    # rsam_df_bbd = rsam_compute('2023-02-01','2023-12-01','KAV00', 'c0bdd')
    rsam_df_bbd = pd.read_pickle(rootdata+'/results/rsam_KAV00.pkl')
    seasons = [['2023-02-01','2023-05-02'],
               ['2023-05-23','2023-08-01'],
               ['2023-09-14','2023-11-13']]
    
    catalog_bbd = pd.read_pickle(rootdata+'/results/cata_9.0_bbd.tbreaktest.lb/cata_9.0_bbd.tbreaktest.lb.20.0s.mlv.pkl')
    breakpoint()
    for ep in range(len(seasons)):
        cata = catalog_bbd[(catalog_bbd['date'] >= seasons[ep][0]) & (catalog_bbd['date'] <= seasons[ep][1])]
        rsam_df_bbd = pd.read_pickle(rootdata+'/results/rsam_KAV00.pkl')
        rsam_df_bbd = rsam_df_bbd[(rsam_df_bbd['time'] >= seasons[ep][0]) & (rsam_df_bbd['time'] <= seasons[ep][1])]

        fig, ax = plt.subplots(3,1,figsize=(10,10),sharex=True)
        fig.suptitle('From' + str(seasons[ep][0])+' to '+str(seasons[ep][1])+'\n BBD setup and 20 s triggerrestingtime \nRSAM and KaVA Idx event count')
        # Plot daily event count on ax[0]
        daily_event_count = cata['date'].dt.floor('d').value_counts().sort_index()
        ax[0].grid(True)
        ax[0].bar(daily_event_count.index, daily_event_count.values, color='tab:blue', label='Daily Event Count')
        ax[0].set_ylabel('Counts')
        ax[0].legend()
        

        # Plot RSAM on ax[1]
        ax[1].plot(rsam_df_bbd['time'], rsam_df_bbd['rsam'], color='tab:orange', label='RSAM')
        ax[1].set_ylabel('RSAM'); ax[1].legend(); ax[1].grid(); ax[1].set_ylim(0, 2*rsam_df_bbd['rsam'].sort_values().iloc[-len(rsam_df_bbd)//1000]*1.1)
        ax[1].grid()
        ax[1].legend()

        # Plot magnitudes on ax[2].
        cata['date'] = cata['date'].dt.floor('d')
        daily_mean_magnitude = cata.groupby('date')['mlv'].mean()
        ax[2].scatter(cata['date'], cata['mlv'], color='#4daf4a', alpha=0.1, label='Magnitude')
        ax[2].plot(daily_mean_magnitude.index, daily_mean_magnitude.values, color='#f781bf', label='Daily Mean Magnitude')
        ax[2].set_ylabel('Magnitude')
        ax[2].legend()
        ax[2].grid()
        ax[2].set_xlabel('Time')

        fig.align_ylabels(ax[:])
        plt.tight_layout()
        plt.savefig(rootdata+'/results/thesis/rsam_bbd.20sec.'+str(seasons[ep][0])+'_'+str(seasons[ep][1])+'.png', format='png', dpi=setdpi)
        plt.show()




if flag_plot_activity_freq:
    catalog = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.20.0s.mlv.pkl')
    tbreak = 20
    event_tdiff_distr(catalog,'KAV11',binmin=tbreak, nbins=100, triggerrestingtime=tbreak)
    
    catalog = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.0.5s.mlv.pkl')
    tbreak = 0.5
    event_tdiff_distr(catalog,'KAV11',binmin=tbreak, nbins=100, triggerrestingtime=tbreak)

    catalog = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.10.0s.mlv.pkl')
    tbreak = 10
    event_tdiff_distr(catalog,'KAV11',binmin=tbreak, nbins=100, triggerrestingtime=tbreak)

    catalog = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.30.0s.mlv.pkl')
    tbreak = 30
    event_tdiff_distr(catalog,'KAV11',binmin=tbreak, nbins=100, triggerrestingtime=tbreak)

    catalog = pd.read_pickle(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/cata_9.0_3D.tbreaktest.lb.40.0s.mlv.pkl')
    tbreak = 40
    event_tdiff_distr(catalog,'KAV11',binmin=tbreak, nbins=100, triggerrestingtime=tbreak)


if flag_plot_rsam_all:
    # do the same as in flag_plot_rsam, but for all *mlv.pkl files in the 3D directory.
    # Plot data for the full length of the catalog together.

    rsam_files = [rootdata+'/results/rsam_KAV11.pkl', rootdata+'/results/rsam_KAV00.pkl']
    catalog_files = glob.glob(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/*.mlv.pkl')
    rsam_files = [rootdata+'/results/rsam_KAV11.pkl', rootdata+'/results/rsam_KAV00.pkl']
    station_ids = ['KAV11', 'KAV00']

    for rsam_file, station_id in zip(rsam_files, station_ids):
        for catalog_file in catalog_files:
            rsam_df = pd.read_pickle(rsam_file)
            catalog = pd.read_pickle(catalog_file)

            fig, ax = plt.subplots(3,1,figsize=(10,10),sharex=True)
            fig.suptitle(f'RSAM and Event Count for {station_id} - {os.path.basename(catalog_file)}')

            # Plot daily event count on ax[0]
            daily_event_count = catalog['date'].dt.floor('d').value_counts().sort_index()
            ax[0].grid(True)
            ax[0].bar(daily_event_count.index, daily_event_count.values, color='tab:blue', label='Daily Event Count')
            ax[0].set_ylabel('Counts')
            ax[0].legend()

            # Plot RSAM on ax[1]
            ax[1].plot(rsam_df['time'], rsam_df['rsam'], color='tab:orange', label='RSAM')
            ax[1].set_ylabel('RSAM')
            ax[1].legend()
            ax[1].grid()
            ax[1].set_ylim(0, 2*rsam_df['rsam'].sort_values().iloc[-len(rsam_df)//1000]*1.1)

            # Plot magnitudes on ax[2]
            catalog['date'] = catalog['date'].dt.floor('d')
            daily_mean_magnitude = catalog.groupby('date')['mlv'].mean()
            ax[2].scatter(catalog['date'], catalog['mlv'], color='#4daf4a', alpha=0.1, label='Magnitude')
            ax[2].plot(daily_mean_magnitude.index, daily_mean_magnitude.values, color='#f781bf', label='Daily Mean Magnitude')
            ax[2].set_ylabel('Magnitude')
            ax[2].legend()
            ax[2].grid()
            ax[2].set_xlabel('Time')

            fig.align_ylabels(ax[:])
            plt.tight_layout()
            plt.savefig(rootdata+f'/results/thesis/rsam_all_{station_id}_{os.path.basename(catalog_file)}.png', format='png', dpi=setdpi)
            plt.close()