# --- Import libraries ------------------------------------------------------------------------------------------------
import os
if not os.getcwd() == 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts':
    os.chdir('C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts')
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
# ----------------------------------------------------------------------------------------------------------------------

catalogfile    = 'catalogue_2.0_lb.txt'

catalog = read_catalog(rootdata+'/'+catalogfile)

enerwind = 5
distance = 8.525

# define output arrays
shift_app = [0]
v_app     = [0]
time2print = tm.time()

start_time = tm.time()
time_step  = tm.time()

firstday     = datetime.datetime(catalog.date.dt.year.values[0],
                                 catalog.date.dt.month.values[0],
                                 catalog.date.dt.day.values[0])
lastday      = datetime.datetime(catalog.date.dt.year.values[-1],
                                 catalog.date.dt.month.values[-1],
                                 catalog.date.dt.day.values[-1])

# for idx, day in catalog.groupby(catalog.date.dt.date):
for idx,day in catalog.groupby(catalog.date.dt.date):

    # print(day)
    idx     = day.index.values
    times   = day.date.dt.time.values
    yoi     = day.date.dt.year.values[0]
    moi     = day.date.dt.month.values[0]
    doi     = day.date.dt.day.values[0]
    if yoi == 2023:
        yoi = 23
    print(yoi, moi, doi)
    
    # if doi > 8:
    #     continue
    # if (moi != 3) or (doi != 21):
    #     continue
    # break

    events = (day.date.dt.hour*60*60 + day.date.dt.minute*60  +  day.date.dt.second) # in total seconds of day
    events = events.to_numpy()
    # evidx = np.arange(len(events))

    # print(events)

    # --- Import data --------------------------------------------------------------------------------------------------
    st1, _  = get_data(year=yoi, month=moi, day=doi, rootdata=rootdata, stationdir=stationdir[0], stationid=stationid[0])
    st2, _  = get_data(year=yoi, month=moi, day=doi, rootdata=rootdata, stationdir=stationdir[1], stationid=stationid[1])
    x1      = st1[0].copy()
    x2      = st2[0].copy()

    # print("x1 meta:", x1.stats)
    # print("x2 meta:", x2.stats)

    if len(x1.data) != len(x2.data):
        print('Data length not equal. Skip this day.')
        continue

    
    x1end   = x1.stats.endtime
    x1end   = (x1end.hour*60*60 + x1end.minute*60  +  x1end.second) # in total seconds of day

    time        = x2.times(reftime = x2.stats.starttime)

    # kava1, d1t, dtspec, d1spec, d1f, cax = compute_kava(x1, frequency_bands=freqbands[stationdir[0]], nfft=int(x1.stats.sampling_rate), noverlap=int(x1.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)
    # kava2, d2t, dtspec, d2spec, d2f, cax = compute_kava(x2, frequency_bands=freqbands[stationdir[1]], nfft=int(x2.stats.sampling_rate), noverlap=int(x2.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)
    kava1, d1t, dtspec, d1spec, d1f, cax = compute_kava(x1, station= stationdir[0], frequency_bands=freqbands, nfft=int(x1.stats.sampling_rate), noverlap=int(x1.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)
    kava2, d2t, dtspec, d2spec, d2f, cax = compute_kava(x2, station= stationdir[1], frequency_bands=freqbands, nfft=int(x2.stats.sampling_rate), noverlap=int(x2.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)


    shifts          = np.arange(0.25, 4.25, 0.25)
    add_shift_app   = np.zeros(len(events))

    for iev in range(len(idx)):

        iwints      = np.where((time >= (events[iev] - enerwind)) & (time <= (events[iev] + 3*enerwind)))
        iwinkava    = np.where((d2t >= (events[iev] - enerwind)) & (d2t <= (events[iev] + 3*enerwind)))
        iwints      = iwints[0]
        iwinkava    = iwinkava[0]

        wintime     = time[iwints]
        wints       = x2.data[iwints]
        winkava2    = kava2[iwinkava]
        
        sharkmax    = np.zeros_like(shifts)
        
        for ish in np.arange(len(shifts)):
            iwinkava_shift  = np.where((d1t >= events[iev]-enerwind+shifts[ish]) & (d1t <= events[iev]+3*enerwind+shifts[ish]))[0]
            winkava1        = kava1[ iwinkava_shift]
            # print( evidx[iev], ish ,len(winkava1), len(winkava2))
            if len(winkava1) != len(winkava2):
                print("break loop here")
                # break

            if len(winkava1) != len(winkava2):
                if abs(len(winkava1) - len(winkava2)) > 2:
                    winkava1 = np.zeros_like(winkava2)
                elif (abs(len(winkava1) - len(winkava2)) <= 2) & (len(winkava1) > len(winkava2)):
                    print("c1")
                    winkava1 = winkava1[:len(winkava2)]
                elif (abs(len(winkava1) - len(winkava2)) <= 2) & (len(winkava1) < len(winkava2)):
                    print("c2")
                    winkava2 = winkava2[:len(winkava1)]
                    print("new length winkava1 / 2", len(winkava1), len(winkava2))

            winshark        = winkava2 * winkava1

            sharkmax[ish]   = np.amax(winshark)

        bestshift           = np.where(sharkmax == np.amax(sharkmax))[0][0]
        add_shift_app[iev]  = shifts[bestshift]
            
    shift_app   = np.concatenate((shift_app, add_shift_app))

    data2plt = pd.DataFrame(data = shift_app[1:], columns=['delaytimes'], dtype="float")
    data2plt.plot.hist(column="delaytimes", bins=np.arange(0.25/2,4+0.25/2,0.25), grid=True)
    plt.savefig(rootproject+'/results/vapp/steps/delaytime_steps_.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('--- runtime  %.02f  ---' % (tm.time() - start_time))
    print('--- time for step  %.02f  ---' % (tm.time() - time_step))
    time_step = tm.time()  
    
    
shift_app   = shift_app[1:]

# print(shift_app)
v_app       = distance / shift_app

figvapp, axvapp = plt.subplots(figsize=(8, 6))
figvapp.suptitle('Distribution apparent delaytimes \n KAV11-KAV04 distance 8.525 km\n from '
                 + firstday.strftime("%b %d %Y") +' to ' + lastday.strftime("%b %d %Y") ,
                 fontsize=12, fontweight='bold')

num_bins = np.arange(0, 4.25, .25)
binslabel = ["%.2f" % bins for bins in num_bins[1:]]

# axvapp.hist(v_app, bins=num_bins, edgecolor='black', align='left', label='resolution 0.25 s')
counts, edges, bars = axvapp.hist(shift_app, bins=num_bins, edgecolor='black', align='left', label='Delaytimes')

# axvapp.set_yticks(np.arange(0,5.5,1))
axvapp.set_xticks(num_bins[1:])
axvapp.set_xticklabels(binslabel)
# axvapp.set_xticklabels(['0.25\n34.1','0.5\n17.1','0.75\n11.4','1\n8.5','1.25\n6.8','1.5\n5.68','1.75\n4.9','2\n4.3','2.25\n3.8','2.5\n3.4','2.75\n3.1','3\n2.8','3.25\n2.6','3.5\n2.4','3.75\n2.3','4\n2.1'])
axvapp.tick_params(axis='x', labelrotation=45)
myplot(axoi=axvapp)
axvapp.yaxis.set_major_locator(MaxNLocator(integer=True))
axvapp.bar_label(bars)

# Add labels and title to the histogram plot
axvapp.set_xlabel('Delaytimes [sec]')
axvapp.set_ylabel('Counts')

# plt.show()
plt.savefig(rootproject+'/results/vapp/v_app_hist_new.png', dpi=300, bbox_inches='tight')
plt.close()

catalog_edit = catalog
print(catalog_edit)

if len(catalog_edit) != len(shift_app):
    print("length of catalog and shift_app not equal")
    print(len(catalog_edit), len(shift_app))
    print(catalog_edit)
    print(shift_app)
    breakpoint()

catalog_edit["delaytime"] = shift_app
with open(rootdata+'/'+newcataoutput, 'a') as f:
    catalog_edit_str = catalog_edit.to_string(header=True, index=False)
    f.write(catalog_edit_str)
    
