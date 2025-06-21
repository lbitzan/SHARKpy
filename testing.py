import os
if not os.getcwd() == 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts':
    os.chdir('C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts')

from    obspy.core           import  read, UTCDateTime, Stream, Trace
from obspy import read, read_inventory
from    obspy.imaging        import  spectrogram
import  numpy                as      np
import  matplotlib.pyplot    as      plt
from    scipy.fft            import  fftshift
import  pandas               as      pd
from kavutil       import * # rolling_stats, myplot, compute_kava, get_data, compute_banded_ratio
from kav_init      import * # rootproject, rootcode, rootdata, rootouts, cataloguefile, newcataoutput, plot_flag, save_flag, write_flag, month_flag, hour_flag, flag_bandedratio, outputlabel, yoi, moi, doi, hoi, stationdir, stationid, zstr, estr, wstr, shifttime, ratiorange, fmin, fmax, kav04bands, kav11bands, kav00bands, freqbands
import seispy as sp
from obspy.signal.invsim import simulate_seismometer
from colorbar_fct   import add_colorbar_outside
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------------------

respbbd     = '/resp/Trillium.resp'
resp45      = '/resp/PE-6B.resp'

moi                 = [7]                       # months of interest
doi                 = [3]                       # days of interest
hoi                 = [0]                       # hours of interest

st45,_      = get_data(year=23,rootdata=rootdata, month=moi[0], day=doi[0], stationid='c0941')
stbbd,_     = get_data(year=23,rootdata=rootdata, month=moi[0], day=doi[0], stationdir="KAV00", stationid='c0bdd')
st04,_      = get_data(year=23,rootdata=rootdata, month=moi[0], day=doi[0], stationdir='KAV04' ,stationid='c0939')

st45[0].stats.network='KAVX';   st45[0].stats.location='00';    st45[0].stats.station='YY'
stbbd[0].stats.network='KAVB';  stbbd[0].stats.location='00';   stbbd[0].stats.station='YY'
st04[0].stats.network='KAVX';   st04[0].stats.location='00';    st04[0].stats.station='YY'

tr45        = st45[0].copy()
trbbd       = stbbd[0].copy()
tr04        = st04[0].copy()

# inv45     = read_inventory(rootdata+resp45, format='RESP' ,level='network')
# # inv45       = read_inventory(rootdata+'/PE-6B.resp', format='RESP' ,level='response')
# invbbd      = read_inventory(rootdata+respbbd, format='RESP', level='network')

# tr45rmv     = tr45.copy(); tr45rmv.detrend('demean')
# trbbdrmv    = trbbd.copy(); trbbdrmv.detrend('demean')

# tr45arr     = simulate_seismometer(tr45rmv.data, tr45rmv.stats.sampling_rate,paz_remove=paz45, taper=False)
# tr45arrtr   = Trace(data=tr45arr, header=tr45rmv.stats)

# trbbdarr    = simulate_seismometer(trbbdrmv.data, trbbdrmv.stats.sampling_rate,paz_remove=pazbbd, taper=False)
# trbbdarrtr  = Trace(data=trbbdarr, header=trbbdrmv.stats)

# tr45rmv.remove_response(inventory=inv45, output="DISP", plot=True)
# trbbdrmv.remove_response(inventory=invbbd)

# tr45rmv.plot(starttime=UTCDateTime(2023,4,5,0,48,0), endtime=UTCDateTime(2023,4,5,0,52,0))
# tr45arrtr.plot(starttime=UTCDateTime(2023,4,5,0,48,0), endtime=UTCDateTime(2023,4,5,0,52,10))
# trbbdrmv.plot(starttime=UTCDateTime(2023,4,5,0,48,0), endtime=UTCDateTime(2023,4,5,0,52,0))
# trbbdarrtr.plot(starttime=UTCDateTime(2023,4,5,0,48,0), endtime=UTCDateTime(2023,4,5,0,52,10))

# fig, [ax1, ax2, ax3, ax4] = plt.subplots(4,1, figsize=(10,10), sharex=True)
# tr45rmv.plot()
# tr45arrtr.plot()
# trbbdrmv.plot()
# trbbdarrtr.plot()
# plt.show()

# --- Compute KAVA ---
x11 = tr45.copy()
x04 = tr04.copy()
x00 = trbbd.copy()

kava11, d11t, dtspec, d11spec, d11f, cax = compute_kava(x11, station= 'KAV11', frequency_bands=freqbands, nfft=int(x11.stats.sampling_rate), noverlap=int(x11.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)
kava04, d04t, dtspec, d04spec, d04f, cax = compute_kava(x04, station= 'KAV04', frequency_bands=freqbands, nfft=int(x04.stats.sampling_rate), noverlap=int(x04.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)
kava00, d00t, dtspec, d00spec, d00f, cax = compute_kava(x00, station= 'KAV00', frequency_bands=freqbands, nfft=int(x00.stats.sampling_rate), noverlap=int(x00.stats.sampling_rate*.75), fmin=fmin, fmax=fmax)


figcomp, [ax11, ax04, ax00] = plt.subplots(3,1, figsize=(10,10), sharex=True)
imin    = np.amax(np.where(d11t <= 90*60))
imax    = np.amin(np.where(d11t >= 95*60))
# ilabel  = np.amax(np.where(d11t <= 3018))
ilabel  = np.amax(np.where(d11t <= 5470))

labelnum11 = np.sum(d11spec[4:60,ilabel])
labelnum04 = np.sum(d04spec[4:60,ilabel])
labelnum00 = np.sum(d00spec[1:60,ilabel])

label11mean = np.mean(np.amax(d11spec[4:60,:], axis=0))
label04mean = np.mean(np.amax(d04spec[4:60,:], axis=0))
label00mean = np.mean(np.amax(d00spec[1:60,:], axis=0))

ax11.set_title( UTCDateTime(x11.stats.starttime).strftime('%b %Y - %d') +' \n KAV11')
cax11= ax11.pcolor(d11t[imin:imax], d11f, np.log10(d11spec[:,imin:imax]), cmap='viridis',
                   label='%.2f mean' % label11mean)
add_colorbar_outside(cax11, ax11)
ax11.axvline(x=5470, color='r', linestyle='--',label='%.2f' % labelnum11); ax11.legend()

ax04.set_title('KAV04')
cax04 = ax04.pcolor(d04t[imin:imax], d04f, np.log10(d04spec[:,imin:imax]), cmap='viridis',
                    label='%.2f mean' % label04mean)
add_colorbar_outside(cax04, ax04)
ax04.axvline(x=5470, color='r', linestyle='--',label='%.2f' % labelnum04); ax04.legend()

ax00.set_title('KAV00')
cax00 = ax00.pcolor(d00t[imin:imax], d00f, np.log10(d00spec[:,imin:imax]), cmap='viridis',
                    label='%.2f mean' % label00mean)
add_colorbar_outside(cax00, ax00)
ax00.axvline(x=5470, color='r', linestyle='--',label='%.2f' % labelnum00); ax00.legend()
ax00.set_xlabel('Time (s)')

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
'''
-Talk to Ayoub. Maybe use obspy.simulate ...
-Check the response removal


- Check for regular spectral Amax to add in denominator for "water_level"
- Check for IR data of region

'''