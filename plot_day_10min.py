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
st04 = read(rootdata+'/KAV04/c0939230405000000.pri0')
st11 = read(rootdata+'/KAV11/c0941230405000000.pri0')

fig4 = plt.figure(figsize=(10,10))
st04.plot(fig=fig4, type="dayplot", dpi=1200, interval = 30, color="blue",title="KAV04 - 05/04/2023", show_y_UTC_label=False, linewidth=1.)
plt.savefig(rootdata+"/results/poster/dayplotKAV04.svg",format="svg", dpi =1200)


fig11 = plt.figure(figsize=(10,10))
st11.plot(fig=fig11, type="dayplot", dpi=1200, interval = 30, color="blue",title="KAV11 - 05/04/2023", show_y_UTC_label=False, linewidth=1.)
plt.savefig(rootdata+"/results/poster/dayplotKAV11.svg", format="svg", dpi=1200)


tr04 = st04[0].copy()
tr11 = st11[0].copy()
tr04.detrend('demean'); tr04.stats.station='KAV04'; tr04.stats.channel=''; d04 = tr04.copy()
tr11.detrend('demean'); tr11.stats.station='KAV11'; tr11.stats.channel=''; d11 = tr11.copy()

time = tr04.times(reftime=tr04.stats.starttime)
sr = tr04.stats.sampling_rate


dspec04, dfreq04, dtime04, dcax = plt.specgram(d04.data, NFFT=int(sr), Fs=int(sr), noverlap=int(sr*.75), scale_by_freq=True, cmap='viridis')
plt.close()
dspec11, dfreq11, dtime11, dcax = plt.specgram(d11.data, NFFT=int(sr), Fs=int(sr), noverlap=int(sr*.75), scale_by_freq=True, cmap='viridis')
plt.close()

itr = np.where((time    >= 60*45) & (time    <= 60*55))
isp = np.where((dtime11 >= 60*45) & (dtime11 <= 60*55))

xticks      = np.array([45, 47, 49, 51, 53, 55]) * 60
test        = pd.date_range("00:45:00", "00:55:00", freq="2min")
xtickstr    = test.strftime('%X') 

f04, (ax0, ax1) = plt.subplots(2,1,figsize=(10,5),sharex=True)
ax0.plot(time[itr]  ,tr04.data[itr], color='tab:blue', linewidth=.5, label='KAV04 WF')
ax0.set_ylim(-np.amax(np.abs(tr04.data[itr])), np.amax(np.abs(tr04.data[itr])))
myplot(axoi=ax0)
ax0.set_ylabel('Amplitude')
ax1.pcolor(dtime11[isp], dfreq11[3:60],np.log10(dspec04[3:60, isp[0][:]]), cmap='jet')
ax1.set_ylabel('[Hz]')
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtickstr)
plt.subplots_adjust(hspace=0.)
plt.savefig(rootdata+'/results/poster/KAV04_10min_jet_png_new.png', format='png',dpi=1200)

f11, (ax0, ax1) = plt.subplots(2,1,figsize=(10,5),sharex=True)
ax0.plot(time[itr]  ,tr11.data[itr], color='tab:blue', linewidth=.5, label='KAV11 WF')
ax0.set_ylim(-np.amax(np.abs(tr11.data[itr])), np.amax(np.abs(tr11.data[itr])))
myplot(axoi=ax0)
ax0.set_ylabel('Amplitude')
ax1.pcolor(dtime11[isp], dfreq11[3:60],np.log10(dspec11[3:60, isp[0][:]]), cmap='jet')
ax1.set_ylabel('[Hz]')
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtickstr)
plt.subplots_adjust(hspace=0.)
plt.savefig(rootdata+'/results/poster/KAV11_10min_jet_png_new.png', format='png',dpi=1200)



