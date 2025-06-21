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

# Import catalog
catalogfile = 'catalog_times_0.1_lb.txt'
headers     = ['date','ratio','shark','delays']
dtypes      = [str, float, float, float]
catalog     = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\s+', names=headers) #parse_dates=['date'])
catalog['index2'] = np.arange(len(catalog))

# Convert date to epoches
totsec              = pd.to_datetime(catalog.date)
catalog['epoches']  = (totsec - datetime.datetime(2023,2,7)).dt.total_seconds()

# Compute apparent velocity
distance    = 8.525
vapp        = distance / catalog.delays.values

# Group by day
puredate = pd.to_datetime(catalog.index + ' ' + catalog.date, format='%Y-%m-%d %H:%M:%S')
df      = pd.DataFrame({'Day': pd.to_datetime(catalog.index), 'Date': puredate}).set_index(np.arange(len(catalog)))
counts  = df.groupby(['Day']).size()
dfnew   = pd.DataFrame({'day': counts.index, 'counts': counts.values})

fqcnts  = np.fft.fft(dfnew.counts)
fs      = 1/(86400) # 1 day in seconds
p2      = abs(fqcnts)/len(fqcnts)
p1      = p2[0:len(fqcnts)//2]
p1[1:-1]  = 2*p1[1:-1]

f = fs/len(fqcnts)*(np.arange(len(fqcnts)//2))


fig, [ax1,ax2] = plt.subplots(2,1,figsize=(10,10))
# ax0.plot(dfnew.day, dfnew.counts, color='tab:blue', marker='o', alpha=0.5, label='Counts per day')
ax1.bar(dfnew.day, dfnew.counts, color='orange', alpha=0.5, label='Histogram')
ax1.plot(dfnew.day, dfnew.counts, color='tab:blue', alpha=0.5, label='Counts per day')
ax1.set_xlabel('Time [Days]'); ax1.set_ylabel('Counts'); ax1.set_title('Analysis of counts per day')
myplot(axoi=ax1)
# ax2.plot(np.fft.fftfreq(len(fqcnts), fs), np.abs(fqcnts), color='tab:red', label='FFT of counts')
# ax2.plot(fs/len(fqcnts)*np.arange(len(fqcnts)) , np.abs(fqcnts), color='tab:red', label='FFT of counts')
ax2.plot(f, p1, color='tab:red', label='FFT of counts')
ax2.set_xlabel('Frequency [Hz]'); ax2.set_ylabel('Amplitude'); ax2.set_title('Single-sided amplitude spectrum of daily events')
myplot(axoi=ax2)
plt.show()
