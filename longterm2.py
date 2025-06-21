# --- Import libraries ------------------------------------------------------------------------------------------------
import os
if not os.getcwd() == 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts':
    os.chdir('C:/Users/Arbeit/Documents/matlab/projects/KavachiProject/KavScripts')
from    obspy.core           import  read, UTCDateTime, Stream, Trace
from    obspy.imaging        import  spectrogram
import  numpy                as      np
import  os, datetime, glob
import  matplotlib.pyplot    as      plt
from    matplotlib           import  gridspec
import  matplotlib.colorbar  as      cbar
from    scipy                import  io, signal
from    scipy.fft            import  fftshift
import  pandas               as      pd
import  time                 as      tm
from    matplotlib.ticker    import MaxNLocator

from kavutil import *
from kav_init import *

# ----------------------------------------------------------------------------------------------------------------------

# Import catalog
catalogfile = 'catalog_times_0.1_lb.txt'
headers     = ['date','ratio','shark','delays']
dtypes      = [str, float, float, float]
catalog     = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\s+', names=headers) #parse_dates=['date'])
catalog['index2'] = np.arange(len(catalog))

# import array catalog
cata_arr      = 'catalog_pl.txt'
headers_arr   = ['dstart','tstart','dend','tend','baz','v_app','rmse']
dtypes_arr    = [str, str, str, str, float, float, float]
catalog_arr   = pd.read_csv(rootdata+'/'+cata_arr,
                          header=0, sep='\s+', names=headers_arr)

# --- count array events
cata_arr_new = pd.DataFrame({'dates_arr':  catalog_arr.dstart,
                             'baz' :       catalog_arr.baz })
cata_arr_new.drop(cata_arr_new[cata_arr_new.baz <= -140 ].index, inplace=True)
cata_arr_new.drop(cata_arr_new[cata_arr_new.baz >= -100 ].index, inplace=True)
df_arr2count= pd.DataFrame({'dates_arr': pd.to_datetime(cata_arr_new.dates_arr)}).set_index(np.arange(len(cata_arr_new)))
counts_arr  = df_arr2count.groupby(['dates_arr']).size()
df_arr      = pd.DataFrame({'day':      counts_arr.index,
                            'counts':    counts_arr.values})

tot_arr         = df_arr.counts.sum()
df_arr['norm']  = df_arr.counts / tot_arr
df_arr['rolling'] = rolling_stats(df_arr['norm'].to_numpy(), np.mean, window=7)


# --- array event frequency ---
fqcnts_arr  = np.fft.fft(df_arr.counts)
fs_arr      = 1/(86400) # 1 day in seconds
p2_arr      = abs(fqcnts_arr)/len(fqcnts_arr)
p1_arr      = p2_arr[0:len(fqcnts_arr)//2]
p1_arr[1:-1] = 2*p1_arr[1:-1]

f_arr = fs_arr/len(fqcnts_arr)*(np.arange(len(fqcnts_arr)//2))



# Compute apparent velocity
distance    = 8.525
vapp        = distance / catalog.delays.values

# Group by day
puredate = pd.to_datetime(catalog.index + ' ' + catalog.date, format='%Y-%m-%d %H:%M:%S')
df      = pd.DataFrame({'Day': pd.to_datetime(catalog.index), 'Date': puredate}).set_index(np.arange(len(catalog)))
counts  = df.groupby(['Day']).size()
dfnew   = pd.DataFrame({'day': counts.index, 'counts': counts.values})
tot     = dfnew.counts.sum()
dfnew['norm'] = dfnew.counts / tot

dfnew['rolling'] = rolling_stats(dfnew['norm'].to_numpy(), np.mean, window=7)

fqcnts  = np.fft.fft(dfnew.counts)
fs      = 1/(86400) # 1 day in seconds
p2      = abs(fqcnts)/len(fqcnts)
p1      = p2[0:len(fqcnts)//2]
p1[1:-1]  = 2*p1[1:-1]

f = fs/len(fqcnts)*(np.arange(len(fqcnts)//2))


# --- New Plot
fig = plt.figure(figsize=(10,10), )
gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])

ax0 = plt.subplot(gs[0])
ax0.set_title('Event distribution for array analysis vs shark trigger')
ax0.bar(    df_arr.day, df_arr.norm, color='tab:red', alpha=0.5, label='Histogram, '+str(tot_arr)+' events')
ax0.plot(   df_arr.day, df_arr.norm, color='k',       alpha=0.5, label='Counts per day by array')
ax0.plot(   df_arr.day, df_arr['rolling'], color='green', linestyle='--', alpha=0.5, label='smoothed')
ax0.plot(   dfnew.day, dfnew['rolling'], color='k',  alpha=0.5, linestyle='--', label='shark smoothed')
ax0.set_xlabel('Time [Days]')
ax0.set_ylabel('Counts'); myplot(axoi=ax0)
ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax0.xaxis.set_label_position('top')
yticks0 = ax0.yaxis.get_major_ticks()
yticks0[0].label1.set_visible(False)

ax1 = plt.subplot(gs[1], sharex= ax0)
ax1.bar(    dfnew.day, dfnew.norm, color='orange',    alpha=0.5, label='Histogram, '+str(tot)+' activies')
ax1.plot(   dfnew.day, dfnew.norm, color='tab:blue',  alpha=0.5, label='Counts per day')
ax1.plot(   dfnew.day, dfnew['rolling'], color='k',  alpha=0.5, linestyle='--', label='smoothed')
ax1.plot(   df_arr.day, df_arr['rolling'], color='green', linestyle='--', alpha=0.5, label='array smoothed')
ax1.set_ylabel('Counts')
plt.setp(ax1.get_xticklabels(), visible=False)
myplot(axoi=ax1)

ax2 = plt.subplot(gs[2])
ax2.plot(f, p1, color='tab:blue', label='FFT events')
ax2.plot(f_arr, p1_arr, color='tab:red', label='FFT events by array')
ax2.set_xlabel('Frequency [Hz] \n Single-sided amplitude spectrum of daily events distribution ')
ax2.set_ylabel('Amplitude')
myplot(axoi=ax2)

plt.subplots_adjust(hspace=.0)
plt.savefig(rootdata+'/statistics/comp_distr_daily_new.png', dpi=300, bbox_inches='tight')
plt.show()

# --- weekly
dfnew2 = pd.DataFrame({ 'day': df['Day']})
dfnew2['weeks'] = dfnew2['day'].dt.strftime('%V/%Y')
countsweek = dfnew2.groupby(['weeks']).size()
dfweek = pd.DataFrame({'week':      countsweek.index,
                       'counts':    countsweek.values,
                       'norm':     (countsweek.values / tot) })

df_arr_2 = df_arr2count.copy()
df_arr_2['weeks'] = df_arr_2['dates_arr'].dt.strftime('%V/%Y')
countsweek_arr = df_arr_2.groupby(['weeks']).size()
dfweek_arr = pd.DataFrame({'week':  countsweek_arr.index,
                           'counts':countsweek_arr.values,
                           'norm': (countsweek_arr.values / tot_arr) })

# --- plot weekly ---
fig2 = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])

ax0 = plt.subplot(gs[0])
ax0.set_title('Event distribution for array analysis vs shark trigger')
ax0.bar(    dfweek_arr.week, dfweek_arr.norm, color='tab:red', alpha=0.5, label='Histogram, '+str(tot_arr)+' events')
ax0.plot(   dfweek_arr.week, dfweek_arr.norm, color='k',       alpha=0.5, label='Counts per week by array')
ax0.set_xlabel('Time [Weeks]')
ax0.set_ylabel('Counts'); myplot(axoi=ax0)
ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, rotation=45)
ax0.xaxis.set_label_position('top')
yticks0 = ax0.yaxis.get_major_ticks()
yticks0[0].label1.set_visible(False)

ax1 = plt.subplot(gs[1], sharex= ax0)
ax1.bar(    dfweek.week, dfweek.norm, color='orange',    alpha=0.5, label='Histogram, '+str(tot)+' activies')
ax1.plot(   dfweek.week, dfweek.norm, color='tab:blue',  alpha=0.5, label='Counts per week')
ax1.set_ylabel('Counts')
plt.setp(ax1.get_xticklabels(), visible=False)
myplot(axoi=ax1)

# --- Correlation "pearson"
s1 = pd.Series(dfweek.norm)
s2 = pd.Series(dfweek_arr.norm)
weekcorr = s1.corr(s2)
# fits1 = np.polynomial.polynomial.polyfit()

ax2 = plt.subplot(gs[2])
# ax2.plot(f, p1, color='tab:blue', label='FFT events')
# ax2.plot(f_arr, p1_arr, color='tab:red', label='FFT events by array')
# ax2.set_xlabel('Frequency [Hz] \n Single-sided amplitude spectrum of daily events distribution ')
# ax2.set_ylabel('Amplitude')
ax2.plot(dfweek.week, dfweek.norm, color='tab:blue',  alpha=0.5, label='shark triggered', linestyle='--')
ax2.plot(dfweek_arr.week, dfweek_arr.norm, color='tab:red', alpha=.5, label='array', linestyle='--')
ax2.set_xlabel('Correlation (after Pearson) = '+str(weekcorr))
myplot(axoi=ax2)

plt.subplots_adjust(hspace=.0)
plt.savefig(rootdata+'/statistics/comp_distr_weekly.png', dpi=300, bbox_inches='tight')
plt.show()

