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
from    scipy                import  io
import scipy as spy
from    scipy.fft            import  fftshift
import  pandas               as      pd
import  time                 as      tm
from matplotlib.ticker import MaxNLocator

from kavutil import *
from kav_init import *

# ----------------------------------------------------------------------------------------------------------------------

# --- Import shark catalog
# catalogfile     = 'catalog_times_0.1_lb.txt'
# headers         = ['date','ratio','shark','delays']
catalogfile     = 'catalogue_4.0_lb.txt'
headers         = ['date','ratio','shark']
dtypes          = [str, float, float]
catalog         = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\s+', names=headers) #parse_dates=['date'])
catalog['index2'] = np.arange(len(catalog))

# --- Import array catalog
cata_arr        = 'catalog_pl.txt'
headers_arr     = ['dstart','tstart','dend','tend','baz','v_app','rmse']
dtypes_arr      = [str, str, str, str, float, float, float]
catalog_arr     = pd.read_csv(rootdata+'/'+cata_arr,
                              header=0, sep='\s+', names=headers_arr)

# --- count array events
cata_arr_new    = pd.DataFrame({'dates_arr':  catalog_arr.dstart,
                                'baz' :       catalog_arr.baz })
cata_arr_new.drop(cata_arr_new[cata_arr_new.baz <= -160 ].index, inplace=True)
cata_arr_new.drop(cata_arr_new[cata_arr_new.baz >= -90 ].index,  inplace=True)
dfarr_counting  = pd.DataFrame({'dates_arr': pd.to_datetime(cata_arr_new.dates_arr)}).set_index(np.arange(len(cata_arr_new)))
countsarr       = dfarr_counting.groupby(['dates_arr']).size()
dfarr           = pd.DataFrame({'day':      countsarr.index,
                                'counts':    countsarr.values})

totarr          = dfarr.counts.sum()
dfarr['norm']   = dfarr.counts / totarr
dfarr['roll'] = rolling_stats(dfarr['norm'].to_numpy(), np.mean, window=7)


# --- Count shark events
# puredate    = pd.to_datetime(catalog.index + ' ' + catalog.date, format='%Y-%m-%d %H:%M:%S')
# df          = pd.DataFrame({'Day': pd.to_datetime(catalog.index), 'Date': puredate}).set_index(np.arange(len(catalog)))
df          = pd.DataFrame({'Day': pd.to_datetime(catalog.index)}).set_index(np.arange(len(catalog)))

counts      = df.groupby(['Day']).size()
df          = pd.DataFrame({'day': counts.index, 'counts': counts.values})
tot         = df.counts.sum()
df['norm']  = df.counts / tot
df['roll'] = rolling_stats(df['norm'].to_numpy(), np.mean, window=7)

# cmap='PRGn',vmin=0, vmax=2,

# --- count weekly
df2          = pd.DataFrame({ 'day':    df['day'],
                              'counts': df['counts']})
df2['weeks'] = df2['day'].dt.strftime('%V/%Y')
counts2      = df2.groupby(['weeks']).agg({'counts':'sum'})

dfweek       = pd.DataFrame({'week':      counts2.index,
                             'counts':    counts2.values[:,0],
                             'norm':      (counts2.values[:,0] / tot)})

dfarr2       = dfarr_counting.copy()
dfarr2['weeks'] = dfarr2['dates_arr'].dt.strftime('%V/%Y')
countsarr2   = dfarr2.groupby(['weeks']).size()
dfarr2       = pd.DataFrame({'week':  countsarr2.index,
                             'counts':countsarr2.values,
                             'norm': (countsarr2.values / totarr) })


# --- separate in two continious timeintervals
t1start = pd.to_datetime(datetime.datetime(2023,2,7))
t1end   = pd.to_datetime(datetime.datetime(2023,5,2))
t2start = pd.to_datetime(datetime.datetime(2023,5,23))
t2end   = pd.to_datetime(datetime.datetime(2023,7,25))


dt = 60*60*24

ft1,     yt1,     ep1     = fft_slice(df.norm,     df.day,    dt, t1start, t1end)
ft2,     yt2,     ep2     = fft_slice(df.norm,     df.day,    dt, t2start, t2end)
farrt1,  yarrt1,  ep1arr  = fft_slice(dfarr.norm,  dfarr.day, dt, t1start, t1end)
farrt2,  yarrt2,  ep2arr  = fft_slice(dfarr.norm,  dfarr.day, dt, t2start, t2end)

fwt1,    ywt1,    epw1    = fft_slice(dfweek.norm, pd.to_datetime(dfweek.week+'-1', format="%V/%G-%u"), dt*7, t1start, t1end)
fwt2,    ywt2,    epw2    = fft_slice(dfweek.norm, pd.to_datetime(dfweek.week+'-1', format="%V/%G-%u"), dt*7, t2start, t2end)
fwarrt1, ywarrt1, epw1arr = fft_slice(dfarr2.norm, pd.to_datetime(dfweek.week+'-1', format="%V/%G-%u"), dt*7, t1start, t1end)
fwarrt2, ywarrt2, epw1arr = fft_slice(dfarr2.norm, pd.to_datetime(dfweek.week+'-1', format="%V/%G-%u"), dt*7, t2start, t2end)

# --- define freq-ticks
day2sec    = 60*60*24
prepfticks  = np.array([3.5, 7, 14, 21, 30.4, 60.8])
fticks      = 1/ (prepfticks * [day2sec])
fticklabels = np.array(['halfweekly','weekly','biweekly','triweekly','monthly','bimonthly'])


# --- Correlation "Spearman"
s11       = pd.Series(ep1.data)
s21       = pd.Series(ep1arr.data)
corr1     = s11.corr(s21, method='spearman')

s12       = pd.Series(ep2.data)
s22       = pd.Series(ep2arr.data)
corr2     = s12.corr(s22, method='spearman')

# fits1 = np.polynomial.polynomial.polyfit()

# --- new plot design
fignewday = plt.figure(figsize=(10,10))
gs      = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])
ax0     = plt.subplot(gs[0])
# ax0.set_title('Compare event distribution for different detection methods')
ax0.bar(    ep1arr.time, ep1arr.data, color='tab:red', alpha=0.5, label='Array analysis, total '+str(totarr)+' events')
ax0.plot(   ep1arr.time, ep1arr.data, color='k', alpha=0.5) #, label='arr.')
ax0.bar(    ep2arr.time, ep2arr.data, color='tab:red', alpha=0.5)
ax0.plot(   ep2arr.time, ep2arr.data, color='k', alpha=0.5)
ax0.hlines(0.001,  pd.to_datetime(datetime.datetime(2023,5,3)),  pd.to_datetime(datetime.datetime(2023,5,22)), color="red", label="no data")
ax0.set_ylabel('Normalised counts'); myplot(axoi=ax0, gridalpha=0.7)
# ax0.xaxis.set_label_position('top')
# plt.setp(ax0.get_xticklabels(), visible=False)
ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
yticks0 = ax0.yaxis.get_major_ticks()
yticks0[0].label1.set_visible(False)

ax1 = plt.subplot(gs[1], sharex= ax0)
ax1.bar(    ep1.time, ep1.data, color='orange',  alpha=0.5, label='dual-station trigger, total '+str(tot)+' events')
ax1.plot(   ep1.time, ep1.data, color='k',  alpha=0.5) #, label='shark')
ax1.bar(    ep2.time, ep2.data, color='orange',  alpha=0.5)
ax1.plot(   ep2.time, ep2.data, color='k',  alpha=0.5)
ax1.hlines(0.001,  pd.to_datetime(datetime.datetime(2023,5,3)),  pd.to_datetime(datetime.datetime(2023,5,22)), color="red", label="no data")

ax1.set_ylabel('Normalised counts')
ax1.set_xlabel('Time [Days]')
ax1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
myplot(axoi=ax1, gridalpha=0.7)

ax2 = plt.subplot(gs[2])
ax2.set_title('')
ax2.plot(ft1,    yt1,    label='dual-station trigger Feb - May', color='blue')
ax2.plot(ft2,    yt2,    label='dual-station trigger, May - July', color='blue',    linestyle='--')
ax2.plot(farrt1, yarrt1, label='array analysis Feb - May', color='red')
ax2.plot(farrt2, yarrt2, label='array analysis, May - July', color='red',     linestyle='--')

ax2.set_xticks(fticks)
ax2.set_xticklabels(fticklabels)
ax2.tick_params(labelrotation=35)
ax2.set_xlabel('Frequency of Kavachi events')# \n "Spearman" correlation between both detection methods {corr1:.2f} (Feb - May) and {corr2:.2f} (May - July) \n'.format(corr1=corr1, corr2=corr2))
ax2.set_ylabel('Amplitude')
myplot(axoi=ax2,gridalpha=0.9)

# plt.subplots_adjust(hspace=0.)
fignewday.align_ylabels()
plt.savefig(rootdata+'/statistics/comp_distr_daily_wlvl_grouped_new.svg', format='svg', dpi=1200, bbox_inches='tight')
plt.show()





# --- plot daily
fig = plt.figure(figsize=(10,10) )
gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])

ax0 = plt.subplot(gs[0])
ax0.set_title('Event distribution for array analysis vs shark trigger')
ax0.bar(    dfarr.day, dfarr.norm, color='tab:red', alpha=0.5, label='Array, total '+str(totarr)+' events')
ax0.plot(   dfarr.day, dfarr.norm, color='tab:red',       alpha=0.5, label='arr.')
ax0.plot(   dfarr.day, dfarr['roll'], color='green', linestyle='--', alpha=0.5, label='arr., smoothed')
ax0.plot(   df.day, df['roll'], color='k',  alpha=0.5, linestyle='--', label='shark smoothed')
ax0.set_ylabel('Counts'); myplot(axoi=ax0)
# ax0.xaxis.set_label_position('top')
# plt.setp(ax0.get_xticklabels(), visible=False)
ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
yticks0 = ax0.yaxis.get_major_ticks()
yticks0[0].label1.set_visible(False)

ax1 = plt.subplot(gs[1], sharex= ax0)
ax1.bar(    df.day, df.norm, color='orange',    alpha=0.5, label='shark, total '+str(tot)+' activies')
ax1.plot(   df.day, df.norm, color='orange',  alpha=0.5, label='shark')
ax1.plot(   df.day, df['roll'], color='k',  alpha=0.5, linestyle='--', label='shark smoothed')
ax1.plot(   dfarr.day, dfarr['roll'], color='green', linestyle='--', alpha=0.5, label='arr. smoothed')
ax1.set_ylabel('Counts')
ax1.set_xlabel('Time [Days]')
ax1.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

myplot(axoi=ax1)

ax2 = plt.subplot(gs[2])
ax2.set_title('')
ax2.plot(ft1,    yt1,    label='shark, episode 1', color='tab:blue')
ax2.plot(ft2,    yt2,    label='shark, episode 2', color='blue',    linestyle='--')
ax2.plot(farrt1, yarrt1, label='array, episode 1', color='tab:red')
ax2.plot(farrt2, yarrt2, label='array, episode 2', color='red',     linestyle='--')

ax2.set_xticks(fticks)
ax2.set_xticklabels(fticklabels)
ax2.tick_params(labelrotation=35)
ax2.set_xlabel('Frequency \n Single-sided amplitude spectrum of daily Kavachi activity')
ax2.set_ylabel('Amplitude')
myplot(axoi=ax2)

# plt.subplots_adjust(hspace=0.)
plt.savefig(rootdata+'/statistics/comp_distr_daily_new.png', dpi=300, bbox_inches='tight')
plt.show()


# --- plot weekly ---
fig2    = plt.figure(figsize=(10,10))
gs      = gridspec.GridSpec(3,1, height_ratios=[1,1,1])

ax0 = plt.subplot(gs[0])
ax0.set_title('Event distribution for array analysis vs shark trigger')
ax0.bar(    dfarr2.week, dfarr2.norm, color='tab:red', alpha=0.5, label='Histogram, '+str(totarr)+' events')
ax0.plot(   dfarr2.week, dfarr2.norm, color='k',       alpha=0.5, label='Counts per week by array')
ax0.set_xlabel('Time [Weeks]')
ax0.set_ylabel('Counts'); myplot(axoi=ax0)
ax0.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, rotation=45)
ax0.xaxis.set_label_position('top')
yticks0 = ax0.yaxis.get_major_ticks()
yticks0[0].label1.set_visible(False)

ax1 = plt.subplot(gs[1])#, sharex= ax0)
ax1.bar(    dfweek.week, dfweek.norm, color='orange',    alpha=0.5, label='Histogram, '+str(tot)+' activies')
ax1.plot(   dfweek.week, dfweek.norm, color='tab:blue',  alpha=0.5, label='Counts per week')
ax1.set_ylabel('Counts')
plt.setp(ax1.get_xticklabels(), visible=False)
myplot(axoi=ax1)

ax2 = plt.subplot(gs[2])
# ax2.plot(f, p1, color='tab:blue', label='FFT events')
# ax2.plot(farr, p1_arr, color='tab:red', label='FFT events by array')
# ax2.set_xlabel('Frequency [Hz] \n Single-sided amplitude spectrum of daily events distribution ')
# ax2.set_ylabel('Amplitude')
ax2.plot(dfweek.week, dfweek.norm, color='tab:blue',  alpha=0.5, label='shark triggered', linestyle='--')
ax2.plot(dfarr2.week, dfarr2.norm, color='tab:red', alpha=.5, label='array', linestyle='--')
# ax2.set_xlabel('Correlation (after Pearson) = '+str(weekcorr))
myplot(axoi=ax2)

plt.subplots_adjust(hspace=.0)
# plt.savefig(rootdata+'/statistics/comp_distr_weekly.png', dpi=300, bbox_inches='tight')
plt.show()

