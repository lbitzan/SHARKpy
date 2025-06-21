from    obspy.core          import  read
import  numpy               as      np
import  os, datetime
import  matplotlib.pyplot   as      plt
from    kavutil            import  myplot

# print('Current working directory: ', os.getcwd())
os.chdir('KavScripts')
rootcode    = os.getcwd(); os.chdir('../')
rootproject = os.getcwd(); os.chdir(rootproject+'/Data')
rootdata    = os.getcwd()
os.chdir(rootcode)
# print('rootcode    :', rootcode); print('rootproject :', rootproject); print('rootdata    :', rootdata)

# Load kavachi data from station kav04 and kav11
kav04       = read(rootdata+'/kav04/c0939230405000000.pri0')
kav11       = read(rootdata+'/kav11/c0941230405000000.pri0')
dist        = 8.525 # km

sr          = kav04[0].stats.sampling_rate
# print('Sampling rate: ', sr)
dt          = 1./sr
tr04        = kav04[0]
tr11        = kav11[0]
time        = tr04.times(reftime = tr04.stats.starttime)
npts        = tr04.stats.npts                # number of samples

# Frequencies for KaVA Index
fmin        = int(3)                        # for general filtering used in bandpass filter
fmax        = int(60)                       

f1min1      = int(4)   # empirical value for remote station
f1max1      = int(8)   # empirical value

f2min1      = int(4)   # empirical value for close station
f2max1      = int(7)   # empirical value
f2min2      = int(12)  # empirical value
f2max2      = int(50)  # empirical value

# Compute specgrams and KaVA Index
nfft        = int(sr)  # int(sr)
noverlap    = int(sr*.75) #= int(sr/2)
d04spec, d04f, d04t, cax = plt.specgram(tr04.data, NFFT=nfft, Fs= int(sr), noverlap=noverlap, detrend='none', scale_by_freq=True, cmap='viridis')
plt.close()
d11spec, d11f, d11t, cax = plt.specgram(tr11.data, NFFT=nfft, Fs= int(sr), noverlap=noverlap, detrend='none', scale_by_freq=True, cmap='viridis')
plt.close()
kava11      = (np.sum(d11spec[f2min1:f2max1,:], axis=0)+np.sum(d11spec[f2min2:f2max2],axis=0)) / (np.sum(d11spec[fmin:f2min1,:],axis=0) + np.sum(d11spec[f2max1:f2min2,:],axis=0) + np.sum(d11spec[f2max2:fmax],axis=0))
kava04      = np.sum(d04spec[f1min1:f1max1,:],axis=0) / (np.sum(d04spec[fmin:f1min1,:],axis=0) +  np.sum(d04spec[f1max1:fmax],axis=0))

# Define the start and end time of the 20-second window around the event
window1_time    = 48*60. + 100.
window1_start   = window1_time
window1_end     = window1_time + 20.

window2_time    = 51*60. + 75.
window2_start   = window2_time
window2_end     = window2_time + 20.

mins            = 60.
hours           = mins*60
winstarts       = [ 0*mins + 35.,
                    3*mins + 100.,
                    3*mins + 135.,
                    6*mins,
                    6*mins + 100.,
                    9*mins + 130.,
                   15*mins + 40.,
                   15*mins + 80.,
                   21*mins + 55.,
                   27*mins + 60.,
                   39*mins + 80.,
                   48*mins + 20.,
                   48*mins + 55.,
                   48*mins + 100., 
                   51*mins + 75.,
                   51*mins + 130., 
                    8*hours+  3*mins + 45., 
                    8*hours+  3*mins + 100.,
                   10*hours+ 30*mins + 135.,]
winlen      = 20

shift_app   = np.zeros_like(winstarts)

for iwin in np.arange(len(winstarts)):

    # Find the indices corresponding to the start and end time in the data
    idx_win     = np.where((time >= winstarts[iwin]) & (time <= winstarts[iwin]+winlen))
    idx_kava    = np.where((d11t >= winstarts[iwin]) & (d11t <= winstarts[iwin]+winlen))

    # Cut out the 20-second window from the kavachi data
    tr11_win    = tr11.data[ idx_win]
    kava11_win  = kava11[ idx_kava]
    
    
    shifts      = np.arange(0.25, 4.25, 0.25)
    kavaprodmax = np.zeros_like(shifts)
    dtspec      = d11t[1] - d11t[0]

    winh        = int(winstarts[iwin] // (60*60))
    winm        = int((winstarts[iwin] % (60*60)) // 60)
    wins        = int((winstarts[iwin] % (60*60)) %  60 )
    moi         = 4 # april
    doi         = 5 # 5th
    customtime  = datetime.datetime(2023, moi, doi, winh, winm, wins)

    figwin, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(8, 10))
    figwin.suptitle('Stations KAV11-KAV04 distance 8.525 km \n 05/04/2023 - '+str(winh)+':'+str(winm)+':'+str(wins) +' UTC', fontsize=12, fontweight='bold')
    
    for i in np.arange(len(shifts)):
        idx_kava_shift  = np.where((d04t >= winstarts[iwin]+shifts[i]) & (d04t <= winstarts[iwin]+winlen+shifts[i]))
        kava04_win      = kava04[ idx_kava_shift]
        kavaprod_win    = kava11_win * kava04_win

        # Plot Kava Product over time
        ax2.plot(d11t[idx_kava]-winstarts[iwin], kavaprod_win, linewidth=.8, alpha=.4, label=str(shifts[i])+' s')

        kavaprodmax[i]  = np.amax(kavaprod_win)

    bestshift       = np.where(kavaprodmax == np.amax(kavaprodmax))
    shift_app[iwin] = shifts[bestshift][0]  #kavaprodmax[bestshift]

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Kava Product')
    myplot(axoi=ax2)

    # Plot Max Kava Product corresponding to each shift
    ax3.plot(shifts, kavaprodmax, 'o', color='orange', alpha=.3, label='Max Kava Product')
    ax3.scatter(shifts[bestshift], kavaprodmax[bestshift], s=80 ,marker=(5,2) ,color='red', label='Best shift')

    ax3.set_ylabel('KaVA Product')
    myplot(axoi=ax3)
    ax3.set_xlabel('Time shift [sec] \n v_app [km/s]')
    ax3.set_xticks(shifts)
    ax3.set_xticklabels(['0.25\n34.1','0.5\n17.1','0.75\n11.4','1\n8.5','1.25\n6.8','1.5\n5.68','1.75\n4.9','2\n4.3','2.25\n3.8','2.5\n3.4','2.75\n3.1','3\n2.8','3.25\n2.6','3.5\n2.4','3.75\n2.3','4\n2.1'])
    ax3.tick_params(axis='x', labelrotation=45)

    # Plot Traces
    ax0.plot(time[idx_win]-winstarts[iwin], tr11_win, color='blue', alpha=.3,linewidth=1,label='KAV11')
    ax1.plot(time[idx_win]-winstarts[iwin], tr04[idx_win+ np.int64(shifts[bestshift]/dt)].flatten(), color='tab:red',alpha=.3,linewidth=1,label='KAV04 \n shift '+str(shifts[bestshift][0])+' s')
    
    ax0.set_xlabel('Time (s)'); ax0.set_ylabel('Amplitude'); myplot(axoi=ax0)
    ax0.set_xticks(np.arange(0, winlen+1, 2.5))
    ax0.tick_params(axis='x', labelbottom=False, labeltop=True, labelrotation=45)

    ax1.set_ylabel('Amplitude')
    ax1.tick_params(axis='x', labelbottom=False, labeltop=False)
    myplot(axoi=ax1)

    # Set show/save/close figure
    # plt.show()
    # plt.savefig(rootproject+'/results/arrtime/arrtime'+customtime.strftime("%H_%M_%S_%B_%d_%Y")+'.png', dpi=300, bbox_inches='tight')
    plt.close(figwin)

print('shift_app: ', shift_app)

v_app = dist / shift_app

# plot v_app
figvapp, ax = plt.subplots(1, 1, figsize=(8, 6))
figvapp.suptitle('Distribution apparent time shifts \n for events triggered by both KaVA product index and Amplitude ratio')
# Calculate the histogram of apparent velocities
num_bins = np.arange(0, 4.25, .25)
ax.hist(v_app, bins=num_bins, edgecolor='black', align='left', label='resolution 0.25 s')
ax.set_yticks(np.arange(0,5.5,1))
ax.set_xticks(np.arange(0.25,4.25,.25))
ax.set_xticklabels(['0.25\n34.1','0.5\n17.1','0.75\n11.4','1\n8.5','1.25\n6.8','1.5\n5.68','1.75\n4.9','2\n4.3','2.25\n3.8','2.5\n3.4','2.75\n3.1','3\n2.8','3.25\n2.6','3.5\n2.4','3.75\n2.3','4\n2.1'])
ax.tick_params(axis='x', labelrotation=45)
myplot(axoi=ax)

# Add labels and title to the histogram plot
ax.set_xlabel('Time shift [sec] \n v_app [km/s]')
ax.set_ylabel('Counts')

# Display & Save the histogram plot
# plt.show()
plt.savefig(rootproject+'/results/vapp/v_app_hist.png', dpi=300, bbox_inches='tight')
plt.close(figvapp)
