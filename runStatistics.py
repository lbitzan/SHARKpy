'''
>>> runStatistics.py <<<
This script is used to run the statistics on the event catalogue created by "runPlotting.py".
By analysing the occurence and rate of events potential abundaces should be unravelled.

L. Bitzan, 2024
2024-01-08 // general setup

'''

def plotKavStats(outputfilename):
    '''
    Plot Kavachi statistics.
    Input:
    outputfilename: string
        name of output file
    '''

    if flag_save:
        plt.savefig(outputdir+outputfilename, dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()    

# --- Import libraries ------------------------------------------------------------------------------------------------
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

# --- SET PATHS --------------------------------------------------------------------------------------------------------
print('Current working directory: ', os.getcwd()); os.chdir('KavScripts')
rootcode    = os.getcwd(); os.chdir('../')
rootproject = os.getcwd(); os.chdir(rootproject+'/Data')
# rootdata    = os.getcwd()                                   # <--- set path to smaller data set on local machines
rootdata    = 'D:/data_kavachi_both'                          # <--- set path to larger data set on external hard drive
os.chdir(rootcode)
print('rootcode    :', rootcode); print('rootproject :', rootproject); print('rootdata    :', rootdata)

outputdir   = rootdata + '/statistics/'
outweek     = 'activity_per_week.png'
outallweek  = 'activity_all_per_week.png'
outmonth    = 'activity_per_month.png'
outday      = 'activity_per_day.png'


# --- Import local libraries --------------------------------------------------------------------------------------------
from kavutil import  myplot, rolling_stats

# --- Flags & Definitions -----------------------------------------------------------------------------------------------
cataloguefile    = 'catalogue_2.0_lb.txt'

startdate   = '2023-02-07'
enddate     = '2023-07-25'

flag_save   = True   # True: save figures; False: show figures
flag_weeks  = False  # True: plot weekly statistics; False: don't plot weekly statistics
flag_months = False  # True: plot monthly statistics; False: don't plot monthly statistics
flag_days   = True  # True: plot daily statistics; False: don't plot daily statistics

# --- Import data -------------------------------------------------------------------------------------------------------
headers     = ['date','ratio','shark']
dtypes      = [str, float, float]
catalogue   = pd.read_csv(rootdata+'/'+cataloguefile, sep=',', header=0, names=headers, parse_dates=['date'])
slen        = len(catalogue['date'])
catalogue['amount']     = np.ones(slen)     # add column of ones for counting
catalogue['event']      = ['counts'] * slen # add column

print(catalogue)

# --- Compute statistics ------------------------------------------------------------------------------------------------

# kw      = lambda x: x.isocalendar()[1]
# kw_year = lambda x: str(x.year) + '-' + str(x.isocalendar()[1])


# --- Weekly statistics ---------------------------------------------------------------------------------------------------
if flag_weeks:
    counts4week = (catalogue
        .groupby('event')                
        .apply(lambda g:               # work on groups of col1
            g.set_index('date')        
            [['amount']]
            .resample('W').agg('sum')  # sum the amount field across weeks
        )
        .unstack(level=0)              # pivot the event index rows to columns
        .fillna(0))

    shark4week = (catalogue.groupby('event').apply(lambda g:               # work on groups of col1
            g.set_index('date')        
            [['shark']]
            .resample('W').agg('mean')  # sum the amount field across weeks
        )
        .unstack(level=0).fillna(0))

    # group4week = pd.DataFrame({'Counts': counts4week.amount['counts'],
    #                         'SHARK mean': shark4week.shark['counts']},index=counts4week.index)
    group4week = pd.DataFrame({'Counts': counts4week.amount['counts']},index=counts4week.index)

    figweek, axweek    = plt.subplots(figsize=(20,10))
    # axweek             = group4week.plot(kind='bar',secondary_y='SHARK mean',color={'Counts':'red','SHARK mean':'tab:blue'})
    axweek             = group4week.plot(kind='bar')

    axweek.set_xticklabels(group4week.index.strftime("%d %b %y"), rotation=45, ha='right', fontsize='x-small')
    axweek.set_title(   'Events per week \n  - based on SHARK index -', fontsize=16)
    axweek.set_xlabel(  'Weeks', fontsize=14)
    axweek.set_ylabel(  'Counts', fontsize=14)
    # axweek.right_ax.set_ylabel('SHARK index', fontsize=14)
    axweek.grid(         color = 'w', alpha = .5)
    # axweek.legend(loc='best')
    axweek.set_facecolor('0.9')

    # Save/Plot figure
    plotKavStats(outweek)


# --- Monthly statistics ---------------------------------------------------------------------------------------------------
if flag_months:
    counts4mon = (catalogue
        .groupby('event')                
        .apply(lambda g:               # work on groups of col1
            g.set_index('date')        
            [['amount']]
            .resample('M').agg('sum')  # sum the amount field across weeks
        )
        .unstack(level=0)              # pivot the event index rows to columns
        .fillna(0))
    
    group4mon = pd.DataFrame({'Counts': counts4mon.amount['counts']}, index=counts4mon.index)

    figmon, axmon   = plt.subplots(figsize=(20,10))
    axmon           = group4mon.plot(kind='bar', color={'Counts':'tab:red'})
    axmon.set_xticklabels(group4mon.index.strftime("%d %b %y"), rotation=45, ha='right', fontsize='x-small')
    axmon.set_title(    'Events per months \n  - based on SHARK index -', fontsize=16)
    axmon.set_xlabel(   'Months', fontsize=14)
    axmon.set_ylabel(   'Counts', fontsize=14)
    axmon.grid(          color = 'w', alpha = .5)
    axmon.legend(        loc='best')
    axmon.set_facecolor('0.9')

    plt.savefig(outputdir+outmonth, dpi=600, bbox_inches='tight')

    # # Save/Plot figure
    # plotKavStats(outmonth)



if flag_days:
    counts4day = (catalogue
        .groupby('event')                
        .apply(lambda g:               # work on groups of col1
            g.set_index('date')        
            [['amount']]
            .resample('d').agg('sum')  # sum the amount field across weeks
        )
        .unstack(level=0)              # pivot the event index rows to columns
        .fillna(0))
    
    group4day  = pd.DataFrame({'Counts': counts4day.amount['counts']},index=counts4day.index)

    figday, axday    = plt.subplots(figsize=(20,10))
    axday             = group4day.plot(kind='bar')
    axday.set_xticklabels(group4day.index.strftime("%d %b %y"), rotation=45, ha='right', fontsize='x-small')
    # organise x-axis
    for i, t in enumerate(group4day.index):
        if t.weekday() == 1:
            axday.axvline(i, color='k', linestyle='--', alpha=.5, linewidth=1)
    for i, t in enumerate(axday.get_xticklabels()):
        if (i % 7) != 0:
            t.set_visible(False)

    axday.set_title(    'Events per day \n  - based on SHARK index -', fontsize=16)
    axday.set_xlabel(   'Days', fontsize=14)
    axday.set_ylabel(   'Counts', fontsize=14)
    axday.grid(          color = 'w', alpha = .5)
    axday.set_facecolor('0.9')

    # Save/Plot figure
    plotKavStats(outday)





# # --- Plot statistics ---------------------------------------------------------------------------------------------------
# # weekly
# evPerWeek = catalogue["times"].groupby(pd.cut(catalogue.index, pd.date_range(startdate, enddate, freq='1W'))).count()
# axw = evPerWeek["times"].plot(kind="bar", figsize=(10,5))
# plt.show()
# # daily
# # monthly


# df2.groupby(df2['times'].dt.day).count().plot(kind="bar")

# df2plot = df2.groupby(df2['times'].dt.day).count()
# ax = df2plot.plot(kind="bar", figsize=(10,5))
# plt.show()

# df2times = df2["times"].groupby(df2['times'].dt.day).count()
# ax = df2times.plot(kind="bar", figsize=(10,5))
# plt.show()

# evPerWeek = df2["times"].groupby(df2['times'].dt.week).count()
# axw = evPerWeek.plot(kind="bar", figsize=(10,5))
# plt.show()

