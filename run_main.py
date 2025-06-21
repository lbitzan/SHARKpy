'''
Test for different scenarios of ratio range to investigate a plateau or similar in the number of events.
'''

if __name__ == '__main__':
    import numpy as np, os as os, pandas as pd, kavutil_2 as kt
    if not os.getcwd() == 'C:/Users/Ludwig/projectsT14/GitHub/KavScripts':
        os.chdir('C:/Users/Ludwig/projectsT14/GitHub/KavScripts')
    
    # from runPlotCatalog import runPlotCatalog
    # from kavutil import rmv_tremorphases
    # from runPlotCatalog1_4 import runPlotCatalog
    from runplotcatalog1_6 import runPlotCatalog
    from kav_init import *
    import kav_init as ini
    from multiprocessing import Pool, Process
    # from concurrent.futures import ProcessPoolExecutor

    # start, end = '2023-02-01', '2023-12-01'
    # tbreakvalues = np.array([20., 1., 5., 10., 30., 40., 50., 60., 120.], dtype=float)
    start, end = '2023-04-05 00:00:00', '2023-04-05 01:20:00'
    ini.sectionlen = 60*20 # set section length in seconds
    ini.fontsizelegend = 8
    runPlotCatalog(datetime_start=start, datetime_end=end)#, trigbreak=tbreakvalues)

    # start, end = '2023-04-05 00:00:00', '2023-04-05 02:00:00'
    # runPlotCatalog(datetime_start=start, datetime_end=end)

    # start, end = '2023-05-30 00:00:20', '2023-05-30 02:00:00'
    # ini.sectionlen = 60*20 # set section length in seconds
    # runPlotCatalog(datetime_start=start, datetime_end=end)
    
    # shifttimes = [3.6, 3.4, 3.78, 4.25, 3.09, 2.83, 2.62, 2.43, 2.27, 2.13]
    # shifttimes = [3.78, 4.25, 3.09, 2.83, 2.62, 2.43, 2.27, 2.13, 3.4]

    # 
    # tbreakvalues = np.array([5.], dtype=float)
    # for delay in shifttimes:
    #     ini.shifttime = delay
    #     runPlotCatalog(datetime_start=start, datetime_end=end, trigbreak=tbreakvalues)

    # ini.stationdir  = ['KAV04', 'KAV11']  # set station directory
    # ini.stationid   = ['c0939', 'c0941']  # set station id
    # ini.outputlabel = 'v11.3D_full_compute_'
    # ini.cataloguefile = 'cata_v11.3D_full_compute_'
    # ini.kavaprodemp = 10
    # start, end      = '2023-02-01', '2023-12-01'
    # runPlotCatalog(datetime_start=start, datetime_end=end, trigbreak=tbreakvalues)

    # ini.outputlabel = 'v11.bbd_full.compute_'
    # ini.cataloguefile       = 'cata_v11.bbd_full.compute_'
    # ini.flag_compute_freqbands = True

    # runPlotCatalog(datetime_start=start, datetime_end=end, trigbreak=tbreakvalues)


    # tbreakvalues = np.array([60.,120.,50.,40.,30.], dtype=float)
    # tbreakvalues = np.array([20., 1., 5., 10.], dtype=float)
    # runPlotCatalog(datetime_start='2023-06-01', datetime_end='2023-06-30', trigbreak=tbreakvalues)

    # [runPlotCatalog(datetime_start=start, datetime_end=end, trigbreak=tbreak) for tbreak in tbreakvalues]
    # --- load data for control
    # dfslim, dfratio = [],[]
    # [dfslim.append(pd.read_pickle(ini.rootouts,'cata_v11.3D_empiric_'+str(tbreak)+'s_slim.pkl')) for tbreak in tbreakvalues]
    # [dfratio.append(pd.read_pickle(ini.rootouts,'cata_v11.3D_empiric_'+str(tbreak)+'s_ratio.pkl')) for tbreak in tbreakvalues]




    # for rmin in np.arange(10,200,10):     # --- test different values for lower ratio range limit
    #     rmax        = ratiorange[1]; del ratiorange
    #     ratiorange  = [rmin, rmax]
        
    #     runPlotCatalog(rrange=ratiorange)

    # for rmax in np.arange(800,1050,50):    # --- test different values for upper ratio range limit
    #     rmin       = ratiorange[0]; del ratiorange
    #     ratiorange = [rmin, rmax]
        
    #     runPlotCatalog(rrange=ratiorange)

    # def run_catalog_sim(trigbreak):
    #     print(trigbreak)
    #     runPlotCatalog(trigbreak=trigbreak, datetime_start='2023-02-01', datetime_end='2023-07-30')

    # trigbreaks = np.array([15., 16., 17., 18., 19., 21., 22., 23., 24., 25., 26., 27., 28., 29., 35.])
    # processes = []

    # for trigbreak in trigbreaks:
    #     p = Process(target=run_catalog_sim, args=(trigbreak,))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # for trigbreak in np.array([.5, 5., 10., 20., 30., 40., 50., 60.]):
    #     runPlotCatalog(trigbreak=trigbreak, datetime_start='2023-02-01', datetime_end='2023-11-30')

    # for trigbreak in np.array([15., 16., 17., 18., 19., 21., 22., 23., 24., 25., 26., 27., 28., 29., 35.,36.,37.,38.,39.,14.,13.,12.,11.]):
    #     runPlotCatalog(trigbreak=trigbreak, datetime_start='2023-02-01', datetime_end='2023-07-30')

    # for trigbreak in np.array([1.5,36.,37.,38.,39.,14.,13.,12.,11.]):
    #     runPlotCatalog(trigbreak=trigbreak, datetime_start='2023-02-01', datetime_end='2023-07-30')
# -------------------

    # for trigbreak in np.array([0.5, 10., 20., 30.]):
    #     runPlotCatalog(trigbreak=trigbreak, datetime_start='2023-04-05 00:40:00', datetime_end='2023-04-05 01:00:00')

    #     runPlotCatalog(trigbreak=trigbreak, datetime_start='2023-05-30 01:00:00', datetime_end='2023-05-30 01:20:00')



    # dfprom = rmv_tremorphases('D:/data_kavachi_both/results/catalogs_withprominence_trigbreak/catalog_7.2_withprominence.trigbreak_00.5.txt',
    #                           episode=['2023-02-01', '2023-08-01'],
    #                           savedf=rootdata+'/catalog_7.2_prominence_rmvtrigger.pkl')
    # dfwidth = rmv_tremorphases('D:/data_kavachi_both/results/catalogs_noprominence_trigbreak/catalog_7.2_noprominence.trigbreak_00.5.txt',
    #                            episode=['2023-02-01', '2023-08-01'],
    #                            savedf=rootdata+'/catalog_7.2_width_rmvtrigger.pkl')



# ###### read catalogs for line counting
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from   kavutil import eval_eventcount, myplot
# from   tqdm import tqdm
# import datetime as datetime
# from   kavutil import rmv_tremorphases

# dirs = np.array(['20230225', '20230330', '20230405', '20230605', '20230701'], dtype=str)
# neventdict = dict()
# # threshold   = np.arange(20,155,5)
# threshold   = np.arange(200,1050,100)

# for j, dir in enumerate(dirs):

#     path_cata   = 'D:/data_kavachi_both/results/evcnt.rrtest.'+dir+'.catalogs' # 20230225, 20230330, 20230405, 20230605, 20230701
#     files       = [os.path.join(path_cata, _) for _ in os.listdir(path_cata) if 'evcnt_rrtest' in _]
#     files.sort()
#     nfiles      = len(files)
#     nevents     = np.zeros(nfiles)

#     for i,f in enumerate(files):
#         with open(f) as file:
#             nevents[i] = len(file.readlines()) - 1

#     neventdict[dir] = nevents

# for key in neventdict.keys():
#     plt.plot(threshold, neventdict[key], marker='.', label=key.format('%Y%b%d'))
# plt.xlabel('Upper ratio range limit\n lower limit fixed at 40')
# plt.ylabel('Eventcount/24h')
# plt.suptitle('Eventcount/24h for varying ratio range')
# plt.legend() # plt.show()
# plt.tight_layout()
# plt.savefig('D:/data_kavachi_both/results/threshold_eventcount_lowerlimit.png', dpi=400)
# plt.close()

# df = pd.DataFrame(neventdict)
# df.set_index(threshold, inplace=True)
# df.to_pickle('D:/data_kavachi_both/results/threshold_eventcount_upperlimit.pkl')

# # df = pd.DataFrame({'threshold':threshold, 'nevents':nevents})
# # df.to_pickle('D:/data_kavachi_both/results/threshold_eventcount_20230405.pkl')

# # plt.plot(threshold[:15],nevents[:15]); plt.show()

# # --- read in data
# dfprominence    = eval_eventcount('D:/data_kavachi_both/results/catalogs_withprominence_trigbreak')
# dfwidth         = eval_eventcount('D:/data_kavachi_both/results/catalogs_noprominence_trigbreak')

# dfpromd1        = dfprominence['counts'].diff()
# dfpromd2        = dfpromd1.diff()
# dfwidthd1       = dfwidth['counts'].diff()
# dfwidthd2       = dfwidthd1.diff()

# fig, ax = plt.subplots(1,1, figsize=(8,6))
# ax.plot(dfprominence['parameter'], dfprominence['counts'], ':x',
#             color='tab:blue',
#             label='with prominence')
# ax.vlines(dfprominence['parameter'][np.where(dfpromd2==0)[0]],
#             ymin=dfprominence['counts'].min(),
#             ymax=dfprominence['counts'].max(),
#             color='tab:blue',
#             linestyle='-')
# ax.plot(dfwidth['parameter'], dfwidth['counts'], '--x',
#             color='tab:red',
#             label='with width')
# ax.vlines(dfwidth['parameter'][dfwidthd2.idxzeros()],
#             ymin=dfwidth['counts'].min(),
#             ymax=dfwidth['counts'].max(),
#             color='tab:red',
#             linestyle='-')
# myplot(axoi=ax)
# plt.show()

# # investigate curvature of eventcount
# fig, ax = plt.subplots(3,1, figsize=(8,6), sharex=True)
# ax[0].set_title('Eventcount for varying trigger rest time')
# cprom = 'tab:blue'; cwidth = 'tab:red'
# ax[0].plot(dfprominence['parameter'],   dfprominence['counts'], ':x',  color=cprom,  label='"prominence based"')
# ax[0].plot(dfwidth['parameter'],        dfwidth['counts'],      '--x', color=cwidth, label='"width based"')
# ax[1].plot(dfprominence['parameter'],   dfpromd1, ':x',     color=cprom)
# ax[1].plot(dfwidth['parameter'],        dfwidthd1, '--x',   color=cwidth)
# ax[2].plot(dfprominence['parameter'],   dfpromd2, ':x',     color=cprom)
# ax[2].plot(dfwidth['parameter'],        dfwidthd2, '--x',   color=cwidth)
# ax[2].hlines(0, dfwidth['parameter'].min(), dfwidth['parameter'].max(),
#              color='k', linestyle=':', alpha=.5, label='7.5 s - inflection point')
# ax[2].set_xlabel('Varying trigger rest time')
# [axi.vlines(7.5, axi.get_ylim()[0], axi.get_ylim()[1], color='k', linestyle=':', alpha=.5) for axi in ax]
# ax[0].set_ylabel('Total eventcount')
# ax[1].set_ylabel('1st derivative')
# ax[2].set_ylabel('2nd derivative')
# [myplot(axoi=axi) for axi in ax]
# fig.align_ylabels(ax)
# plt.savefig('D:/data_kavachi_both/results/eventcount_curvature.png', dpi=400)
# plt.show()

# # nvestigate distribution of events over time. Plot different catalogs in one plot

# path ='D:/data_kavachi_both/results/catalogs_withprominence_trigbreak'
# data_gap = pd.to_datetime(np.array(['2023-05-02', '2023-05-23'], dtype=str))
# catadict ={}
# files =[os.path.join(path, _) for _ in os.listdir(path) if '*.txt']
# files.sort()
# nfiles =len(files)
# nevents =np.zeros(nfiles)

# threshold = [float(fname[-8:-4]) for fname in files]
# thresholdstring = [fname[-8:-4] for fname in files]

# from kavutil import read_catalog
# from multiprocessing import Pool


# df = read_catalog(files[0])
# df.drop(columns=['ratio','kavachi_index'], inplace=True)
# # df['counter'] = 1; df = df.groupby(df['date'].dt.date).count()
# dfday = pd.DataFrame({'date':    pd.to_datetime(df.groupby(df['date'].dt.date).count()['date'].index),
#                      'counter': df.groupby(df['date'].dt.date).count()['date'].values})

# print('Reading in catalogs:')
# dfdict = {}
# for i,f in enumerate(tqdm(files)):

#     df = read_catalog(f)
#     df.drop(columns=['ratio','kavachi_index'], inplace=True)
#     dfnew = pd.DataFrame({'date':    pd.to_datetime(df.groupby(df['date'].dt.date).count()['date'].index),
#                      'counter': df.groupby(df['date'].dt.date).count()['date'].values})
#     dfdict[threshold[i]] = dfnew.copy()
#     del dfnew, df

# dfprom = rmv_tremorphases('D:/data_kavachi_both/results/catalogs_withprominence_trigbreak/catalog_7.2_withprominence.trigbreak_00.5.txt',
#                            episode=['2023-02-01', '2023-08-01'])
# dfprom.drop(columns=['ratio','kavachi_index','index'], inplace=True)
# # dfprom.drop(dfprom['tremor'] == True, inplace=True)
# df.drop(df[df['tremor'] == False].index, inplace=True)
# dfprom = dfprom[dfprom['tremor']]
# dfpromnew = pd.DataFrame({'date':    pd.to_datetime(dfprom.groupby(dfprom['date'].dt.date).count()['date'].index),
#                           'counter': dfprom.groupby(dfprom['date'].dt.date).count()['date'].values})
# dfpromnew_ep1, dfpromnew_ep2 = dfpromnew.copy(deep=True), dfpromnew.copy(deep=True)
# dfpromnew_ep1.drop(dfpromnew_ep1.index[dfpromnew_ep1['date'].dt.date > data_gap[0].date()], inplace=True)
# dfpromnew_ep2.drop(dfpromnew_ep2.index[dfpromnew_ep2['date'].dt.date < data_gap[1].date()], inplace=True)

# print('Split in two data episodes:')
# dfdict_ep1, dfdict_ep2 = {}, {}
# for key in tqdm(dfdict.keys()):
#     dfdict_ep1[key], dfdict_ep2[key] = dfdict[key].copy(deep=True), dfdict[key].copy(deep=True)
# [dfdict_ep1[key].drop(dfdict_ep1[key].index[dfdict_ep1[key]['date'].dt.date > pd.to_datetime('2023-05-02').date()], inplace=True) for key in dfdict_ep1.keys()]
# [dfdict_ep2[key].drop(dfdict_ep2[key].index[dfdict_ep2[key]['date'].dt.date < pd.to_datetime('2023-05-23').date()], inplace=True) for key in dfdict_ep2.keys()]

# # Plot episode 1:
# fig_ep1, ax = plt.subplots(1,1, figsize=(8,6))
# ax.set_title('Eventcount from ' + dfdict_ep1[list(dfdict_ep1.keys())[0]]['date'][dfdict_ep1[list(dfdict_ep1.keys())[0]].index[ 0]].strftime('%Y-%b-%d') +
#              ' to ' +             dfdict_ep1[list(dfdict_ep1.keys())[0]]['date'][dfdict_ep1[list(dfdict_ep1.keys())[0]].index[-1]].strftime('%Y-%b-%d'))
# [ax.fill_between(dfdict_ep1[key]['date'], dfdict_ep1[key]['counter'],
#                  alpha=.7,
#                  label=key) for key in dfdict_ep1.keys()]
# ax.set_xlabel('Date')
# ax.set_ylabel('Eventcount per day')
# ax.tick_params(axis='x', rotation=45)
# myplot(axoi=ax)
# plt.tight_layout()
# plt.savefig('D:/data_kavachi_both/results/eventcount_trigbreak_episode1.png', dpi=400)
# plt.show()

# # Plot episode 2:
# fig_ep2, ax = plt.subplots(1,1, figsize=(8,6))
# ax.set_title('Eventcount from ' + dfdict_ep2[list(dfdict_ep2.keys())[0]]['date'][dfdict_ep2[list(dfdict_ep2.keys())[0]].index[ 0]].strftime('%Y-%b-%d') +
#              ' to ' +             dfdict_ep2[list(dfdict_ep2.keys())[0]]['date'][dfdict_ep2[list(dfdict_ep2.keys())[0]].index[-1]].strftime('%Y-%b-%d'))
# [ax.fill_between(dfdict_ep2[key]['date'], dfdict_ep2[key]['counter'],
#                  alpha=.7,
#                  label=key) for key in dfdict_ep2.keys()]
# ax.set_xlabel('Date')
# ax.set_ylabel('Eventcount per day')
# ax.tick_params(axis='x', rotation=45)
# myplot(axoi=ax)
# plt.tight_layout()
# plt.savefig('D:/data_kavachi_both/results/eventcount_trigbreak_episode2.png', dpi=400)
# plt.show()

# # Plot both episodes in one plot and mark data gap with horizontal line
# # Define colors for each threshold 
# colors = ['cornflowerblue', 'orange', 'mediumseagreen', 'indianred', 'plum',
#           'tab:red','orchid', 'darkgray', 'yellowgreen', 'paleturquoise',
#           'cadetblue', 'darkorange', 'darkseagreen', 'lightcoral', 'mediumpurple',
#           'darkgoldenrod','gold', 'rosybrown']

# fig, ax = plt.subplots(1,1, figsize=(8,6))
# ax.set_title('With varying triggerresting time \n Eventcount from ' + 
#              dfdict[list(dfdict.keys())[0]]['date'][0].strftime('%Y-%b-%d') +' to ' +
#              dfdict[list(dfdict.keys())[0]]['date'][len(dfdict[list(dfdict.keys())[0]])-1].strftime('%Y-%b-%d'))
# for i, key in enumerate(dfdict_ep1.keys()):
#     ax.fill_between(dfdict_ep1[key]['date'], dfdict_ep1[key]['counter'],
#                     alpha=.7,
#                     label=key,
#                     color=colors[i])
#     ax.fill_between(dfdict_ep2[key]['date'], dfdict_ep2[key]['counter'],
#                     alpha=.7,
#                     color=colors[i])

# ax.hlines(0, pd.to_datetime('2023-05-03'), pd.to_datetime('2023-05-22'), color='k', linestyle=':', alpha=.5, label='Data gap')
# ax.set_xlabel('Date')
# ax.set_ylabel('Eventcount per day')
# ax.tick_params(axis='x', rotation=45)
# myplot(axoi=ax)
# plt.tight_layout()
# plt.savefig('D:/data_kavachi_both/results/eventcount_trigbreak_bothepisodes.png', dpi=400)
# plt.show()



