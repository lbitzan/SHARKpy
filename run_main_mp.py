'''
Test for different scenarios of ratio range to investigate a plateau or similar in the number of events.
'''

if __name__ == '__main__':
    import numpy as np
    import os
    # from runPlotCatalog import runPlotCatalog
    from runPlotCatalog1_4 import runPlotCatalog
    from kav_init import *
    from kavutil import rmv_tremorphases
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from   kavutil import eval_eventcount, myplot, eval_eventcount_withtremorphaseremoval, read_catalog, myplot_noleg
    from   tqdm import tqdm
    import datetime as datetime
    from   kavutil import rmv_tremorphases
    from   kav_init import *
    from obspy import Stream
    breakpoint()

    if '/KavScripts' in os.getcwd():
        print('Current working directory is correct.')
    else:
        if '/KavScripts' not in os.listdir():
            print('KavScripts not in current directory. Please change directory to KavScripts and come back.')
            breakpoint()
        else:
            os.chdir('KavScripts')
            print('Current working directory is correct now.')


    # from kavutil import run4catalog
    # run4catalog(datetime_start  = '2023-02-01 00:00',
    #             datetime_end    = '2023-08-01 12:00:00')
    # breakpoint()

    # breakpoint()
    # read in data with varying triggerrestingtime
    dfprom = eval_eventcount(rootdata + '/results/catalogs_withprominence_trigbreak/', flag_individual=True)
    dfwidth = eval_eventcount(rootdata + '/results/catalogs_noprominence_trigbreak/', flag_individual=True)

    dfrmin = eval_eventcount(rootdata + '/results/catalogs_8.0_varyrmin/', flag_individual=False, thresholdinfo=[10, 200, 10])
    dfrmax = eval_eventcount(rootdata + '/results/catalogs_8.0_varyrmax/', flag_individual=False, thresholdinfo=[200, 1000, 50])

    # breakpoint()
    
    def plot_eventcount(df, ax, title=None, xlabel=None, color=None, label=None):
        if title is not None:
            ax.set_subtitle(title)
        if label is not None and color is not None:
            ax.plot(df['parameter'], df['counts'], 'o-', color=color, label=label)
        elif label is not None and color is None:
            ax.plot(df['parameter'], df['counts'], 'o-', label)
        elif label is None and color is not None:
            ax.plot(df['parameter'], df['counts'], 'o-', color=color)
        else:
            ax.plot(df['parameter'], df['counts'], 'o-')
        ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.set_ylabel('Eventcount')
    
    # plot & compare data to see which triggerresttime is best
    # breakpoint()
    # fig, ax = plt.subplots(1,1, figsize = (8, 6))
    # plot_eventcount(dfprom, ax, color='tab:blue', label='prominence based')
    # plot_eventcount(dfwidth, ax, color='tab:red', label='width based')
    # ax.set_title('Eventcount for different triggerrestingtimes')
    # ax.set_xlabel('Triggerrestingtime [s]')
    # myplot(axoi=ax)
    # plt.savefig(rootouts + '/thesis/triggerresttimes_and_detectionmethod.png', dpi=600)
    # plt.show()
    breakpoint()

    # compare data for different rmin values to see which is best
    dfR         = dict()
    itrates     = np.arange(2,13,1)
    pathrmin    = rootdata + '/results/catalogs_8.0_varyrmin/'
    pathrmax    = rootdata + '/results/catalogs_8.0_varyrmax/'
    files       = [os.path.join(pathrmin, _) for _ in os.listdir(pathrmin) if '*.txt']
    nfiles      = len(files)
    listdf_rmin     = {}
    listdf_rmax     = {}
    listdf_cata     = {}
    listdf_catarmv  = {}

    for i, itrate in enumerate(tqdm(itrates)):
        df4trate_rmin = eval_eventcount_withtremorphaseremoval(
                            pathrmin,
                            flag_individual=True,
                            thresholdinfo=[10, 200, 10],
                            flag_plot=False,
                            tremor_rate=itrate,
                            episode=['2023-02-01 00:00', '2023-08-01 12:00:00'])
        
        df4trate_rmax = eval_eventcount_withtremorphaseremoval(
                            pathrmax,
                            flag_individual=True,
                            thresholdinfo=[200, 1000, 50],
                            flag_plot=False,
                            tremor_rate=itrate,
                            episode=['2023-02-01 00:00', '2023-08-01 12:00:00'])
        
        df4trate_rmv = rmv_tremorphases(rootouts + '/catalogs_8.0_varyrmin/rmin040catalog_8.0_varylwrbnd.lb.txt',
                                        itrate,
                                        episode=['2023-02-01 00:00', '2023-08-01 12:00:00'],
                                        remove_tremor=True)
        
        df4trate = rmv_tremorphases(rootouts + '/catalogs_8.0_varyrmin/rmin040catalog_8.0_varylwrbnd.lb.txt',
                                        itrate,
                                        episode=['2023-02-01 00:00', '2023-08-01 12:00:00'],
                                        remove_tremor=False)
        
        listdf_catarmv[itrate]  = df4trate_rmv
        listdf_cata[itrate]     = df4trate
        listdf_rmin[itrate]     = df4trate_rmin
        listdf_rmax[itrate]     = df4trate_rmax

        # Compute listdf_catatremor by iterating over all dataframes in listdf_cata and drop all events, whicht are marked as False in column 'tremor'.
        listdf_catatremor = {}
        for idx, f in enumerate(listdf_cata):
            listdf_catatremor[f] = listdf_cata[f].loc[listdf_cata[f]['tremor'] == True]

    # plot comparison eventcount/rmin & cnts/rmax depending on tremor rate
    breakpoint()
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    plot_eventcount(listdf_rmin[itrates[0]], ax[0], color='C0', label='2  [events / 5min] ')
    [plot_eventcount(listdf_rmin[itrate], ax[0], color=f'C{i}', label=f'{itrate}') for i, itrate in enumerate(itrates) if i > 0]
    [plot_eventcount(listdf_rmax[itrate], ax[1], color=f'C{i}') for i, itrate in enumerate(itrates)]
    [myplot(axoi=axi) for axi in ax]
    ax[0].set_title('Eventcount for varying tremor rates')
    ax[0].set_xlabel('ratiorange minimum'); ax[0].set_ylabel('Eventcount')
    ax[1].set_xlabel('ratiorange maximum')
    plt.tight_layout()
    plt.savefig(rootouts + '/thesis/rmin_rmax_eventcount.png', dpi=600)
    plt.show()

    breakpoint()

    gs_kw    = dict(width_ratios=[3, 1])
    fig, ax  = plt.subplot_mosaic([[0, 1]], gridspec_kw = gs_kw, figsize = (12, 6), layout = "constrained")
    
    ax[0].set_title('Eventcount for varying tremor rates')
    ax[0].set_xlabel('Time'); ax[0].set_ylabel('Eventcount')
    ax[1].set_title('Total Eventcount')
    ax[1].set_xlabel('Tremor rate'); ax[1].set_ylabel('Total Eventcount')

    ax[0].plot(listdf_catarmv[itrates[0]]['date'].groupby(listdf_catarmv[itrates[0]]['date'].dt.date).count().drop(listdf_catarmv[itrates[0]]['date'].groupby(listdf_catarmv[itrates[0]]['date'].dt.date).count().index[(
            pd.to_datetime(listdf_catarmv[itrates[0]]['date'].groupby(listdf_catarmv[itrates[0]]['date'].dt.date).count().index) >= pd.to_datetime(seasons[0][0])) & 
            (pd.to_datetime(listdf_catarmv[itrates[0]]['date'].groupby(listdf_catarmv[itrates[0]]['date'].dt.date).count().index) <= pd.to_datetime(seasons[0][1])) ]) ,
            label=f'{itrates[0]}  [events / 5min]', color='C0')
    [ax[0].plot(
        listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count(
        ).drop(listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count().index[(
            pd.to_datetime(listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count().index) >=
            pd.to_datetime(seasons[0][0])) & (
            pd.to_datetime(listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count().index) <= 
            pd.to_datetime(seasons[0][1])) ]) ,
            label=f'{f}', color=f'C{idx+1}')
        for idx, f in enumerate(itrates[1:])]
            
    bins    = itrates
    bindata = [len(listdf_catarmv[itrate]['date']) for itrate in itrates]
    ax[1].bar(bins, bindata)

    [myplot(axoi=ax[axi]) for axi in ax]
    [ax[0].plot(
        listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count(
        ).drop(listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count().index[(
            pd.to_datetime(listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count().index) >=
            pd.to_datetime(seasons[1][0])) & (
            pd.to_datetime(listdf_catarmv[f]['date'].groupby(listdf_catarmv[f]['date'].dt.date).count().index) <= 
            pd.to_datetime(seasons[1][1])) ]),
            color=f'C{idx}')
        for idx, f in enumerate(itrates)]
    plt.tight_layout()
    plt.savefig(rootouts + '/thesis/tremorrate_eventcount_time_with_total.png', dpi=600)
    plt.show()





    breakpoint()
    # Fuse all datasets from listdf_cata... into one dataframe structure in a fused dataframe.
    def fuse_df_eventcount(listofdataframes):
        '''
        Fuse all datasets from listdf_cata... into one dataframe structure in a fused dataframe.
        Therefore, compute for each dataframe in listdf_catarmv the daily eventcount.
        Store that data in fused dataframe with the corresponding tremor rate as column name.
        '''
        
        dffuse              = pd.DataFrame()                                 # Initialize empty dataframe
        for idx, f in enumerate(listofdataframes):                          # Compute daily eventcount for each dataframe
            print(f'Processing dataframe {idx+1} of {len(listofdataframes)}: "{f}"')
            dailycounts     = listofdataframes[f]['date'].groupby(listofdataframes[f]['date'].dt.date).count()
            dffuse[f]     = dailycounts
            
        dffuse.index.name   = 'Date'
        dffuse.reset_index(inplace=True)

        dffuse['mean_dly']  = dffuse.iloc[:, 1:].mean(axis=1)                # Add columns for mean, std and nans
        dffuse['std_dly']   = dffuse.iloc[:, 1:-1].std(axis=1)
        dffuse['nans']      = dffuse.iloc[:, 1:-2].isna().sum(axis=1)

        return dffuse


    dffuse_rmv    = fuse_df_eventcount(listdf_catarmv)
    dffuse        = fuse_df_eventcount(listdf_cata)
    dffuse_tremor = fuse_df_eventcount(listdf_catatremor)

    breakpoint()
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title('Eventcount for varying tremor rates')
    ax[1].set_ylabel('Eventcount')
    ax[1].set_xlabel('Time'); ax[0].set_ylabel('Eventrate [events / 5min]')
    dffuse_rmv_ep1 = dffuse_rmv[(pd.to_datetime(dffuse_rmv['Date']) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(dffuse_rmv['Date']) <= pd.to_datetime(seasons[0][1]))]
    dffuse_rmv_ep2 = dffuse_rmv[(pd.to_datetime(dffuse_rmv['Date']) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(dffuse_rmv['Date']) <= pd.to_datetime(seasons[1][1]))]
    ax[0].plot(dffuse_rmv_ep1['Date'], dffuse_rmv_ep1['mean_dly'], label='mean', color='C0')
    ax[0].plot(dffuse_rmv_ep2['Date'], dffuse_rmv_ep2['mean_dly'], color='C0')
    ax[0].fill_between(dffuse_rmv_ep1['Date'], dffuse_rmv_ep1['mean_dly'] - dffuse_rmv_ep1['std_dly'], dffuse_rmv_ep1['mean_dly'] + dffuse_rmv_ep1['std_dly'],
                       label='Std', alpha=0.7, color='C2')
    ax[0].fill_between(dffuse_rmv_ep2['Date'], dffuse_rmv_ep2['mean_dly'] - dffuse_rmv_ep2['std_dly'], dffuse_rmv_ep2['mean_dly'] + dffuse_rmv_ep2['std_dly'],
                       alpha=0.7, color='C2')
    ax[1].plot(dffuse_rmv_ep1['Date'], dffuse_rmv_ep1[2], label='2  [events / 5min]', color='C0')
    ax[1].plot(dffuse_rmv_ep2['Date'], dffuse_rmv_ep2[2], color='C0')
    [ax[1].plot(dffuse_rmv_ep1['Date'], dffuse_rmv_ep1[f], label=f'{f}', color=f'C{idx+1}') for idx, f in enumerate(itrates[1:])]
    [ax[1].plot(dffuse_rmv_ep2['Date'], dffuse_rmv_ep2[f], color=f'C{idx+1}') for idx, f in enumerate(itrates[1:])]
    [myplot(axoi=axi) for axi in ax]
    plt.tight_layout()
    plt.savefig(rootouts + '/thesis/tremorrate_eventcount_mean_std.png', dpi=600)
    plt.show()
    breakpoint()

    # plot events for 2023-04-24
    # Upper ax_ Plot activity rate per 5min.
    # Lower ax: Plot effective eventcount per 10 min for each tremor rate.
    datesofinterest = [[4,5],[4,24],[4,18],[5,29],[6,5]]
    for m, d in datesofinterest:
        dateofinterest = datetime.date(2023, m, d)
        fig, ax = plt.subplots(1,1, figsize=(8, 6))
        ax.set_title('Activity rate for '+dateofinterest.strftime('%Y-%m-%d'))
        ax.set_xlabel('Time')
        ax.set_ylabel('Activity rate [events / 5min]')
        # ax[1].set_title('Eventcount for 2023-04-24')
        # ax[1].set_xlabel('Time'); ax[1].set_ylabel('Eventcount')

        ax.plot(listdf_cata[2]['date'][          listdf_cata[2]['date'].dt.date == dateofinterest],
                    listdf_cata[2]['activity_rate'][  listdf_cata[2]['date'].dt.date == dateofinterest],
                    label='activityrate / 5min', color='tab:blue')
        ax.fill_between(listdf_cata[7]['date'][ (listdf_cata[7]['date'].dt.date == dateofinterest)], 0,
                            listdf_cata[7]['activity_rate'][(listdf_cata[7]['date'].dt.date == dateofinterest)],
                            where=listdf_cata[7]['tremor'][listdf_cata[7]['date'].dt.date== dateofinterest]==True,
                            label='tremor phases', color='tab:red', alpha=0.5)
        # Add marker for events on dateofinterest
        # ax.scatter(listdf_cata[7]['date'][listdf_cata[7]['date'].dt.date == dateofinterest],
        #                 listdf_cata[7]['activity_rate'][listdf_cata[7]['date'].dt.date == dateofinterest],
        #                 color='k',marker='x', label='events')
        ax.set_xlim(datetime.datetime(2023,m,d,0,0)-datetime.timedelta(minutes=30), datetime.datetime(2023,m,d,23,59)+datetime.timedelta(minutes=30))
        ax.set_xticks([datetime.datetime(2023,m,d,h,0) for h in range(0,24,2)])
        ax.set_xticklabels([f'{h}:00' for h in range(0,24,2)])
        ax.set_ylim(0, 30)
        myplot(axoi=ax)
        # ax[1].plot(
        #     listdf_catarmv[2][listdf_catarmv[2]['date'].dt.date == datetime.date(2023,4,24)].groupby(pd.Grouper(key='date', freq='5min')).count().iloc[:, -1].index,
        #     listdf_catarmv[2][listdf_catarmv[2]['date'].dt.date == datetime.date(2023,4,24)].groupby(pd.Grouper(key='date', freq='5min')).count().iloc[:, -1].values,
        #     label='2  [events / 5 min]', color='C0')
        # [ax[1].plot(
            # listdf_catarmv[f][listdf_catarmv[f]['date'].dt.date == datetime.date(2023,4,24)].
            #     groupby(pd.Grouper(key='date', freq='5min')).count().iloc[:, -1].index,
            # listdf_catarmv[f][listdf_catarmv[2]['date'].dt.date == datetime.date(2023,4,24)].
            #     groupby(pd.Grouper(key='date', freq='5min')).count().iloc[:, -1].values,
            # label=f'{f}', color=f'C{i}') for i,f in enumerate(itrates[1:])]
        # [myplot(axoi=axi) for axi in ax]
        plt.savefig(rootouts + f'/thesis/activityrate_daily_with_tremorphases_'+dateofinterest.strftime('%Y_%m_%d')+'.png', dpi=600)
        plt.show()
    

    # Save reference event catalog as reference event catalog with tremor phases removed.
    breakpoint()
    dfrefrmv = rmv_tremorphases(rootdata + '/catalog_8.0_reference.lb.txt',
                                episode = ['2023-02-01 00:00', '2023-08-01 12:00:00'],
                                savedf = rootdata+'/catalog_8.0_reference.lb.notremor.pkl',
                                remove_tremor =True)
    dfrefrmv.to_csv(rootdata+'/catalog_8.0_reference.lb.notremor.txt', index=False)
    dfref    = rmv_tremorphases(rootdata + '/catalog_8.0_reference.lb.txt',
                                episode = ['2023-02-01 00:00', '2023-08-01 12:00:00'],
                                savedf=rootdata+'/catalog_8.0_reference.lb.tremor.pkl')
    dfref.to_csv(rootdata+'/catalog_8.0_reference.lb.tremor.txt', index=False)

    # ---------------------------------------------------------------------------------------
    # Run Mlv for reference catalog with tremor phases removed.
    from instrument_restitution import run_Mlv, read_magnitude_catalog, plot_magnitudes
    
    catalog_1 = 'catalog_8.0_reference.lb.tremor.pkl'
    catalog_2 = 'catalog_pl.txt'
    name4output = 'catalog_8.0_reference.lb.tremor_'
    # year, month, day, hour, minute, stationdir, stationid, ..
    cases = {'a': (2023, 2, None, None, None, 'KAV11','c0941',None),
             'b': (2023, 3, None, None, None, 'KAV11','c0941',None),
             'c': (2023, 4, None, None, None, 'KAV11','c0941',None),
             'd': (2023, 5, None, None, None, 'KAV11','c0941',None),
             'e': (2023, 6, None, None, None, 'KAV11','c0941',None),
             'f': (2023, 7, None, None, None, 'KAV11','c0941',None)}
            #  'g': (2023, 8, None, None, None, 'KAV11','c0941',None)}

    mlv_cases = {}

    for i in cases:
        print(i)
        mlv, selected_events, stream_events, stream_events_gndmtn, stream_events_restored = run_Mlv( catalog_1,catalog_2,*cases[i], drop_details=False)
        
        if len(mlv) > 100:
            del stream_events, stream_events_gndmtn, stream_events_restored
            print('Empty variables deleted. Only Mlv magnitudes and selected events are returned.')
            mlv_cases[i] = {'Mlv':                      mlv,
                            'selected_events':          selected_events}
        else:
            mlv_cases[i] = {'Mlv':                      mlv,
                            'selected_events':          selected_events,
                            'stream_events':            stream_events,
                            'stream_events_gndmtn':     stream_events_gndmtn,
                            'stream_events_restored':   stream_events_restored}
        print(mlv)
       
        if len(mlv) < 100:
            mlv_print = mlv_cases[i].copy()
            del mlv_print['stream_events'], mlv_print['stream_events_gndmtn'], mlv_print['stream_events_restored']
        else:
            mlv_print = mlv_cases[i].copy()

        df_print = pd.DataFrame({'mlv': mlv_print['Mlv'],
                                 'selected_events': mlv_print['selected_events']['puretime'],
                                 'idx_catalog': mlv_print['selected_events'].index })
        
        outputpathcatalog = rootdata+"/results/magnitudes/"+name4output+"case_" + str(i) + "_magnest_"+str(cases[i])
        # df_print.to_csv(outputpathcatalog+'.txt', sep="\t", index=True)
        df_print.to_pickle(outputpathcatalog+'.pkl')
        dftest = pd.read_pickle(outputpathcatalog+'.pkl')

        # magnitude_catalog = read_magnitude_catalog(name4output+"case_"+str(i)+"_magnest_"+str(cases[i])+".txt")

        magnitude_catalog   = read_magnitude_catalog(outputpathcatalog+'.pkl')

        namespecifics       = name4output+'_'+str(i)+'_'+str(cases[i])
        plot_magnitudes(magnitude_catalog,namespecifics)

    breakpoint()

    
    pathmagresults = rootdata+'/results/magnitudes/magnitudes_catalog_8.0_reference.lb.tremor/'

    fnames      = [f for f in os.listdir(pathmagresults) if os.path.isfile(os.path.join(pathmagresults, f))]
    fnames.sort()
    dataframes  = []
    for f in fnames:
        df_partial  = pd.read_pickle(os.path.join(pathmagresults, f))
        dataframes.append(df_partial)
    df          = pd.concat(dataframes)
    
    df.rename(columns={'selected_events': 'events'}, inplace=True)
    df.to_pickle(rootdata+'/results/magnitudes/magnitudes_catalog_8.0_reference.lb.tremor.pkl')
    df.to_csv(rootdata+'/results/magnitudes/magnitudes_catalog_8.0_reference.lb.tremor.txt', index=False)
    plot_magnitudes(df, 'magnitudes_catalog_8.0_reference.lb.tremor')

    # Define function to read in reference catalog and add column containing the respective local vertical magnitude mlv. Save the resulting dataframe as pickle and txt file.
    # Name the files same as the reference files, just insert a '.mlv' before the '.pkl' suffix, '.txt' respectively. 
    def add_mlv_to_catalog(catalogfile, mlvcatalog):
        '''
        Read in reference catalog and coresponding mlv catalog. Add column with local vertical magnitude to the reference catalog.
        If both input catalogs are not the same length, check columns refdf['date'] and mlvdf['idx_catalog'] for matching entries.
        If matching entries are found, add mlv to refdf in mlv column. If not, write a NaN value.
        '''
        refdf = pd.read_pickle(catalogfile)
        mlvdf = pd.read_pickle(mlvcatalog)
        if len(refdf) == len(mlvdf):
            refdf['mlv'] = mlvdf['mlv']
        else:
            refdf['mlv'] = np.nan
            for idx, date in enumerate(refdf['date']):
                if date in mlvdf['events']:
                    refdf['mlv'].iloc[idx] = mlvdf['mlv'].iloc[mlvdf['events'].index(date)]
        
        refdf.to_pickle(catalogfile.replace('.pkl', '.mlv.pkl'))
        refdf.to_csv(catalogfile.replace('.pkl', '.mlv.txt'), index=False)

        return refdf
    

dfref_rmv_mlv = add_mlv_to_catalog(rootdata+'/catalog_8.0_reference.lb.notremor.pkl',
                                   rootdata+'/results/magnitudes/magnitudes_catalog_8.0_reference.lb.notremor.pkl')
dfref_mlv     = add_mlv_to_catalog(rootdata+'/catalog_8.0_reference.lb.tremor.pkl',
                                   rootdata+'/results/magnitudes/magnitudes_catalog_8.0_reference.lb.tremor.pkl')

plot_magnitudes(dfref_mlv, 'magnitudes_catalog_8.0_reference.lb.tremor', nolegend=True)
plot_magnitudes(dfref_rmv_mlv, 'magnitudes_catalog_8.0_reference.lb.notremor', nolegend=True)


# ---------------------------------------------------------------------------------------
# Evaluate catalog from bbd configuration

dfbbd = read_catalog(rootdata + '/catalogfull_8.0_bbd.lb.txt')
dfbbd.to_pickle(rootdata + '/catalogfull_8.0_bbd.lb.pkl')

dfbbd_notremor = rmv_tremorphases(rootdata + '/catalogfull_8.0_bbd.lb.txt',episode=['2023-02-01 00:00', '2023-08-01 12:00:00'],
                                  savedf=rootdata+'/catalogfull_8.0_bbd.lb.notremor.pkl', remove_tremor=True)
dfbbd_tremor   = rmv_tremorphases(rootdata + '/catalogfull_8.0_bbd.lb.txt',episode=['2023-02-01 00:00', '2023-08-01 12:00:00'],
                                  savedf=rootdata+'/catalogfull_8.0_bbd.lb.tremor.pkl', remove_tremor=False)

#seasons
dfbbd_ep1           = dfbbd[(pd.to_datetime(dfbbd['date']) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(dfbbd['date']) <= pd.to_datetime(seasons[0][1]))]
dfbbd_ep2           = dfbbd[(pd.to_datetime(dfbbd['date']) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(dfbbd['date']) <= pd.to_datetime(seasons[1][1]))]
dfbbd_notremor_ep1  = dfbbd_notremor[(pd.to_datetime(dfbbd_notremor['date']) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(dfbbd_notremor['date']) <= pd.to_datetime(seasons[0][1]))]
dfbbd_notremor_ep2  = dfbbd_notremor[(pd.to_datetime(dfbbd_notremor['date']) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(dfbbd_notremor['date']) <= pd.to_datetime(seasons[1][1]))]
dfbbd_tremor_ep1    = dfbbd_tremor[(pd.to_datetime(dfbbd_tremor['date']) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(dfbbd_tremor['date']) <= pd.to_datetime(seasons[0][1]))]
dfbbd_tremor_ep2    = dfbbd_tremor[(pd.to_datetime(dfbbd_tremor['date']) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(dfbbd_tremor['date']) <= pd.to_datetime(seasons[1][1]))]


# plot
fig, ax = plt.subplots(1,1, figsize=(8, 6))
ax.set_title('Daily eventcount for broadband station setup')
ax.set_xlabel('Time')
ax.set_ylabel('Eventcount')
ax.plot(dfbbd_ep1['date'].groupby(dfbbd_ep1['date'].dt.date).count(), color='tab:blue', label='tremor activies - included')
ax.plot(dfbbd_ep2['date'].groupby(dfbbd_ep2['date'].dt.date).count(), color='tab:blue')
ax.plot(dfbbd_notremor_ep1['date'].groupby(dfbbd_notremor_ep1['date'].dt.date).count(), color='tab:red', label='tremor activities - excluded')
ax.plot(dfbbd_notremor_ep2['date'].groupby(dfbbd_notremor_ep2['date'].dt.date).count(), color='tab:red')
# ax.fill_between(dfbbd_tremor['date'], 0,
#                             dfbbd_tremor['date'].groupby(dfbbd_tremor['date'].dt.date).count().max(),
#                             where=dfbbd_tremor['tremor']==True,
#                             label='tremor phases', color='tab:blue', alpha=0.5)
myplot(axoi=ax)
plt.savefig(rootouts + '/thesis/eventcount_bbd_tremor.notremor.png', dpi=600)
plt.show()


dfbbd_notremor['date'].groupby(dfbbd_notremor['date'].dt.date).count()


# ---------------------------------------------------------------------------------------
df_bbd_full = rmv_tremorphases(
    rootdata + '/catalogfull_9.0_bbd.lb.txt',
    episode=['2023-02-01 00:00', '2023-12-01 12:00:00'],
    savedf=rootdata + '/catalogfull_9.0_bbd.lb.tremor.pkl',
    remove_tremor=False)
df_3d_full = rmv_tremorphases(
    rootdata + '/catalogfull_9.0_3D.lb.txt',
    episode=['2023-02-01 00:00', '2023-12-01 12:00:00'],
    savedf=rootdata + '/catalogfull_9.0_3D.lb.tremor.pkl',
    remove_tremor=False)

df_bbd_full_cnts          = df_bbd_full['date'].groupby(df_bbd_full['date'].dt.date).count()
df_3d_full_cnts           = df_3d_full['date'].groupby(df_3d_full['date'].dt.date).count()

df_3d_full_cnts_notremor  = df_3d_full.iloc[df_3d_full.index[df_3d_full['tremor'] == False]]['date'].groupby(df_3d_full['date'].dt.date).count()
df_bbd_full_cnts_notremor = df_bbd_full.iloc[df_bbd_full.index[df_bbd_full['tremor'] == False]]['date'].groupby(df_bbd_full['date'].dt.date).count()

df_bbd_full_cnts_ep1          = df_bbd_full_cnts[(pd.to_datetime(df_bbd_full_cnts.index) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(df_bbd_full_cnts.index) <= pd.to_datetime(seasons[0][1]))]
df_bbd_full_cnts_ep2          = df_bbd_full_cnts[(pd.to_datetime(df_bbd_full_cnts.index) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(df_bbd_full_cnts.index) <= pd.to_datetime(seasons[1][1]))]
df_bbd_full_cnts_ep3          = df_bbd_full_cnts[(pd.to_datetime(df_bbd_full_cnts.index) >= pd.to_datetime(seasons[2][0])) & (pd.to_datetime(df_bbd_full_cnts.index) <= pd.to_datetime(seasons[2][1]))]

df_bbd_full_cnts_notremor_ep1 = df_bbd_full_cnts_notremor[(pd.to_datetime(df_bbd_full_cnts_notremor.index) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(df_bbd_full_cnts_notremor.index) <= pd.to_datetime(seasons[0][1]))]
df_bbd_full_cnts_notremor_ep2 = df_bbd_full_cnts_notremor[(pd.to_datetime(df_bbd_full_cnts_notremor.index) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(df_bbd_full_cnts_notremor.index) <= pd.to_datetime(seasons[1][1]))]
df_bbd_full_cnts_notremor_ep3 = df_bbd_full_cnts_notremor[(pd.to_datetime(df_bbd_full_cnts_notremor.index) >= pd.to_datetime(seasons[2][0])) & (pd.to_datetime(df_bbd_full_cnts_notremor.index) <= pd.to_datetime(seasons[2][1]))]

df_3d_full_cnts_ep1           = df_3d_full_cnts[(pd.to_datetime(df_3d_full_cnts.index) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(df_3d_full_cnts.index) <= pd.to_datetime(seasons[0][1]))]
df_3d_full_cnts_ep2           = df_3d_full_cnts[(pd.to_datetime(df_3d_full_cnts.index) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(df_3d_full_cnts.index) <= pd.to_datetime(seasons[1][1]))]
df_3d_full_cnts_notremor_ep1  = df_3d_full_cnts_notremor[(pd.to_datetime(df_3d_full_cnts_notremor.index) >= pd.to_datetime(seasons[0][0])) & (pd.to_datetime(df_3d_full_cnts_notremor.index) <= pd.to_datetime(seasons[0][1]))]
df_3d_full_cnts_notremor_ep2  = df_3d_full_cnts_notremor[(pd.to_datetime(df_3d_full_cnts_notremor.index) >= pd.to_datetime(seasons[1][0])) & (pd.to_datetime(df_3d_full_cnts_notremor.index) <= pd.to_datetime(seasons[1][1]))]

# Plotting function for event counts
def plot_event_counts_full(df_ep1, df_ep2, df_ep3, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(df_ep1.index, df_ep1.values, label='Episode 1', color='tab:blue')
    ax.plot(df_ep2.index, df_ep2.values, label='Episode 2', color='tab:orange')
    if df_ep3 is not None:
        ax.plot(df_ep3.index, df_ep3.values, label='Episode 3', color='tab:green')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Event Count')
    myplot(axoi=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Plotting for broadband station setup
plot_event_counts_full(df_bbd_full_cnts_ep1, df_bbd_full_cnts_ep2, df_bbd_full_cnts_ep3, 
                  'Daily Event Count for Broadband Station Setup', 
                  rootouts + 'thesis/eventcount_bbd_full.png')

plot_event_counts_full(df_bbd_full_cnts_notremor_ep1, df_bbd_full_cnts_notremor_ep2, df_bbd_full_cnts_notremor_ep3, 
                  'Daily Event Count for Broadband Station Setup (No Tremor)', 
                  rootouts + 'thesis/eventcount_bbd_full_notremor.png')

# Plotting for 3D station setup
plot_event_counts_full(df_3d_full_cnts_ep1, df_3d_full_cnts_ep2, None, 
                  'Daily Event Count for 3D Station Setup', 
                  rootouts + 'thesis/eventcount_3d_full.png')

plot_event_counts_full(df_3d_full_cnts_notremor_ep1, df_3d_full_cnts_notremor_ep2, None, 
                  'Daily Event Count for 3D Station Setup (No Tremor)', 
                  rootouts + 'thesis/eventcount_3d_full_notremor.png')

# ---------------------------------------------------------------------------------------
def plot_event_counts_comparison(df1_ep1, df1_ep2, df1_ep3, df2_ep1, df2_ep2, df2_ep3,
                                 title, filename, shortlegendstr1, shortlegendstr2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Plot first dataset (background)
    ax.plot(df1_ep1.index, df1_ep1.values, label=shortlegendstr1+' - Episode 1', color='tab:blue', alpha=0.5)
    ax.plot(df1_ep2.index, df1_ep2.values, label=shortlegendstr1+' - Episode 2', color='tab:orange', alpha=0.5)
    if df1_ep3 is not None:
        ax.plot(df1_ep3.index, df1_ep3.values, label=shortlegendstr1+' - Episode 3', color='tab:green', alpha=0.5)
    
    # Plot second dataset (foreground)
    ax.plot(df2_ep1.index, df2_ep1.values, label=shortlegendstr2+' - Episode 1', color='tab:blue')
    ax.plot(df2_ep2.index, df2_ep2.values, label=shortlegendstr2+' - Episode 2', color='tab:orange')
    if df2_ep3 is not None:
        ax.plot(df2_ep3.index, df2_ep3.values, label=shortlegendstr2+' - Episode 3', color='tab:green')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Event Count')
    myplot(axoi=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Example usage:
plot_event_counts_comparison(df_bbd_full_cnts_ep1, df_bbd_full_cnts_ep2, df_bbd_full_cnts_ep3, 
                             df_bbd_full_cnts_notremor_ep1, df_bbd_full_cnts_notremor_ep2, df_bbd_full_cnts_notremor_ep3, 
                             'Daily Event Count Comparison for Broadband Station Setup', 
                             rootouts + 'thesis/eventcount_bbd_full_comparison.png',
                             'tremor included', 'tremor excluded')

plot_event_counts_comparison(df_3d_full_cnts_ep1, df_3d_full_cnts_ep2, None, 
                             df_3d_full_cnts_notremor_ep1, df_3d_full_cnts_notremor_ep2, None, 
                             'Daily Event Count Comparison for 3D Station Setup', 
                             rootouts + 'thesis/eventcount_3d_full_comparison.png',
                             'tremor included', 'tremor excluded')
# ---------------------------------------------------------------------------------------
def plot_event_counts_comparison_ep1_ep2(df1_ep1, df1_ep2, df2_ep1, df2_ep2, df3_ep1, df3_ep2, df4_ep1, df4_ep2,
                                         title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot first dataset (bbd tremor excluded)
    ax.plot(df1_ep1.index, df1_ep1.values, color='tab:blue', label='tremor excluded - Broadband')
    ax.plot(df1_ep2.index, df1_ep2.values, color='tab:blue')
    
    # Plot second dataset (bbd tremor included)
    ax.plot(df2_ep1.index, df2_ep1.values, color='tab:blue', alpha=.5, label='tremor included')
    ax.plot(df2_ep2.index, df2_ep2.values, color='tab:blue', alpha=.5)
    
    # Plot third dataset (3D tremor excluded)
    ax.plot(df3_ep1.index, df3_ep1.values, color='tab:orange', label='tremor excluded - 4.5 Hz geophone')
    ax.plot(df3_ep2.index, df3_ep2.values, color='tab:orange')
    
    # Plot fourth dataset (3D tremor included)
    ax.plot(df4_ep1.index, df4_ep1.values, color='tab:orange', alpha=.5, label='tremor included')
    ax.plot(df4_ep2.index, df4_ep2.values, color='tab:orange', alpha=.5)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Event Count')
    myplot(axoi=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

# Example usage:
plot_event_counts_comparison_ep1_ep2(df_bbd_full_cnts_notremor_ep1, df_bbd_full_cnts_notremor_ep2, 
                                     df_bbd_full_cnts_ep1, df_bbd_full_cnts_ep2, 
                                     df_3d_full_cnts_notremor_ep1, df_3d_full_cnts_notremor_ep2, 
                                     df_3d_full_cnts_ep1, df_3d_full_cnts_ep2, 
                                     'Daily Event Count Comparison for Broadband and 4.5 Hz Geophone Station Setup', 
                                     rootouts + 'thesis/eventcount_full_comparison_bbd_3D_ep1_ep2.png')


# compute magnitudes again but for tsor cleansed catalogs
# compare magnitudes cleansed vs magnitudes not cleansed vs patricks sets
# compute catalog for second station set
# do all the configurations
# two frequency analysis for all configurations

