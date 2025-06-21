import numpy            as np
import pandas           as pd
import multiprocessing  as mp
import warnings
import datetime

import matplotlib.pyplot as plt
import matplotlib       as mlp

from   typing   import Union, Tuple
from   tqdm     import tqdm
from   scipy    import linalg

from obspy.clients.fdsn import Client
from obspy import read_events, UTCDateTime, Inventory
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.core.event import (
    Magnitude, StationMagnitude, StationMagnitudeContribution, 
    ResourceIdentifier, Catalog, Event, Amplitude, Arrival)
from datetime import timedelta
from kavutil import myplot, add_colorbar_outside
from kav_init import rootdata, cataloguefile, newcataoutput, kav_angle
import params_plot as pplt


def check_catalog_similarity(
        catalog1:       str,
        catalog2:       str,
        time_tolerance: int = 4,
        to_ns:          int = 1000000000,
        flag_cluster:   bool = False,
        degree360:      bool = False):
    """
    Check if two catalogs are similar by comparing the number of events in catalog 1 which are listed in catalog 2.

    Parameters
    ----------
    catalog1 : string
        First catalog to compare.
    catalog2 : string
        Second catalog to compare.
    time_tolerance : float, optional
        Maximum allowed time difference between events in both catalogs to be recognised as the same. In seconds, by default 2.0 s.

    Returns
    -------
    Dictionary
        Dictionary containing matching events. Keys are of catalog 1.
    """
    # --- Import shark catalog
    catalogfile, catalogfile_array = catalog1, catalog2
    headers             = ['date','ratio','shark']
    catalog             = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\s+', names=headers)
    catalog['puretime'] = pd.to_datetime(catalog.index + ' ' + catalog.date, format='%d-%b-%Y %H:%M:%S,')
    catalog.drop(columns=['date','ratio','shark'], inplace=True)
    catalog.set_index(np.arange(len(catalog)),inplace=True)

    # --- Import array catalog
    if flag_cluster == True:
        # headers_arr         ['starttime','endtime','baz','v_app','rmse','cluster']
        catalog_arr         = pd.read_pickle(rootdata+'/'+catalogfile_array)
    else:
        headers_arr         = ['dstart','tstart','dend','tend','baz','v_app','rmse']
        catalog_arr         = pd.read_csv(rootdata+'/'+catalogfile_array, header=0, sep='\s+', names=headers_arr)
    



    if degree360 == False & flag_cluster == False:
        catalog_arr.drop(catalog_arr[catalog_arr.baz <= kav_angle[0] ].index, inplace=True)
        catalog_arr.drop(catalog_arr[catalog_arr.baz >= kav_angle[1] ].index, inplace=True)
    elif degree360 == False & flag_cluster == True:
        catalog_arr.drop(catalog_arr[catalog_arr.BAZ <= kav_angle[0] ].index, inplace=True)
        catalog_arr.drop(catalog_arr[catalog_arr.BAZ >= kav_angle[1] ].index, inplace=True)
    else:
        print('all events taken')
    
    if flag_cluster == True:
        catalog_arr['puretime'] = catalog_arr.Starttime.copy()
        catalog_arr.drop(columns=['Starttime','Endtime','v_app','rmse'], inplace=True)
    else:
        catalog_arr['puretime'] = pd.to_datetime(catalog_arr.dstart + ' ' + catalog_arr.tstart, format='%Y-%m-%d %H:%M:%S.%f')
        catalog_arr.drop(columns=['dstart','tstart','dend','tend','v_app','rmse'], inplace=True)

    if flag_cluster == True:
        df1 = pd.DataFrame({'event':    catalog_arr['puretime'],
                            'cluster':  catalog_arr['Clusters']}).set_index(catalog_arr.index)
    else:
        df1 = pd.DataFrame({'event':    catalog_arr['puretime']}).set_index(catalog_arr.index)
    df2 = pd.DataFrame({'event':    catalog['puretime']    }).set_index(catalog.index)

    events1, events2 = df1.event, df2.event
    # events1, events2 = df1.event.to_numpy(), df2.event.to_numpy()
    
    # swap the catalogs if the first catalog is larger
    swapped = False
    if len(df2) < len(df1):
        df1, df2    = df2, df1
        swapped     = True

    # Check regarding clusters
    if flag_cluster == True:
        matching_events = pd.DataFrame(columns=['event1_idx', 'event1time' ,'event2_idx','time_diff','cluster'])    
        k               = 0
        for i, event1 in enumerate(tqdm(events1)):
            # print(i, event1)
            # break
            for j, event2 in enumerate(events2[k:]):
                time_diff = np.abs(event1 - event2)
                if time_diff < np.timedelta64(time_tolerance * to_ns, 'ns'):
                    matching_events = matching_events._append(
                        pd.DataFrame(         [[df1.iloc[i].name,
                                                np.datetime_as_string(np.datetime64(df1.values[i][0])),
                                                df2.index[k+j],
                                                time_diff,
                                                df1.cluster.iloc[i]]],
                                    columns = ['event1_idx',
                                               'event1time',
                                               'event2_idx',
                                               'time_diff',
                                               'cluster']),
                                    ignore_index=True)
                    k = k + j
                    print('Match found! k = ' + str(k) + '; i = ' + str(i) + '; j = ' + str(j))
    else:
        matching_events = pd.DataFrame(columns=['event1_idx', 'event1time' ,'event2_idx','time_diff'])    
        k               = 0
        for i, event1 in enumerate(tqdm(events1)):
            for j, event2 in enumerate(events2[k:]):
                time_diff = np.abs(event1 - event2)
                if time_diff < np.timedelta64(time_tolerance * to_ns, 'ns'):
                    matching_events = matching_events._append(
                        pd.DataFrame(         [[df1.index[i],   np.datetime_as_string(df1.values[i][0]),  df2.index[k+j],  time_diff ]],
                                    columns = ['event1_idx',    'event1time',                             'event2_idx',    'time_diff']),
                                    ignore_index=True)
                    k = k + j
                    print('Match found! k = ' + str(k) + '; i = ' + str(i) + '; j = ' + str(j))



    # Save results in catalog
    if flag_cluster == True:
        matching_events.to_csv(rootdata + '/results/matching_test_ttol_cluster_'+str(time_tolerance)+'sec_cluster.txt', index=False, sep='\t')
    else:
        matching_events.to_csv(rootdata + '/results/matching_test_ttol_ratiorange40_'+str(time_tolerance)+'sec.txt', index=False, sep='\t')
    
    return matching_events



def read_cluster_catalog(catalogname: str = 'matching_test_ttol_cluster_10sec.txt'):
    if catalogname == '*.pkl':
        df = pd.read_pickle(rootdata + '/'+ catalogname)
        return df
    elif catalogname == '*.txt':
        headers = ['event1_idx','event1time','event2_idx','time_diff','cluster']
        df      = pd.read_csv(rootdata + '/'+ catalogname, header=0, sep='\t')#, names=headers)
        return df
    else:
        print('No supportted file format given! Try again with *.pkl or *.txt')



def plot_val_events_baz(catalogname:    str= '/results/matching_test_ttol_ratiorange50_15sec.txt',
                        refcata1:       str='catalogue_5.0_amplituderatio_50_lb.txt',
                        refcata2:       str='catalog_pl.txt'):
    
    cname0  = rootdata + catalogname
    cname1  = rootdata + '/'+ refcata1
    cname2  = rootdata + '/'+ refcata2

    headers0            = ['idx_cata2','eventtime','idx_cata1','tdiff']
    cata0               = pd.read_csv(cname0, header=0, sep='\t', names=headers0)
    cata0['eventtime']  = pd.to_datetime(cata0['eventtime'])
    cata0.drop(cata0[cata0['eventtime'].dt.month == 5 ].index, inplace=True) # drop events during low battery phase


    headers1            = ['date','ratio','shark']
    cata1               = pd.read_csv(cname1, header=0, sep='\s+', names=headers1)
    cata1['puretime']   = pd.to_datetime(cata1.index + ' ' + cata1.date, format='%d-%b-%Y %H:%M:%S,')
    cata1.drop( columns=['date','ratio','shark'], inplace=True)
    cata1.set_index( np.arange(len(cata1)),inplace=True)
    cata1.drop( cata1[ cata1['puretime'].dt.month == 5].index, inplace=True)

    headers2         = ['dstart','tstart','dend','tend','baz','v_app','rmse']
    cata2         = pd.read_csv(cname2, header=0, sep='\s+', names=headers2)
    cata2['puretime'] = pd.to_datetime(cata2.dstart + ' ' + cata2.tstart, format='%Y-%m-%d %H:%M:%S.%f')
    cata2.drop( columns=['dstart','tstart','dend','tend','v_app','rmse'], inplace=True)
    cata2.drop(cata2[cata2['puretime'].dt.month == 5].index, inplace=True)

    no0 = len(cata0)
    no1 = len(cata1)
    no2 = len(cata2)

    idxnew          = cata0['idx_cata2'].copy()
    cata0['baz']    = cata2['baz'][idxnew].values.copy()
    cinput          = cata2['baz'][idxnew].values.copy()

    for i in range(len(idxnew)):
        if cinput[i] >= 0:
            if cinput[i] <= 180-125:
                cinput[i] =  cinput[i] + 125
            else:
                cinput[i] = (180 - cinput[i] +180-125)*-1
        elif cinput[i] < 0 :
            cinput[i] = cinput[i] + 125


    # numbers

    # drop events from array catalog outside aperture
    cnew = cata2.copy()
    cnew.drop(cnew[cnew.baz <= kav_angle[0] ].index, inplace=True)
    cnew.drop(cnew[cnew.baz >= kav_angle[1] ].index, inplace=True)
    ncnew = len(cnew)

    # drop events fused catalog correponding to array aperture
    c0new = cata0.copy()
    c0new.drop(c0new[c0new.baz <= kav_angle[0] ].index, inplace=True)
    c0new.drop(c0new[c0new.baz >= kav_angle[1] ].index, inplace=True)
    no0new = len(c0new)
    
    # transform results 0-360° domain
    cata0.loc[cata0['baz'] < 0, 'baz'] = 360 + cata0['baz']
    c0new.loc[c0new['baz'] < 0, 'baz'] = 360 + c0new['baz']

    # print results in-line
    print('Of '+str(no1)+' events for kava idx and '+str(no2)+ ' for array analysis, '+ str(no0) +' events were in common!')
    print(' They show a mean backazimuth of '+ str(cata2['baz'][idxnew].mean()))
    print(' The amount of common events within the critical angle is '+str(no0new)+' against to '+str(no0) + ' ttl common events or '+ str(ncnew)+' total array events within the angle')

    ax = plt.scatter(cata0['eventtime'], cata0['baz'],#cata2['baz'][idxnew],
                c = 'tab:blue',
                alpha=0.1,
                label='no data in may\n mean = '+str(cata0['baz'].mean()) +'°,\n std = '+str(cata0['baz'].std())+'°')
    
    ax = plt.suptitle('Common events for both detections methods\npick +- 10 seconds')
    ax = plt.xlabel('Time')
    ax = plt.tick_params(axis='x',labelrotation = 35)
    ax = plt.ylabel('Backzimuth [° degree]')
    ax = plt.grid(color = pplt.gridcolor, alpha = pplt.gridalpha)
    ax = plt.legend(loc=pplt.legloc, fontsize=pplt.legfontsize)
    axoi = plt.gca()
    axoi.set_facecolor(pplt.facecolor)
    plt.tight_layout()
    plt.show()

    ax1 = plt.scatter(c0new['eventtime'], c0new['baz'],
                      c='tab:blue',
                      s= 20,
                      alpha= 0.1,
                      label='no data in may\n mean = '+str(c0new['baz'].mean())+'°,\n std = '+str(c0new['baz'].std())+'°')
    ax1 = plt.suptitle('Distribution of common events wthin critical angular range, \n pick +- 10 seconds')
    ax1 = plt.xlabel('Time')
    ax1 = plt.tick_params( axis='x', labelrotation = 35)
    ax1 = plt.ylabel('Backzimuth [° degree]')
    
    ax1 = plt.grid(color = pplt.gridcolor, alpha = pplt.gridalpha)
    ax1 = plt.legend(loc=pplt.legloc, fontsize=pplt.legfontsize)
    ax1oi = plt.gca()
    ax1oi.set_facecolor(pplt.facecolor)
    plt.tight_layout()
    plt.show()

def analyse_cluster(matching_events_w_clusters: pd.DataFrame):
    """
    Analyse the matching events with clusters.

    Parameters
    ----------
    matching_events_w_clusters : pd.DataFrame
        DataFrame containing the matching events with clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the analysis of the matching events with clusters.
    """
    df      = matching_events_w_clusters.copy()
    df_new  = df[['event1time', 'cluster']].copy()
    df_new  = df_new.groupby('cluster').agg({'event1time': lambda x: list(x), 'cluster': 'size'}).rename(columns={'cluster': 'ntries'})

    # fig, ax = plt.subplots()
    # for i in range(len(df_new)):
    #     ax.plot(df_new.iloc[i].event1time, df_new.iloc[i].ntries, label='Cluster ' + str(df_new.index[i]))
    
    fig, axes = plt.subplots(len(df_new), 1, sharex=True, figsize=(10, 5*len(df_new)))
    for i, (cluster, data) in enumerate(df_new.iterrows()):
        ax = axes[i]
        ax.scatter(pd.to_datetime(data['event1time']), np.ones_like(data['event1time']), label= str(data['ntries']) + ' events')#, color='tab:red')
        # ax.fill_between(data['event1time'], 0, 1, label= str(data['ntries']) + ' events', color='tab:blue', alpha=0.5)
        # for j in range(len(data['event1time'])):
        #     ax.broken_barh([(data['event1time'][j], timedelta(seconds=10))], (0, 1), facecolors='tab:blue')  # , label= str(data['ntries']) + ' events')
        ax.set_ylabel(f'Cluster {cluster} ')# \n {data["ntries"]} events')
        # ax.grid(True)
        myplot(axoi=ax) 

    fig.subplots_adjust(hspace=0.0) # plt.subplots_adjust(hspace=0.0)
    plt.xlabel('Event Time')
    plt.tight_layout()
    plt.show()

def analyse_match_in_cluster(catalogname: str, catalog2: str):
    lw = 3
    df          = read_cluster_catalog(catalogname)
    df_arr      = pd.read_pickle(rootdata+'/'+catalog2)

    df["baz"]   = (df_arr["BAZ"][df["event1_idx"]].values).copy()
    df['vapp']  = (df_arr['v_app'][df['event1_idx']].values).copy()

    df.loc[df['baz'] < 0, 'baz'] = 360 + df['baz']
    df.loc[df['baz'] < 0, 'baz'] = 360 + df['baz']

    df_gr = df[['event1_idx','event1time', 'baz', 'vapp', 'time_diff', 'cluster']].copy()
    df_gr = df_gr.groupby('cluster').agg({'event1time':     lambda x: list(x),
                                            'event1_idx':   lambda x: list(x),
                                            'baz':          lambda x: list(x),
                                            'vapp':         lambda x: list(x),
                                            'time_diff':    lambda x: list(x),
                                            'cluster':      'size'}).rename(columns={'cluster': 'ntries'})
    
    df_gr           = df_gr[df_gr['ntries'] > 5]
    df_gr_baz_min   = np.min(df_gr['baz'].apply(lambda x: min(x)))
    df_gr_baz_max   = np.max(df_gr['baz'].apply(lambda x: max(x)))
    df_gr_vapp_min  = np.min(df_gr['vapp'].apply(lambda x: min(x)))
    df_gr_vapp_max  = np.max(df_gr['vapp'].apply(lambda x: max(x)))
    
    fig, axes = plt.subplots(len(df_gr), 1, sharex=True, figsize=(8, 2*len(df_gr)))
    for i in range(len(df_gr)):
        ax              = axes[i]
        cluster_data    = (df_gr.iloc[i]).copy()
        event_times     = pd.to_datetime(cluster_data['event1time'])
        baz_values      = cluster_data['baz']
        vapp_values     = cluster_data['vapp']
        
        # Set x-axis values
        x_start         = event_times
        x_end = x_start + timedelta(seconds=10)
        x_values = np.dstack((x_start, x_end))
        x_values = x_values.reshape(-1, 2)
        
        # Set y-axis values
        y_values = [0, 1]
        
        # Set color based on baz values
        normalized_baz  = (np.array(baz_values) - df_gr_baz_min) / (df_gr_baz_max - df_gr_baz_min)
        color_baz       = mlp.cm.viridis(normalized_baz)   

        # Plot fill_between
        cax = ax.vlines(x_start, 0, 1, color=color_baz, lw=lw, alpha=0.5,
                  label=str(cluster_data['ntries']) + ' events \n mean baz = ' + str(np.round(np.mean(baz_values),2)) + '° \n std baz = ' + str(np.round(np.std(baz_values),2)) + '°')
        myplot(axoi=ax)
        ax.set_ylabel(f'Cluster {cluster_data.name} ')
        ax.tick_params(axis='y', labelleft=False, left=False)
        cbar = add_colorbar_outside(cax, ax, return_cbar=True)
        cbar.set_label('Backazimuth [°]')
        cbar.ax.set_yticks([0, 1], [np.round(df_gr_baz_min), np.round(df_gr_baz_max)], rotation=45)

    axes[-1].set_xlabel('Event Time')
    axes[-1].tick_params(axis='x', labelrotation=45)
    fig.suptitle('Cluster Analysis') # plt.tight_layout()
    plt.subplots_adjust(hspace=0.0) 
    plt.savefig(rootdata + '/results/cluster_analysis_baz.png', dpi=300, bbox_inches='tight')
    plt.close() # plt.show()

    fig, axes = plt.subplots(len(df_gr), 1, sharex=True, figsize=(8, 2*len(df_gr)))
    for i in range(len(df_gr)):
        ax              = axes[i]
        cluster_data    = (df_gr.iloc[i]).copy()
        event_times     = pd.to_datetime(cluster_data['event1time'])
        baz_values      = cluster_data['baz']
        vapp_values     = cluster_data['vapp']
        
        # Set x-axis values
        x_start         = event_times
        x_end = x_start + timedelta(seconds=10)
        x_values = np.dstack((x_start, x_end))
        x_values = x_values.reshape(-1, 2)
        
        # Set y-axis values
        y_values = [0, 1]
        
        # Set color based on baz values
        normalized_vapp  = (np.array(vapp_values) - df_gr_vapp_min) / (df_gr_vapp_max - df_gr_vapp_min)
        color_vapp       = mlp.cm.viridis(normalized_vapp)   

        # Plot fill_between
        cax = ax.vlines(x_start, 0, 1, color=color_vapp, lw=lw, alpha=0.5,
                  label=str(cluster_data['ntries']) + ' events \n mean vapp = ' + str(np.round(np.mean(vapp_values),2)) + '° \n std vapp = ' + str(np.round(np.std(vapp_values),2)) + '°')
        myplot(axoi=ax)
        ax.set_ylabel(f'Cluster {cluster_data.name} ')
        ax.tick_params(axis='y', labelleft=False, left=False)
        cbar = add_colorbar_outside(cax, ax, return_cbar=True)
        cbar.set_label('Apparent Velocity [°]')
        cbar.ax.set_yticks([0, 1], [np.round(df_gr_vapp_min), np.round(df_gr_vapp_max)], rotation=45)

    axes[-1].set_xlabel('Event Time')
    axes[-1].tick_params(axis='x', labelrotation=45)
    fig.suptitle('Cluster Analysis') # plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(rootdata + '/results/cluster_analysis_vapp.png', dpi=300, bbox_inches='tight')
    plt.close() # plt.show()

if __name__ == '__main__':
    # check_catalog_similarity(catalog1='catalogue_4.0_lb.txt', catalog2='catalog_pl.txt', time_tolerance=15, degree360=True)
    # check_catalog_similarity(catalog1='catalogue_4.0_lb.txt', catalog2='catalog_pl.txt', time_tolerance=15, degree360=True)
    

    # check_catalog_similarity(catalog1='catalogue_4.0_lb.txt', catalog2='Cluster_df.pkl', time_tolerance=10, degree360=True, flag_cluster=True)

    plot_val_events_baz()

    # catalogname = 'results/matching_test_ttol_cluster_10sec_cluster.txt'
    # catalog2    = 'Cluster_df.pkl'
    # catalog1    = 'catalogue_4.0_lb.txt'

    # analyse_match_in_cluster(catalogname, catalog2)

