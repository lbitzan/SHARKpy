import numpy    as np
import pandas   as pd
import warnings
import datetime

import matplotlib.pyplot as plt

from   typing   import Union, Tuple
from   tqdm import tqdm

from   scipy    import linalg

from obspy.clients.fdsn import Client
from obspy import read_events, UTCDateTime, Inventory
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.core.event import (
    Magnitude, StationMagnitude, StationMagnitudeContribution, 
    ResourceIdentifier, Catalog, Event, Amplitude, Arrival)

from kav_init import rootdata, cataloguefile, newcataoutput, kav_angle
import multiprocessing as mp



def read_catalogs_from_files(catalog_1: str, catalog2: str):
    """
    Read two catalogs from files and return them as pandas DataFrames.
    Parameters:
    -----------
    catalog_1: str
        Name of the first catalog file. Individual calibration for shark results.
    catalog_2: str
        Name the second catalog file. Individual calibration for array results.
    Returns:
    --------
    df1: pd.DataFrame
        DataFrame containing the first catalog.
    df2: pd.DataFrame
        DataFrame containing the second catalog.
    events1: pd.Series
        Series containing the events of the first catalog.
    events2: pd.Series
        Series containing the events of the second catalog.

    """
    from kav_init import rootdata, kav_angle
    catalogfile, catalogfile_array = catalog_1, catalog2
    # Import shark catalog
    headers             = ['date','ratio','shark']
    catalog             = pd.read_csv(rootdata+'/'+catalogfile, header=0, sep='\s+', names=headers) #parse_dates=['date'])
    catalog['puretime'] = pd.to_datetime(catalog.index + ' ' + catalog.date, format='%d-%b-%Y %H:%M:%S,')
    catalog.drop(columns=['date','ratio','shark'], inplace=True)
    catalog.set_index(np.arange(len(catalog)),inplace=True)

    # Import array catalog
    headers_arr         = ['dstart','tstart','dend','tend','baz','v_app','rmse']
    catalog_arr         = pd.read_csv(rootdata+'/'+catalogfile_array, header=0, sep='\s+', names=headers_arr)
    
    catalog_arr.drop(catalog_arr[catalog_arr.baz <= kav_angle[0] ].index, inplace=True)
    catalog_arr.drop(catalog_arr[catalog_arr.baz >= kav_angle[1] ].index, inplace=True)
    
    catalog_arr['puretime']     = pd.to_datetime(catalog_arr.dstart + ' ' + catalog_arr.tstart, format='%Y-%m-%d %H:%M:%S.%f')
    catalog_arr.drop(columns    =['dstart','tstart','dend','tend','v_app','rmse'], inplace=True)
    
    df1 = pd.DataFrame({'event':    catalog_arr['puretime']}).set_index(catalog_arr.index)
    df2 = pd.DataFrame({'event':    catalog['puretime']    }).set_index(catalog.index)

    events1, events2 = df1.event, df2.event

    return df1, df2, events1, events2

# --------------------------------------------------------------------------------
# ---  Set up multiprocessing via parallel computing on two kernels

def process_data(procn, events1, events2, df1, df2, k_ini, return_pdDF, time_tolerance:int = 4, to_ns: int=1000000000, ):
    k = k_ini
    matching_events = pd.DataFrame(columns=['event1_idx', 'event1time', 'event2_idx', 'time_diff'])
    for i, event1 in enumerate(events1):
        for j, event2 in enumerate(events2[k:]):
            time_diff = np.abs(event1 - event2)
            if time_diff < np.timedelta64(time_tolerance * to_ns, 'ns'):
                matching_events = matching_events._append(
                    pd.DataFrame([[df1.index[i + k_ini],
                                    df1.values[i + k_ini],
                                    df2.index[k + j],
                                    time_diff]],
                                    columns=['event1_idx', 'event1time', 'event2_idx', 'time_diff']),
                    ignore_index=True)
                k = k + j
                # print('Match found! k = ' + str(k) + '; i = ' + str(i) + '; j = ' + str(j))

    return_pdDF = return_pdDF._append(matching_events, ignore_index=True)
    # return matching_events


def parallel_process(events1, events2, time_tolerance, df1, df2):
    num_events = len(events1)
    half = num_events // 2

    return_df = pd.DataFrame()
    # Create two separate processes to compute first and second half of the data
    process1 = mp.Process(target=process_data, args=(0, events1[:half], events2, time_tolerance, df1, df2, 0,    return_df))
    process2 = mp.Process(target=process_data, args=(1, events1[half:], events2, time_tolerance, df1, df2, half, return_df))
    # Start the processes
    process1.start()
    process2.start()

    # Wait for the processes to finish
    process1.join()
    process2.join()

       
    # Get the results from the processes
    matching_events1 = process1.return_value
    matching_events2 = process2.return_value

    # Merge the results into a single DataFrame
    matching_events = pd.concat([matching_events1, matching_events2], ignore_index=True)

    return matching_events


# __name__ = "__main__"
if __name__ == "__main__":
    df1, df2, events1, events2 = read_catalogs_from_files('catalogue_4.0_lb.txt','catalog_pl.txt')
    matching_events            = parallel_process(events1=events1, events2=events2, time_tolerance=int(4), df1=df1, df2=df2)
    