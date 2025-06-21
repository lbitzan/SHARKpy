import  numpy                as np
import  pandas               as pd
import  multiprocessing      as mp
import  matplotlib.pyplot    as plt
from    obspy                import read, Trace
from    obspy.core           import Stream
from    kav_init             import rootdata, kav_angle, hyp_dist, flag_matching, catalog_array
from    tqdm                 import tqdm
from    datetime             import timedelta
import  matplotlib.pyplot    as plt
import  matplotlib           as mpl
import  os                   as os
from kavutil import read_catalog, read_catalogs_from_files, select_events, retrieve_trace_data, select_events_v2, myplot
from instrument_restitution import *

listofcatalogs = os.listdir(rootdata+'/results/cata_9.0_3D.tbreaktest.lb/')
print(listofcatalogs)

from multiprocessing import Pool
from functools import partial

def run_mlv_parallel(catalog):
    _,_ = run_Mlv_v2(
            catalogfile=rootdata + '/results/cata_9.0_3D.tbreaktest.lb/' + catalog,
            datetime_start='2023-02-01',
            datetime_end='2023-08-01',
            distance=None,
            stationdir='KAV11',
            stationid='c0941',
            drop_details=False,
            add_2_catalog=True)

with Pool(mp.cpu_count()) as pool:
    pool.map(run_mlv_parallel, listofcatalogs)