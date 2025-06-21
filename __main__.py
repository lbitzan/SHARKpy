import  numpy                as np
import  pandas               as pd
import  multiprocessing      as mp
import  matplotlib.pyplot    as plt
from    obspy                import read, Stream, Trace
from    kav_init             import rootdata, kav_angle, hyp_dist, flag_matching, catalog_array
import  matplotlib.pyplot    as plt
import  matplotlib           as mpl
import  os                   as os
from    kavutil              import myplot, add_colorbar_outside, select_events, read_catalogs_from_files, retrieve_trace_data
from    plutil               import plot_spectra


if __name__ == '__main__':

    catalog_1, catalog_2, drop_details = 'catalogue_4.0_lb.txt','catalog_pl.txt', False
    
    cases = {'a': (2023, 2,    9, None, None, 'KAV11','c0941',None),
             'b': (2023, 4, 5,     0, None, 'KAV11','c0941',None)}
            #  'c': (2023, 2, None, None, None, 'KAV04','c0939',None)}
            #  'd': (2023, 3, 30, None, None, 'KAV11','c0941',None)}


    for case in cases:
        cata,_          = read_catalogs_from_files(catalog_1, catalog_2, drop_details)
        selected_events = select_events(cata, cases[case][0], cases[case][1], cases[case][2], cases[case][3], cases[case][4])
        print(selected_events)
        
        plot_spectra(selected_events=selected_events, stationdir=cases[case][5], stationid=cases[case][6])