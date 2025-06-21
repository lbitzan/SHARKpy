import numpy    as np
import pandas   as pd
import warnings

import matplotlib.pyplot as plt

from   typing   import Union, Tuple

from   progressbar import ProgressBar

from   scipy    import linalg

from obspy.clients.fdsn import Client
from obspy import read_events, UTCDateTime, Inventory
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.core.event import (
    Magnitude, StationMagnitude, StationMagnitudeContribution, 
    ResourceIdentifier, Catalog, Event, Amplitude, Arrival)

def find_matching_events(
    catalog_1: Catalog, 
    catalog_2: Catalog, 
    time_difference: float = 5.0,
    epicentral_difference: float = 20.0,
    depth_difference: float = 40.0,
    magnitude_type: str = None,
) -> dict:
    """
    Find matching events between two catalogs.

    Parameters
    ----------
    catalog_1
        A catalog to compare to catalog_2
    catalog_2
        A catalog to compare to catalog_1
    time_difference
        Maximum allowed difference in origin time in seconds between matching 
        events
    epicentral_difference
        Maximum allowed difference in epicentral distance in km between 
        matching events
    depth_difference
        Maximum allowed difference in depth in km between matching events.
    magnitude_type
        Magnitude type for comparison, will only return events in catalog_1
        with this magnitude

    Returns
    -------
    Dictionary of matching events ids. Keys will be from catalog_1.
    """
    df_1 = summarize_catalog(catalog_1, magnitude_type=magnitude_type) # -> pd.DataFrame of catalog event_id=event_ids, origin_time=origin_times
    df_2 = summarize_catalog(catalog_2)
    if len(df_2) == 0:
        return None

    swapped = False
    if len(df_1) > len(df_2):
        # Flip for efficiency, will loop over the shorter of the two and use 
        # more efficient vectorized methods on the longer one
        df_1, df_2  = df_2, df_1
        swapped     = True

    timestamp           = min(min(df_1.origin_time), min(df_2.origin_time))
    comparison_times    = np.array([t - timestamp for t in df_2.origin_time])

    print("Starting event comparison.")
    bar = ProgressBar(max_value=len(df_1))
    matched_quakes = dict()
    for i in range(len(df_1)):
        origin_time     = UTCDateTime(df_1.origin_time[i])
        origin_seconds  = origin_time - timestamp
        deltas          = np.abs(comparison_times - origin_seconds)
        index           = np.argmin(deltas)
        delta           = deltas[index]
        if delta > time_difference:
            continue  # Too far away in time
        depth_sep       = abs(df_1.depth[i] - df_2.depth[index]) / 1000.0  # to km
        if depth_sep > depth_difference:
            continue  # Too far away in depth
        # distance check
        dist, _, _ = gps2dist_azimuth(
            lat1    = df_2.latitude[index],
            lon1    = df_2.longitude[index],
            lat2    = df_1.latitude[i],
            lon2    = df_1.longitude[i])
        dist /= 1000.
        if dist > epicentral_difference:
            continue  # Too far away epicentrally
        matched_id = df_2.event_id[index]
        if matched_id in matched_quakes.keys():
            # Check whether this is a better match
            if delta > matched_quakes[matched_id]["delta"] or\
                 dist > matched_quakes[matched_id]["dist"] or\
                 depth_sep > matched_quakes[matched_id]["depth_sep"]:
                continue  # Already matched to another, better matched event
        matched_quakes.update(
            {matched_id: dict(
                delta=delta, dist=dist, depth_sep=depth_sep, 
                matched_id=df_1.event_id[i])})
        bar.update(i + 1)
    bar.finish()
    
    # We just want the event mapper
    if not swapped:
        return {key: value["matched_id"] 
                for key, value in matched_quakes.items()}
    else:
        return {value["matched_id"]: key 
                for key, value in matched_quakes.items()}