import  numpy                as np
import  pandas               as pd
import  multiprocessing      as mp
import  matplotlib.pyplot    as plt
from    obspy                import read, Trace
from    obspy.core           import Stream
from    kav_init             import rootdata
from    tqdm                 import tqdm
from    datetime             import timedelta
import  matplotlib.pyplot    as plt
import  matplotlib           as mpl
import  os                   as os
from    kavutil_2 import read_catalogs_from_files, select_events_v2, retrieve_trace_data, select_events, read_catalog, myplot

"""
Module for instrument restitution, ground motion computation and further calculation of magnitude.

"""


def geophone_tf(
        trace:  Trace,
        Gs:     float = 27.7,
        Cv:     float = 6.5574e7,
        k:      float = 1.0,

        p1 = -19.78 + 20.20j,
        p2 = -19.78 - 20.20j,
        z1 = 0.0,
        z2 = 0.0
):
    """
    Calculate the transfer function of a geophone.

    Parameters
    ----------
    Gs : float, optional
        Geophone sensitivity in V/m/s, by default 27.7.
    Cv : float, optional
        Geophone compliance in m/N, by default 6.5574e7.
    k : float, optional
        Geophone damping ratio, by default 1.0.
    p1 : complex, optional
        Pole 1, by default -19.78 + 20.20j.
    p2 : complex, optional
        Pole 2, by default -19.78 - 20.20j.
    z1 : complex, optional
        Zero 1, by default 0.0.
    z2 : complex, optional
        Zero 2, by default 0.0.

    Returns
    -------
    function
        Transfer function of the geophone.
    """


    signal = trace.data.copy()

    # fft of tapered signal
    signal_fft = np.fft.fft(signal)
    
    n  = len(signal)                                # length of signal
    dt = 1/trace.stats.sampling_rate                # sampling interval
    tf = np.ones_like(signal_fft)                   # initialise tf array

    # broadcast over positive frequencies to generate frequencies & calculate the transfer function
    idx         = np.array(range(n//2+1))
    tf_slice    = tf[idx]                           # include Nyquist frequency for even n
    f           = idx / (n*dt)                      # compute frequency at index i
    omega       = 2*np.pi*f                         # compute angular frequency
    s           = 1j*omega                          # compute complex frequency
    H           = k * s**2 / ((s - p1) * (s - p2))  # compute transfer function
    tf_slice    = H * Gs * Cv                       # consider sensitivities to retrieve output in counts
    tf[idx]     = tf_slice                          # fill in tf array

    transfer_function = tf

    return transfer_function

def apply_bb_tf_inv(trace:  Trace):
    """
    Remove the instrument transfer function (sensor + data logger) from a trace object containing counts.
    Input is a trace object in counts, output is a trace object in ground motion.
    Setup: Trillium compact 120 and DataCube3D

    Input:
    -------
    trace: Trace
        Trace object with unit counts.
    Gs: float, optional
        Geophone sensitivity in V/m/s, by default 27.7.
    Cv: float, optional
        Geophone compliance in m/N, by default 6.5574e7.
    k: float, optional
        Geophone damping ratio, by default 1.0.
    p1: complex, optional
        Pole 1, by default -19.78 + 20.20j.
    p2: complex, optional
        Pole 2, by default -19.78 - 20.20j.
    z1: complex, optional
        Zero 1, by default 0.0.
    z2: complex, optional
        Zero 2, by default 0.0.

    Output:
    -------
    Trace
        Trace object with unit m/s^2.

    """
    # --- Geophone parameters --- (if Trillium Compact 120 and DataCube3D are used, apply factor 1/10 at either Gs or Cv)
    Gs     = 754.3      # Geophone sensitivity in V/m/s (Trillium Compact 120: 754.3 V/(m/s))
    Cv     = 1.6393e6   # Geophone/Datalogger Sensitivity [counts/Volt] (DataCube3D: 1.6393e7 counts/V)
    k      = 1.0        # Geophone damping ratio
    a0     = 4.34493e17 # Normalisation factor
    
    p1, p2 = -0.03691 - 0.03702j, -0.03691 + 0.03702j
    p3     = -343.0
    p4, p5 = -370.0 - 467.0j, -370.0 + 467.0j
    p6, p7 = -836.0 - 1522.0j, -836.0 + 1522.0j
    p8, p9 = -4900.0 - 4700.0j, -4900.0 + 4700.0j
    p10    = -6900.0 + 0.0j
    p11    = -15000.0 + 0.0j
    
    z1, z2 = 0.0, 0.0
    z3, z4 = -392.0, -1960.0
    z5, z6 = -1490.0 + 1740.0j, -1490.0 - 1740.0j

    eps = 1e-6

    # --- Read trace data ---
    tr = trace.copy()
    tr.detrend('demean')
    tr.taper(max_percentage=0.1, type='hann')
    tr.detrend('demean')
    signal = tr.data.copy()

    # --- FFT of tapered signal ---
    signal_fft  = np.fft.fft(signal)
    n           = len(signal)
    dt          = 1/tr.stats.sampling_rate
    tf          = np.ones_like(signal_fft)

    # --- Transfer function ---
    idx         = np.array(range(n//2+1))   # Indexing positive frequencies
    tf_slice    = tf[idx]                   # Include Nyquist frequency for even n
    f           = idx / (n*dt)              # Frequency
    omega       = 2*np.pi*f                 # Angular frequency
    s           = 1j*omega                  # Complex frequency
    Hnum        = k * (s - z1) * (s - z2) * (s - z3) * (s - z4) * (s - z5) * (s - z6)
    Hden        = (s - p1) * (s - p2) * (s - p3) * (s - p4) * (s - p5) * (s - p6) * (s - p7) * (s - p8) * (s - p9) * (s - p10) * (s - p11)
    H           = Hnum / Hden               # Transfer function without sensitivities
    tf_slice    = H * Gs * Cv * a0          # Transfer function with sensitivities
    tf[idx]     = tf_slice                  # Fill in tf array

    # --- Apply transfer function ---
    signal_fft[idx]     = signal_fft[idx] / (tf[idx] + eps*np.max(np.abs(tf)))
    idx                 = np.array(range(1, n//2 +1)) # np.array(range(1, n//2))
    signal_fft[n-idx]   = np.conj(signal_fft[idx])
    signal_modified     = np.fft.ifft(signal_fft)

    # breakpoint()

    # --- Create new trace object ---
    tr_modified                     = Trace(data=signal_modified)
    tr_modified.stats.sampling_rate = tr.stats.sampling_rate
    tr_modified.stats.starttime     = tr.stats.starttime

    return tr_modified

def simulate_45Hz(trace: Trace):
    """
    Simulate the instrument response of a 4.5 Hz geophone, applied to a trace object in units velocity/s.
    Instrument: 3D geophone PE-6/B, 4.5 ... 500 Hz
    Datalogger: DataCube3D

    Input:
    -------
    trace: Trace
        Trace object with unit m/s.

    Output:
    -------
    Trace
        Trace object with unit counts.
    """
    # --- Geophone parameters ---
    Gs:     float       = 27.7,
    Cv:     float       = 6.5574e7,
    k:      float       = 1.0,
    p1, p2, z1, z2      = -19.78 + 20.20j, -19.78 - 20.20j, 0.0, 0.0

    # --- Read trace data ---
    tr = trace.copy()
    tr.detrend('demean')
    tr.taper(max_percentage=0.1, type='hann')
    tr.detrend('demean')
    signal = tr.data.copy()
    
    # --- FFT of tapered signal ---
    signal_fft  = np.fft.fft(signal)
    n           = len(signal)
    dt          = 1/tr.stats.sampling_rate
    tf          = np.ones_like(signal_fft)

    # --- Transfer function ---
    idx         = np.array(range(n//2+1))   # Indexing positive frequencies
    tf_slice    = tf[idx]                   # Include Nyquist frequency for even n
    f           = idx / (n*dt)              # Frequency
    omega       = 2*np.pi*f                 # Angular frequency
    s           = 1j*omega                  # Complex frequency
    H           = k * s**2 / ((s - p1) * (s - p2))  # Transfer function without sensitivities
    tf_slice    = H * Gs * Cv               # Transfer function with sensitivities
    tf[idx]     = tf_slice                  # Fill in tf array

    # --- Apply transfer function ---
    signal_fft[idx]     = signal_fft[idx] * (tf[idx]+ 1e-6*np.max(np.abs(tf)))
    idx                 = np.array(range(1, n//2 +1 ))
    signal_fft[n-idx]   = np.conj(signal_fft[idx])
    signal_modified     = np.fft.ifft(signal_fft)

    # --- Create new trace object ---
    tr_modified                     = Trace(data=signal_modified)
    tr_modified.stats.sampling_rate = tr.stats.sampling_rate
    tr_modified.stats.starttime     = tr.stats.starttime

    return tr_modified


def apply_geophone_tf(
        trace:  Trace
):
    """
    Calculate the transfer function of a geophone.

    Parameters
    ----------
    Gs : float, optional
        Geophone sensitivity in V/m/s, by default 27.7.
    Cv : float, optional
        Geophone compliance in m/N, by default 6.5574e7.
    k : float, optional
        Geophone damping ratio, by default 1.0.
    p1 : complex, optional
        Pole 1, by default -19.78 + 20.20j.
    p2 : complex, optional
        Pole 2, by default -19.78 - 20.20j.
    z1 : complex, optional
        Zero 1, by default 0.0.
    z2 : complex, optional
        Zero 2, by default 0.0.

    Returns
    -------
    function
        Transfer function of the geophone.
    """

    signal_tr       = trace.copy()
    signal_tr.taper(type='hann', max_percentage=0.1)# apply taper to signal
    signal          = signal_tr.data.copy()         # copy signal data

    signal_fft      = np.fft.fft(signal)            # fft of tapered signal
    
    n               = len(signal)                   # length of signal
    idx             = np.array(range(n//2+1))

    tf              = geophone_tf(trace)
    signal_fft[idx] = signal_fft[idx] * tf[idx]    # apply transfer function to signal

    del idx

    idx             = np.array(range(1,n//2))       # starts from 2 to exclude DC; handles both even and odd n
    signal_fft[n-idx] = np.conj(signal_fft[idx])    # fill in negative frequencies

    signal_modified = np.fft.ifft(signal_fft)       # inverse fft to get the filtered signal

    # create a new trace object with the filtered signal
    trace_modified      = trace.copy()
    trace_modified.data = signal_modified

    return trace_modified

def apply_geophone_tf_inv(
        trace:              Trace,
        transfer_function:  np.ndarray = None,
        eps:                float = 1e-06 
        ):
    
    """
    This function removes the instrument response from a trace object in unit [counts].
    By applying the inverse transfer function, the trace is converted to the physical unit [m/s].
    Ground velocity is obtained.

    Parameters
    ----------
    trace : Trace
        Trace object containing the data to be filtered.
    transfer_function : np.ndarray, optional
        Transfer function of the geophone, by default None.
    eps : float, optional
        Small value to prevent division by zero, by default 1.e9.
    
    Returns
    -------
    Trace
        Trace object containing the filtered signal.
    """
    tr          = trace.copy()                                  # copy trace object
    tr.detrend('demean')                                        # 1st demean   
    tr.taper(max_percentage=0.1, type='hann')                   # taper
    tr.detrend('demean')                                        # 2nd demean
    signal      = tr.data.copy()                                # copy signal data

    if transfer_function is None:                               # compute transfer function if not provided
        Gs:     float       = 27.7, # Geophone sensitivity in V/m/s
        Cv:     float       = 6.5574e7, # Digital compliance in counts/V
        k:      float       = 1.0,
        p1, p2, z1, z2      = -19.78 + 20.20j, -19.78 - 20.20j, 0.0, 0.0

        trace4tf            = trace.copy()
        trace4tf.detrend('demean')
        trace4tf.taper(max_percentage=0.1, type='hann') 
        signal4tf           = trace4tf.data.copy()

        signal_fft_4tf      = np.fft.fft(signal4tf)             # fft of tapered signal
        n4tf                = len(signal4tf)                    # length of signal
        dt                  = 1/trace.stats.sampling_rate       # sampling interval
        tf                  = np.ones_like(signal_fft_4tf)      # initialise tf array

        # broadcast over positive frequencies to generate frequencies & calculate the transfer function
        idx                 = np.array(range(n4tf//2+1))
        tf_slice            = tf[idx]                           # include Nyquist frequency for even n
        f                   = idx / (n4tf*dt)                   # compute frequency at index i
        omega               = 2*np.pi*f                         # compute angular frequency
        s                   = 1j*omega                          # compute complex frequency
        H                   = k * s**2 / ((s - p1) * (s - p2))  # compute transfer function
        tf_slice            = H * Gs * Cv                       # consider sensitivities to retrieve output in counts
        tf[idx]             = tf_slice                          # fill in tf array

        transfer_function   = tf

    # breakpoint()
    tf                              = transfer_function.copy()              # rename variable
    npts                            = len(signal)                           # length of signal
    idx                             = np.array(range(npts//2+1))            # indexing positive frequencies

    signal_fft                      = np.fft.fft(signal)                    # fft of tapered signal
    inv_tf                          = np.zeros_like(tf)                     # initialise inv tf
    inv_tf[idx]                     = 1. / (tf[idx]+ eps*np.max(np.abs(tf)))# compute inv tf

    signal_fft_restored             = signal_fft.copy()                     # initialise restored signal
    signal_fft_restored[idx]        = signal_fft[idx] * inv_tf[idx]         # apply inverse transfer function to positive frequencies
    del idx                                                                 # clear variable
    idx                             = np.array(range(1,npts//2 +1))         # starts from 2 to exclude DC; handles both even and odd n
    signal_fft_restored[npts-idx]   = np.conj(signal_fft_restored[idx])     # reconstruct full spectrum by filling in the negative frequencies using conjugate symmetry

    signal_restored                 = np.fft.ifft(signal_fft_restored)      # inverse fft to get the filtered signal

    tr_restored                     = Trace(data=signal_restored)
    tr_restored.stats.sampling_rate, tr_restored.stats.starttime = tr.stats.sampling_rate, tr.stats.starttime
    tr_restored.filter('highpass', freq=0.5, zerophase=True)

    return tr_restored

def hfilt_rmv_instr(
        trace:      Trace,
        freq:       int = 1,
        detrend:    str = None
):
    """
    High-pass filter to remove instrument response. Uses ObsPy's filter function, which bases on the scipy.signal implementation.

    Parameters
    ----------
    trace : Trace
        Trace object containing the data to be filtered.
    freq : float, optional
        Frequency to high-pass filter the data, by default 0.1.
    corners : int, optional
        Number of corners for the filter, by default 4.

    Returns
    -------
    Trace
        Trace object containing the filtered signal.
    """
    tr = trace.copy()

    if detrend == 'demean':
        tr.detrend('demean')
    elif detrend == 'linear':
        tr.detrend('linear')

    tr.filter('highpass', freq=freq)

    return tr

def compute_ground_motion(trace: Trace) -> Trace:
    """
    Compute the ground motion from a trace object with data in ground velocity (m/s).
    Therefore an integration is applied in frequency domain.
    Conversion: Ground velocity [m/s] -> Ground displacement [m]

    Parameters
    ----------
    trace : Trace
        Trace object containing the counts.

    Returns
    -------
    Trace
        Trace object containing the ground motion.
    """
    tr = trace.copy()
    tr.detrend('demean')
    tr.taper(max_percentage=0.05, type='hann') #,max_length=None)
    tr.detrend('demean')

    signal                  = tr.data.copy()                # Copy the signal data
    signal_fft              = np.fft.fft(signal)            # transform signal to frequency domain
    n                       = len(signal)                   # length of the signal
    idx                     = np.array(range(n // 2 + 1))   # indexing positive frequencies
    f                       = idx / (n * trace.stats.delta) # compute frequency at index i
    omega                   = 2 * np.pi * f                 # compute angular frequency
    s                       = 1j * omega                    # compute complex frequency
    signal_fft[idx]         = signal_fft[idx] / (s + 1e-9)  # apply division by "j*2*np.pi*f"
    del idx 

    idx                     = np.array(range(1, n // 2 +1))    # Starts from 2 to exclude DC; handles both even and odd n
    signal_fft[n - idx]     = np.conj(signal_fft[idx])      # Fill in negative frequencies using conjugate symmetry
    signal_modified         = np.fft.ifft(signal_fft)       # Inverse FFT to get the modified signal

    trace_ground_motion                         = Trace(data=signal_modified)   # Create a new trace object with the modified signal
    trace_ground_motion.stats.starttime         = tr.stats.starttime
    trace_ground_motion.stats.sampling_rate     = tr.stats.sampling_rate
    trace_ground_motion.filter('highpass', freq=1., zerophase=True)

    return trace_ground_motion

def calculate_Mlv(trace:        Trace,
                  distance:     float,
                  flag_obspy:   bool = False,
                  ex_WA_waveform: bool = False) -> float:
    """
    Calculate the vertical local magnitude (MLv) from a Trace object containing seismic event data.
    Implementation follows the equation from Bakun and Joyner (1984).

    Parameters
    ----------
    trace : Trace
        Trace object containing the seismic event data.
    distance : float
        Distance from the event to the seismograph station.
    flag_obspy : bool, optional
        Flag to use ObsPy's simulate_seismometer function, by default False.

    Returns
    -------
    float
        Vertical local magnitude (MLv) of the event.
    """
    from obspy.signal.invsim import simulate_seismometer
    import math
    tr                      = trace.copy()
    tr.detrend('demean'); tr.taper(max_percentage=0.05, type='hann'); tr.detrend('demean')

    p0, p1, z0, z1  = -5.49779 - 5.60886j, -5.49779 + 5.60886j, 0.0, 0.0

    

    # --- Wood-Anderson simulation via ObsPy ---
    if flag_obspy:
        signal_modified   = simulate_seismometer(
            tr.data.copy(),
            samp_rate               = tr.stats.sampling_rate,
            paz_remove              = None,
            paz_simulate            = {'poles': [-5.49779 - 5.60886j, -5.49779 + 5.60886j],
                                       'zeros': [0.+0j, 0j], 'gain': 1.0},
            remove_sensitivity      = False,
            simulate_sensitivity    = False,
            zero_mean               = False,
            taper                   = False,
            nfft_pow2               = True)
    else:
        signal                      = tr.data.copy()
        signal_fft                  = np.fft.fft(signal)
        tf_woodanderson             = np.zeros_like(signal_fft, dtype=complex)
        n                           = len(signal)
        idx                         = np.array(range(n//2+1))
        f                           = idx / (n * tr.stats.delta)
        omega                       = 2 * np.pi * f
        s                           = 1j * omega
        H                           = (s - z0) * (s - z1) / ((s - p0) * (s - p1))
        tf_woodanderson[idx]        = H
        signal_fft[idx]             = signal_fft[idx] * tf_woodanderson[idx]
        idx                         = np.array(range(1, n//2))
        signal_fft[n-idx]           = np.conj(signal_fft[idx])
        signal_modified             = np.fft.ifft(signal_fft)

    trace_WA                        = Trace(data=signal_modified)
    trace_WA.stats.sampling_rate    = tr.stats.sampling_rate
    trace_WA.stats.starttime        = tr.stats.starttime
    trace_WA.filter('highpass', freq=0.5, zerophase=True)
    trace_WA.detrend('demean')

    amplitude                       = max(abs(trace_WA.data))
    amplitude_nano                  = amplitude * 10**9

    # Calculate vertical local magnitude (MLv)
    mlv = math.log10(amplitude_nano) + 1.11 * math.log10(distance) + 0.0018 * distance - 2.09
    if ex_WA_waveform is True:
        return mlv, trace_WA
    else:
        return mlv

def run_Mlv(
        catalog_1:      str,
        catalog_2:      str,
        year:           int | list | None,
        month:          int | list | None,
        day:            int | list | None,
        hour:           int | list | None,
        minute:         int | list | None,
        stationdir:     str,
        stationid:      str,
        distance:       float | None,
        drop_details:   bool    = False
    ):

    from kav_init import hyp_dist, flag_matching
    from obspy import Stream
    from magnitude_estimation import apply_geophone_tf_inv, compute_ground_motion, calculate_Mlv

    df1, _                  = read_catalogs_from_files(catalog_1, catalog_2, drop_details=drop_details)
    selected_events         = select_events(df1, year, month, day, hour, minute)

    print(selected_events)

    stream_events           = retrieve_trace_data(selected_events, stationdir, stationid)
    stream_events_restored  = Stream()
    stream_events_gndmtn    = Stream()
    mlv_box                 = np.array([],dtype=float)
    if flag_matching == True:
        idx_catalog_array       = np.array(df1.index.copy(), dtype=int)

    if distance is None:
        distance = hyp_dist[stationdir]

    for i in range(len(stream_events)):

        tr_restored    = apply_geophone_tf_inv( stream_events[i])
        tr_gndmtn      = compute_ground_motion( tr_restored)
        mlv            = calculate_Mlv(         tr_gndmtn, distance)

        stream_events_restored.append(  tr_restored)
        stream_events_gndmtn.append(    tr_gndmtn)
        mlv_box        = np.append(     mlv_box, mlv)

    if len(mlv_box) > 100:
        print('More than 100 events selected. Only the MLv values and events are returned.')
        return mlv_box, selected_events, _, _, _
    else:
        return mlv_box, selected_events, stream_events, stream_events_gndmtn, stream_events_restored




def compare_sim45_vs_geophone(catalog_filename:     str,
                            datetime_start:       str,
                            datetime_end:         str,
                            stationdir_geophone:  str,
                            stationdir_bb:        str,
                            stationid_geophone:   str,
                            stationid_bb:         str,
                            savepath:             str=None):
    
    if savepath:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            print('Directory created:', savepath)
    
    if '.txt' in catalog_filename:
        catalog = read_catalog(catalog_filename)
    elif '.pkl' in catalog_filename:
        catalog = pd.read_pickle(catalog_filename)
    else:
        print('Catalog format not recognized. Must be .txt or .pkl.')
        return

    events = select_events_v2(catalog, datetime_start, datetime_end, asstring=True)
    stream = retrieve_trace_data(events, stationdir_bb, stationid_bb)
    stream_ref = retrieve_trace_data(events, stationdir_geophone, stationid_geophone)

    for i, _ in enumerate(tqdm(stream)):
        trace, trace_ref = stream[i].copy(), stream_ref[i].copy()

        trace_inv = apply_bb_tf_inv(trace);           trace_inv.stats.station = trace.stats.station
        trace_sim = apply_geophone_tf(trace_inv);     trace_sim.stats.station = trace.stats.station

        trace_inv_ref = apply_geophone_tf_inv(trace_ref); trace_inv_ref.stats.station = trace_ref.stats.station
        trace_inv2_ref = apply_geophone_tf(trace_inv_ref); trace_inv2_ref.stats.station = trace_ref.stats.station

        lw = .5
        time = pd.date_range(trace.stats.starttime.datetime, periods=trace.stats.npts, freq=str(trace.stats.delta)+'S')
        filedatestr = (time[0]+timedelta(seconds=10)).strftime('%Y%m%dT%H%M%S')

        fig, ax = plt.subplots(3, 2, figsize=[20, 8], sharex=True)
        fig.suptitle('Simulating 4.5 Hz geophone on broad band data \n'+ (trace.stats.starttime + 10).strftime('%Y-%m-%d %H:%M:%S'))
        ax[0,0].set_title('Reference 4.5 Hz geophone: '+ trace_ref.stats.station)
        ax[0,0].plot(time, trace_ref.data-np.mean(trace_ref.data),      'k', linewidth=lw, label='ref raw');    ax[0,0].set_ylabel('Counts')
        ax[1,0].plot(time, trace_inv_ref.data,                          'r', linewidth=lw, label='ref vel');    ax[1,0].set_ylabel('Velocity [m/s]')
        ax[2,0].plot(time, trace_inv2_ref.data,                         'b', linewidth=lw, label='ref counts'); ax[2,0].set_ylabel('Counts')
        ax[2,0].set_xlabel('Time [s]'); ax[2,0].tick_params(axis='x',labelrotation = 45)

        ax[0,1].set_title('Broadband station: '+ trace.stats.station)
        ax[0,1].plot(time, trace.data-np.mean(trace.data),              'k', linewidth=lw, label='raw')
        ax[1,1].plot(time, trace_inv.data,                              'r', linewidth=lw, label='vel')
        ax[2,1].plot(time, trace_sim.data,                              'b', linewidth=lw, label='counts')
        ax[2,1].set_xlabel('Time [s]'); ax[2,1].tick_params(axis='x',labelrotation = 45)
        [myplot(axoi=i) for i in ax.flatten()]; fig.align_labels(); plt.tight_layout()
        if savepath:
            plt.savefig(savepath+'/test_sim_45hzgeophone.compact.'+filedatestr+'_'+stationdir_bb+'_'+stationdir_geophone+'.png')
            plt.close()
        else:
            plt.show()

        del trace, trace_ref, trace_inv, trace_sim, trace_inv_ref, trace_inv2_ref, time
    
    return


def run_Mlv_v2(
        catalogfile:      str,
        datetime_start: str,
        datetime_end:   str,
        distance:       float | None,
        stationdir:     str='KAV11',
        stationid:      str='c0941',
        drop_details:   bool    = False,
        add_2_catalog:  bool    = False,
        batch_size:     int     = 300
    ):

    from kav_init import hyp_dist, bbd_ids
    from obspy import Stream
    from magnitude_estimation import apply_geophone_tf_inv, compute_ground_motion, calculate_Mlv

    if add_2_catalog:
        if '.txt' in catalogfile:
            catalogoutfilename = catalogfile.replace('.txt','.mlv.txt')
        elif '.pkl' in catalogfile:
            catalogoutfilename = catalogfile.replace('.pkl','.mlv.pkl')
    else:
        catalogoutfilename = None

    cata            = read_catalog(catalogfile)
    selected_events = select_events_v2(cata, datetime_start, datetime_end, asstring=True)
    catalog         = cata[cata.index.isin(selected_events.index)].copy()
    length_flag     = False
    if len(selected_events) < 100:
        length_flag = True

    print(catalog[:5])
    print(selected_events)
    mlv_box         = np.array([],dtype=float)
    mlv_times       = []

    if distance is None:
        distance = hyp_dist[stationdir]

    if length_flag:
        stream_events, stream_event_times = retrieve_trace_data(selected_events, stationdir, stationid)
        stream_events_restored  = Stream()
        stream_events_gndmtn    = Stream()

    batch_idx_start = np.arange(0, len(selected_events), batch_size)
    batch_idx_end   = np.append(batch_idx_start[1:], len(selected_events))
    print(batch_idx_start[:5],  batch_idx_end[:5])
    print(batch_idx_start[-5:], batch_idx_end[-5:])

    for i, _ in enumerate(tqdm(batch_idx_start)):
        print('Processing batch ', i+1, ' of ', len(batch_idx_start))
        print(selected_events.iloc[batch_idx_start[i]:batch_idx_end[i]])
        stream_events_batch, stream_event_times_batch = retrieve_trace_data(selected_events.iloc[batch_idx_start[i]:batch_idx_end[i]], stationdir, stationid)

        for j in range(len(stream_events_batch)):
            if stationdir in bbd_ids:
                tr_restored     = apply_bb_tf_inv( stream_events_batch[j])
            else:
                tr_restored     = apply_geophone_tf_inv(stream_events_batch[j])
            tr_gndmtn      = compute_ground_motion( tr_restored)
            mlv            = calculate_Mlv(         tr_gndmtn, distance)

            mlv_box        = np.append(mlv_box, mlv)
            mlv_times.append(stream_event_times_batch[j])
            if length_flag:
                stream_events_restored.append(  tr_restored)
                stream_events_gndmtn.append(    tr_gndmtn)

        print(mlv_box[-100:])
        print(len(mlv_box))

    if add_2_catalog:
        mlv_times = pd.to_datetime(np.array(mlv_times))
        magnitudes_and_dates = pd.DataFrame({
            'mlv': mlv_box,
            'Eventtime(UTC)': mlv_times
            })
        magnitudes_and_dates.set_index('Eventtime(UTC)', inplace=True)
        magnitudes_and_dates['mlv'] = magnitudes_and_dates['mlv'].astype(float)

        if 'puretime' in catalog.columns:
            tvar = 'puretime'
        elif 'date' in catalog.columns:
            tvar = 'date'
        elif 'Eventtime(UTC)' in catalog.columns:
            tvar = 'Eventtime(UTC)'
        else:
            raise ValueError('No datetime column found in DataFrame. Please provide a DataFrame with a "puretime", "date" or "Eventtime(UTC)" column.')
        # breakpoint()
        # Ensure both are datetime and have the same format for mapping
        catalog[tvar] = pd.to_datetime(catalog[tvar])
        magnitudes_and_dates.index = pd.to_datetime(magnitudes_and_dates.index)
        catalog['mlv'] = catalog[tvar].map(magnitudes_and_dates['mlv'])
        catalog.to_csv(catalogoutfilename, sep='\t', index=False)
        catalog.to_pickle(catalogoutfilename.replace('.txt','.pkl'))
        # breakpoint()
        return catalog, catalogoutfilename
    else:
        if length_flag:
            return mlv_box, [selected_events, stream_events, stream_events_gndmtn, stream_events_restored]
        else:
            return mlv_box, selected_events
    

def test_geophone_simulation(catalogfile: str,
                             datetime_start: str='2023-03-30T02:00:00',
                             datetime_end: str='2023-03-30T03:00:00',
                             broad_band_station_dir: str='KAV00',
                             broad_band_station_id: str='c0bdd',
                             localoutputdirectory: str = None,
                             save_plots: bool = False,
                             linewidth: float = 0.5):
    
    '''
    Test simulation of 4.5 Hz geophone on broad band data. 
    The function reads a catalog file, selects events within a given time window, retrieves the traces from a broad band station,
    applies the inverse transfer function of the geophone, and simulates the 4.5 Hz geophone on the broad band data.
    The function plots the raw broad band data, the ground motion, and the simulated 4.5 Hz geophone data.

    Parameters
    ----------
    catalogfile : str
        File containing the event catalog.
    datetime_start : str, optional
        Start time of the time window, by default '2023-03-30T02:00:00'.
    datetime_end : str, optional
        End time of the time window, by default '2023-03-30T03:00:00'.
    broad_band_station_dir : str, optional
        Directory of the broad band station, by default 'KAV00'.
    broad_band_station_id : str, optional
        Station ID of the broad band station, by default 'c0bdd'.
    localoutputdirectory : str, optional
        Local output directory for the plots, by default ''.
    save_plots : bool, optional
        Flag to save the plots, by default False.
    linewidth : float, optional
        Line width of the plots, by default 0.5.
    
    Returns
    -------
    None

    @ L.Bitzan, 2024
    '''

    from kavutil_2 import select_events_v2, read_catalog, define_ylim, retrieve_trace_data
    catalog = read_catalog(catalogfile)
    events  = select_events_v2(catalog, datetime_start, datetime_end, asstring=True)
    stream  = retrieve_trace_data(events, broad_band_station_dir, broad_band_station_id)

    for i, _ in enumerate(tqdm(stream)):
        trace       = stream[i].copy()
        trace_vis   = trace.detrend('demean')

        trace_inv   = apply_bb_tf_inv(trace)
        trace_sim   = simulate_45Hz(trace_inv)
        # trace_sim   = apply_geophone_tf(trace_inv)

        lw          = linewidth
        time        = pd.date_range(trace.stats.starttime.datetime, periods=trace.stats.npts, freq=str(trace.stats.delta)+'S')
        filedatestr = (time[0]+timedelta(seconds=10)).strftime('%Y%m%dT%H%M%S')

        fig, ax = plt.subplots(3, 1, figsize=[10, 8], sharex=True)
        fig.suptitle('Simulating 4.5 Hz geophone on broad band data \n'+ (trace.stats.starttime + 10).strftime('%Y-%m-%d %H:%M:%S') + '\n Broadband station: '+ broad_band_station_dir)
        # ax[0].set_title('Broadband station: '+ str(broad_band_station_dir))
        ax[0].plot(time, trace_vis.data, 'k',           linewidth=lw, label='bbd data')
        # ax[0].legend()
        ax[1].plot(time, trace_inv.data, 'tab:blue',    linewidth=lw, label='ground motion vel.')
        ax[2].plot(time, trace_sim.data, 'tab:orange',  linewidth=lw, label='simulated geophone')
        ax[2].set_xlabel('Time [s]')
        ax[2].tick_params(axis='x',labelrotation = 45)
        define_ylim(ax[0], trace_vis.data); define_ylim(ax[1], trace_inv.data); define_ylim(ax[2], trace_sim.data)
        [axi.grid() for axi in ax]; [axi.legend() for axi in ax]
        
        fig.align_labels()
        plt.tight_layout()
        if save_plots:
            if localoutputdirectory:
                plt.savefig(rootdata+'/results/'+localoutputdirectory+'/test_sim_45hzgeophone.'+filedatestr+'.png')
            else:
                plt.savefig(rootdata+'/results/test_sim_45hzgeophone.'+filedatestr+'.png')
            plt.close()
        else:
            plt.show()
        del trace, trace_inv, trace_sim, time




    return



if __name__ == '__main__':

    for cfile in os.listdir(os.path.join(rootdata,'results','compute_mags')):
        if cfile.endswith('.pkl'):
            cfilepath = os.path.join(rootdata,'results','compute_mags', cfile)
            print('Processing catalog file:', cfilepath)
            _, _ = run_Mlv_v2(cfilepath,
                            datetime_start='2023-02-01',
                            datetime_end='2023-12-01',
                            distance=None,
                            stationdir='KAV11',
                            stationid='c0941',
                            drop_details=False,
                            add_2_catalog=True,
                            batch_size=200)
            print('Catalog processed:', cfile)
            
            

 


