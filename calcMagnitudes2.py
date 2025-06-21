import  numpy               as np
import  pandas              as pd
import  multiprocessing     as mp
import  matplotlib.pyplot   as plt
from    obspy               import read, Stream, Trace
from    kav_init            import rootdata, kav_angle, hyp_dist, catalog_array
import  matplotlib.pyplot   as plt
import  matplotlib          as mpl
from    kavutil             import myplot

from instrument_restitution import select_events, read_catalogs_from_files, retrieve_trace_data

from obspy.signal.invsim import simulate_seismometer
import math


# --- initialise event trace -----------------------------------
catalog_1, catalog_2, drop_details = 'catalogue_4.0_lb.txt','catalog_pl.txt', False
year, month, day, hour, minute = 2023, 4, 5, 0, 45
stationdir, stationid = 'KAV11', 'c0941'

distance = hyp_dist[stationdir]

df1, _                  = read_catalogs_from_files(catalog_1, catalog_2, drop_details)
selected_events         = select_events(df1, year, month, day, hour, minute)
stream_events           = retrieve_trace_data(selected_events, stationdir, stationid)
stream_events_restored  = Stream()
stream_events_gndmtn    = Stream()
mlv_box                 = np.array([],dtype=float)


trace = stream_events[0].copy()
plt.plot(trace.times(), trace.data); plt.suptitle('Raw trace'); myplot(); plt.show()

tr = trace.copy()
tr.detrend('demean')
tr.taper(max_percentage=0.05, type='hann')
tr.detrend('demean')

signal = tr.data.copy()

plt.plot(tr.times(), signal); plt.suptitle('Raw trace - demeaned+tapered'); myplot(); plt.show()

# --- Transfer function calculation ----------------------------

Gs:     float = 27.7,
Cv:     float = 6.5574e7,
k:      float = 1.0,
p1, p2      = -19.78 + 20.20j, -19.78 - 20.20j,
z1, z2      = 0.0, 0.0
eps         = 1e-6
trace4tf    = trace.copy()
trace4tf.detrend('demean')
trace4tf.taper(max_percentage=0.1, type='hann') #, max_length=None) 
signal4tf   = trace4tf.data.copy()

# fft of tapered signal
signal_fft_4tf = np.fft.fft(signal4tf)

n4tf    = len(signal4tf)                        # length of signal
dt      = 1/trace.stats.sampling_rate           # sampling interval
tf      = np.ones_like(signal_fft_4tf)          # initialise tf array

# broadcast over positive frequencies to generate frequencies & calculate the transfer function
idx         = np.array(range(n4tf//2+1))
tf_slice    = tf[idx]                           # include Nyquist frequency for even n
f           = idx / (n4tf*dt)                   # compute frequency at index i
omega       = 2*np.pi*f                         # compute angular frequency
s           = 1j*omega                          # compute complex frequency
H           = k * s**2 / ((s - p1) * (s - p2))  # compute transfer function
tf_slice    = H * Gs * Cv                       # consider sensitivities to retrieve output in counts
tf[idx]     = tf_slice                          # fill in tf array

transfer_function = tf

plt.plot(f, np.abs(transfer_function)[:len(f)]); plt.suptitle('Transfer function - seismometer+digitiser'); myplot();plt.show() ### PLOT TRANSFER FUNCTION

npts        = len(signal)                                                   # length of signal
idx         = np.array(range(npts//2+1))                                    # indexing positive frequencies

signal_fft                      = np.fft.fft(signal)                        # fft of tapered signal
inv_tf                          = np.zeros_like(tf)                         # initialise inv tf
inv_tf[idx]                     = 1. / (tf[idx]+ eps*np.max(np.abs(tf)))    # compute inv tf

signal_fft_restored             = signal_fft.copy()                         # initialise restored signal
signal_fft_restored[idx]        = signal_fft[idx] * inv_tf[idx]             # apply inverse transfer function to positive frequencies
del idx                                                                     # clear variable
idx                             = np.array(range(1,npts//2 +1))             # starts from 2 to exclude DC; handles both even and odd n
signal_fft_restored[npts-idx]   = np.conj(signal_fft_restored[idx])         # reconstruct full spectrum by filling in the negative frequencies using conjugate symmetry
# signal_fft_restored[0]      = 0.0

signal_restored                 = np.fft.ifft(signal_fft_restored)          # inverse fft to get the filtered signal

plt.plot(f,             np.abs(signal_fft[:len(f)]));           myplot(); plt.show() ### PLOT FFT OF SIGNAL
plt.plot(f,             np.abs(signal_fft_restored[:len(f)]));  myplot(); plt.show() ### PLOT FFT OF RESTORED SIGNAL
plt.plot(trace.times(), signal);               plt.suptitle('Raw trace prepared');         myplot(); plt.show() ### PLOT SIGNAL
plt.plot(trace.times(), signal_restored);      plt.suptitle('Signal in velocity');         myplot(); plt.show() ### PLOT RESTORED SIGNAL

tr_restored = Trace(data=signal_restored)
tr_restored.stats.starttime = trace.stats.starttime
tr_restored.stats.sampling_rate = trace.stats.sampling_rate
tr_restored.filter('highpass', freq=0.5, zerophase=True)

plt.plot(tr_restored.times(), tr_restored.data); myplot(); plt.show()

del tr
tr = tr_restored.copy()

tr.detrend('demean')
tr.taper(max_percentage=0.05, type='hann')
tr.detrend('demean')

signal = tr.data.copy()
plt.plot(tr.times(), signal); plt.suptitle('Signal in velocity'); myplot(); plt.show() ### PLOT RESTORED SIGNAL

# --- Integration to displacement ------------------------------
del signal_fft, idx, f, s, omega

signal_fft          = np.fft.fft(signal)            # transform signal to frequency domain
n                   = len(signal)                   # length of the signal
idx                 = np.array(range(n // 2 + 1))   # indexing positive frequencies
f                   = idx / (n * trace.stats.delta) # compute frequency at index i
omega               = 2 * np.pi * f                 # compute angular frequency
s                   = 1j * omega                    # compute complex frequency
signal_fft[idx]     = signal_fft[idx] / (s + 1e-9)  # apply division by "j*2*np.pi*f"

del idx 

idx                 = np.array(range(1, n // 2 +1))    # Starts from 2 to exclude DC; handles both even and odd n
signal_fft[n - idx] = np.conj(signal_fft[idx])      # Fill in negative frequencies using conjugate symmetry
# signal_fft[0]       = 0                              # set DC to zero
signal_modified     = np.fft.ifft(signal_fft)       # Inverse FFT to get the modified signal

plt.plot(tr.times(),    np.abs(np.fft.fft(signal)));    plt.suptitle('fft Signal in velocity');     myplot(); plt.show() ### PLOT SIGNAL
plt.plot(f,             np.abs(signal_fft)[:len(f)]);   plt.suptitle('fft signal displacement');    myplot(); plt.show() ### PLOT FFT OF SIGNAL
plt.plot(tr.times(),    signal);                        plt.suptitle('Signal in velocity');         myplot(); plt.show() ### PLOT RESTORED SIGNAL
plt.plot(tr.times()[1:],   signal_modified[1:]);               plt.suptitle('Signal in displacement');     myplot(); plt.show() ### PLOT MODIFIED SIGNAL
plt.plot(tr.times(),    signal_modified);               plt.suptitle('Signal in displacement');     myplot(); plt.show() ### PLOT MODIFIED SIGNAL

# trace_ground_motion      = trace.copy()             # Create a new trace object with the modified signal
# trace_ground_motion.data = signal_modified

trace_ground_motion         = Trace(data=signal_modified)
trace_ground_motion.stats.starttime, trace_ground_motion.stats.sampling_rate = trace.stats.starttime, trace.stats.sampling_rate
trace_ground_motion.filter('highpass', freq=1., zerophase=True)

plt.plot(trace_ground_motion.times(), trace_ground_motion.data); plt.suptitle('Signal in displacement'); myplot(); plt.show() ### PLOT MODIFIED SIGNAL

del tr

tr = trace_ground_motion.copy()
tr.detrend('demean')
tr.taper(max_percentage=0.1, type='hann') #, max_length=None)
tr.detrend('demean')
    

p0, p1 = -5.49779 - 5.60886j, -5.49779 + 5.60886j
z0, z1 = 0.0, 0.0

signal                  = tr.data.copy()
signal_fft              = np.fft.fft(signal)
plt.plot(np.abs(signal_fft)); plt.show()

tf_woodanderson         = np.zeros_like(signal_fft, dtype=complex)
n                       = len(signal)
idx                     = np.array(range(n//2+1))
f                       = idx / (n * tr.stats.delta)
omega                   = 2 * np.pi * f
s                       = 1j * omega
H                       = (s - z0) * (s - z1) / ((s - p0) * (s - p1))
tf_woodanderson[idx]    = H
signal_fft[idx]         = signal_fft[idx] * tf_woodanderson[idx]

idx                     = np.array(range(1, n//2))
signal_fft[n-idx]       = np.conj(signal_fft[idx])

signal_modified         = np.fft.ifft(signal_fft)

trace_woodanderson      = Trace(data=signal_modified)
trace_woodanderson.stats.starttime, trace_woodanderson.stats.sampling_rate = tr.stats.starttime, tr.stats.sampling_rate


plt.plot(tr.times(), signal_modified); plt.suptitle('Signal in displacement'); myplot(); plt.show()

np.max(np.abs(trace_woodanderson.data))

amplitude = np.max(np.abs(trace_woodanderson.data))

mlv = math.log10(amplitude*10**9) + 1.11 * math.log10(distance) + 0.0018 * distance - 2.09


# ---Wood-Anderson Obspy ---------------------------------------
# simulated_data = simulate_seismometer(
#         tr.data.copy(),
#         samp_rate           = tr.stats.sampling_rate,
#         paz_remove          = None,
#         # paz_simulate        = {'poles': [-6.2832 - 0.9425j, -6.2832 + 0.9425j], 'zeros': [0j], 'gain': 2080.0},
#         paz_simulate        = {'poles': [-5.49779 - 5.60886j, -5.49779 + 5.60886j], 'zeros': [0.+0j, 0j], 'gain': 1.0},
#         remove_sensitivity  = False,
#         simulate_sensitivity = False,
#         # water_level = 60.0,
#         zero_mean           = False,
#         taper               = False,
#         nfft_pow2           = True)


# plt.plot(simulated_data); plt.suptitle('Simulated Wood-Anderson'); myplot(); plt.show()
# mlv = math.log10(np.max(np.abs(simulated_data))*10**9) + 1.11 * math.log10(distance) + 0.0018 * distance  - 2.09
