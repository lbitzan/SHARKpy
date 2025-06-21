'''
>>> kav_init.py <<<
Initial-file for KavachiProject. Predefines all empirical values and paths.

L. Bitzan, 2024

'''
# --- SET PATHS --------------------------------------------------------------------------------------------------------
rootproject         = 'C:/User/GitHub'              # <- Set path to project directory
rootcode            = rootproject + '/SHARKpy'      # <- Set path to code.
rootdata            = 'D:/User/data'                # <- Set path to data root
rootouts            = rootdata + '/results/'        # <- set path to output directory. Create manually if not existing

outputlabel         = 'run.xx.test'                 # <- specify label attached to output directory
cataloguefile       = 'cata.run.xx.test.trest_'     # <- Set catalog file name, e.g. for writing catalog to file. Appends trigger rest time automatically.

# if no pkl file exist, create first with 
freqbandpklfile     = 'fbands_computed.pkl' # pkl file with analytically determined frequency bands

# --- SET FLAGS --------------------------------------------------------------------------------------------------------
plot_flag           = False          # True: plot figures, False: do not plot figures
save_flag           = False          # True: save figures, False: show figures [False is not recommended]
write_flag          = True         # True: write output to file, False: do not write output to file

flag_wlvl           = 'constant'    # bool | 'constant' Apply water level for kava index computation

use_three_components            = True
flag_compute_freqbands          = True
exclude_artifact_fbands:   bool = True # True: exclude artifact frequency bands, False: do not exclude artifact frequency bands

visualise_shark_peaks_and_rmv:  bool = False
visualise_detectionfilter:      bool = False


# --- SET TIME  --------------------------------------------------------------------------------------------------------
datetime_start      = '2023-02-01 00:00:00'  # start date for analysis
datetime_end        = '2023-08-01 00:00:00'  # end date for analysis

seasons             = [['2023-02-01','2023-05-02'],
                       ['2023-05-23','2023-08-01'],
                       ['2023-09-14','2023-11-13']]

# --- SET STATION ------------------------------------------------------------------------------------------------------
stationdir          = ['KAV04', 'KAV11']        # station-specific data directory[far field, near-field] 
stationid           = ['c0939','c0941']         # station id # [far field, near-field]

zstr, estr, wstr    = '000000.pri0', '000000.pri1', '000000.pri2' # z, e, n component of station data

# --- General parameters -----------------------------------------------------------------------------------------------
shifttime           = 3.6           # [sec] empirical value. Can be inferred from apparent velocity  

# --- FREQUENCY DOMAIN ANALYSIS -------------------------------------------------------------------------
fmin, fmax, fminbbd = int(4), int(60), int(0.1)       # for general filtering used in bandpass filter

fminbandpass        = {'KAV04': 4.5, 'KAV11': 4.5, 'KAV00': 0.1}
fmaxbandpass        = {'KAV04': 60, 'KAV11': 60, 'KAV00': 60}

# - Frequency bands preset. Required, if "flag_compute_freqbands" == False
kav04bands          = [[4, 7]]
kav11bands          = [[4, 7], [12, 50]]
kav00bands          = [[2, 7], [10, 13]]

freqbands           = {'KAV04': kav04bands,
                       'KAV11': kav11bands,
                       'KAV00': kav00bands}
# - Frequency bands that carry artifacts and should be excluded from analysis
fbands_artifact     = {'KAV04': [[27, 36]],
                       'KAV11': None,
                       'KAV00': None}

# --- SET EMPIRICAL VALUES ---------------------------------------------------------------------------------------------
kavaprodemp         = 100 # 3D empiric - 100 (150) | 3D compute - 10 | bbd empiric - 100 | bbd compute - 100

# --- PLOTTING PARAMETERS ----------------------------------------------------------------------------------------------
sectionlen          = 60*20                      # [s] length of section that is evaluated/plotted


# --- SET Poles and Zeros ------------------------------------------------------------------------------------------------
# - 4.5 Hz 3D geophone
paz45 = {'poles':       [-1.58336E+01  -2.34251E+01j, -1.58336E+01  +2.34251E+01j],
         'zeros':       [+0.00000E+00  +0.00000E+00j, +0.00000E+00  +0.00000E+00j],
         'gain':        +1.0,
         'sensitivity':  2.88000E+01}

# - Trillium Compact 120s Broadband seismometer
pazbbd= { 'poles':      [-3.69100E-02  -3.70200E-02j, -3.69100E-02  +3.70200E-02j,
                         -3.43000E+02  +0.00000E+00j, -3.70000E+02  -4.67000E+02j,
                         -3.70000E+02  +4.67000E+02j, -8.36000E+02  -1.52200E+03j,
                         -8.36000E+02  +1.52200E+03j, -4.90000E+03  -4.70000E+03j,
                         -4.90000E+03  +4.70000E+03j, -6.90000E+03  +0.00000E+00j,
                         -1.50000E+04  +0.00000E+00j ],
          'zeros':      [+0.00000E+00  +0.00000E+00j, +0.00000E+00  +0.00000E+00j,
                         -3.92000E+02  +0.00000E+00j, -1.49000E+03  +1.74000E+03j,
                         -1.49000E+03  -1.74000E+03j, -1.96000E+03  +0.00000E+00j],
          'gain':        +4.34493E+17,
          'sensitivity':  7.54300E+02}

# - Wood-Anderson horinzontal component torsion seismometer
pazWA = {'poles':       [-5.49779E+00  -5.60886E+00j, -5.49779E+00  +5.60886E+00j],
         'zeros':       [+0.00000E+00  +0.00000E+00j, +0.00000E+00  +0.00000E+00j],
         'gain':        2080.0}

# --- SET Reading parameters -------------------------------------------------------------------------------------------
kav_angle               = [-148, -113]             # [deg] Expected backazimuth angles for Kavachi volcano

hyp_dist                = {'KAV04': 36.207,
                           'KAV11': 27.687,
                           'KAV00': 36.600}        # [km] Hypocentral distance from seismometer to volcano

bbd_ids                 = ['KAV00']                # [str] Station name of broadband stations

triggerresttime         = [20.0]                   # list[float] Trigger rest time for detections


# --- Plotting parameters
pltcolors               = ['k', 'tab:blue',   'tab:red',
                           'tab:orange',      '#1f77b4',
                           '#ff7f0e',       '#2ca02c',
                           '#d62728',       '#9467bd',
                           '#8c564b',       '#e377c2',
                           '#7f7f7f',       '#bcbd22',
                           '#17becf']
fontsize                = 12
fontsizelegend          = 8
textwidth               = 354.
cmapnamespec            = 'jet' #'viridis'

bandpasscorners         = 2
use_empiric_freqbands   = 1 # 1=True | 0=False
filter_kava             = 'gaussian' # |'gaussian1d' | None | 'none'
specnoverlap_factor     = .75
window_smooth_kavaprod  = 6     # number of points for smoothing the kava prod
