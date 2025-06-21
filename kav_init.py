'''
>>> kav_init.py <<<
Initial-file for KavachiProject. Predefines all empirical values and paths.

L. Bitzan, 2024
2024-01-08 // general setup py-file

'''
# --- SET PATHS --------------------------------------------------------------------------------------------------------
# rootproject         = 'C:/Users/Arbeit/Documents/matlab/projects/KavachiProject'
rootproject         = 'C:/Users/Ludwig/projectsT14/GitHub'
rootcode            = rootproject + '/KavScripts'
# rootdata            = rootproject + '/Data'                         # <--- set path to smaller data set on local machine
rootdata            = 'D:/data_kavachi_both'                          # <--- set path to larger data set on external hard drive
rootouts            = rootdata + '/results/'                          # <--- set path to output directory

# outputlabel         = 'cata_10.3D.testplot' # '_5.0_.20h.tremor_' # specify label attached to output directory e.g. plots from runPlotting.py
# cataloguefile       = 'cata_10.3D.newcata.lb.' # 'catalogue_5.0_rrtest_lb.txt'
# # newcataoutput       = 'catalog_times_0.1_lb.txt'                      # for apparent velocity computation

# outputlabel         = 'v11.3D_long_emp' # '_5.0_.20h.tremor_' # specify label attached to output directory e.g. plots from runPlotting.py
# cataloguefile       = 'cata_v11.3D_long.emp_tb_' # 'catalogue_5.0_rrtest_lb.txt'
outputlabel, cataloguefile = 'vis.thesis.v11', 'vis.thesis.cata.v11'

catalog_array       = rootdata + '/catalog_pl.txt'                          # path to array catalog for matching events & retrieval of baz data
catalog_magnitudes  = rootdata + '/results/magnitudes_catalog_4_lb.pkl'     # path to magnitude catalog e.g. for plotting
# --- SET FLAGS --------------------------------------------------------------------------------------------------------
plot_flag           = True  # True: plot figures, False: do not plot figures
save_flag           = True  # True: save figures, False: show figures [False is not recommended]
write_flag          = False   # True: write output to file, False: do not write output to file
pkl_only            = False

month_flag          = True # True: run loop over months,        False: run loop over manually set days
hour_flag           = True  # True: run loop over whole day,     False: run loop over manually set hours;// relevant for plotting only

flag_bandedratio    = True  # True: compute amplitude ratio based on particular frequency bands, False: compute amplitude ratio based on whole frequency band
flag_wlvl           = 'constant'  # bool | 'constant' Apply water level for kava index computation
flag_group          = True  # True: group activities to single events, False: keep all activities separate
flag_horizontal     = False

flag_matching       = False # True: Read in matching events, False: Read in all events

use_three_components            = True
flag_compute_freqbands          = False
compare_different_triggers      = 'frequency' # True | 'amplitude' | 'frequency'
exclude_artifact_fbands:   bool = True # True: exclude artifact frequency bands, False: do not exclude artifact frequency bands

visualise_shark_peaks_and_rmv   = False
visualise_detectionfilter       = False


# --- SET TIME  --------------------------------------------------------------------------------------------------------
yoi                 = 23                        # years of interest
moi                 = [2,3,4,5,6,7]                       # months of interest
doi                 = [1]                      # days of interest
hoi                 = [1]                       # hours of interest

seasons             = [['2023-02-01','2023-05-02'],
                       ['2023-05-23','2023-08-01'],
                       ['2023-09-14','2023-11-13']]

# --- SET STATION ------------------------------------------------------------------------------------------------------
# stationdir          = ['KAV00', 'KAV11']        # station of interest
# stationid           = ['c0bdd','c0941']         # station id

stationdir          = ['KAV04', 'KAV11']        # [far field, near-field] station of interest
stationid           = ['c0939','c0941']         # station id

zstr                = '000000.pri0'
estr                = '000000.pri1'
wstr                = '000000.pri2'

# --- General parameters -----------------------------------------------------------------------------------------------
# shifttime           = 2.0                       # empirical value [sec] - time delay between stations; alternatively 2.5 sec
shifttime           = 3.6           # empirical value [sec] - time delay between stations; alternatively 2.5 sec    


# --- EMPIRICALS FOR TIME DOMAIN ANALYSIS ------------------------------------------------------------------------------
ratiorange          = [40, 500]                 # range of characteristic amplitude ratio [40,200]

# ratiorange = {  'KAV04': [40, 200],
#                 'KAV11': [40, 200]}


# --- EMPIRICALS FOR FREQUENCY DOMAIN ANALYSIS -------------------------------------------------------------------------
fmin                = int(4)                    # for general filtering used in bandpass filter
fmax                = int(60)
fminbbd             = int(0)
fminbandpass        = {'KAV04': 4.5, 'KAV11': 4.5, 'KAV00': 0.1}
fmaxbandpass        = {'KAV04': 60, 'KAV11': 60, 'KAV00': 60}

kav04bands          = [[4, 7]]
kav11bands          = [[4, 7], [12, 50]]
kav00bands          = [[2, 7], [10, 13]]

freqbands           = {'KAV04': kav04bands,
                       'KAV11': kav11bands,
                       'KAV00': kav00bands}

fbands_artifact     = {'KAV04': [[27, 36]],
                       'KAV11': None,
                       'KAV00': None}  # frequency bands that are considered as artifact bands and should be excluded from analysis

waterlvl            = {'KAV04': 400,
                       'KAV11': 2200,
                       'KAV00': 150}

# --- SET EMPIRICAL VALUES ---------------------------------------------------------------------------------------------
ratioemp            = 1                         # normalisation coefficient for amplitude ratio [close station/remote station]
conflim             = 0.2                       # confidence limit for amplitude ratio trigger
# kavaprodemp         = 300                       # trigger threshold for characteristic SHARK index
kavaprodemp         = 100 # 3D empiric - 100 (150) | 3D compute - 10 | bbd empiric - 100 | bbd compute - 100


# --- PLOTTING PARAMETERS ----------------------------------------------------------------------------------------------
sectionlen          = 60*20                      # length of section that is evaluated/plotted [s]; default: 3 min


# --- SET Poles and Zeros ------------------------------------------------------------------------------------------------
paz45 = {'poles':       [-1.58336E+01 -2.34251E+01j, -1.58336E+01  +2.34251E+01j],
         'zeros':       [+0.00000E+00  +0.00000E+00, +0.00000E+00  +0.00000E+00],
         'gain':        +1.0,
         'sensitivity':  2.88000E+01}


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

pazWA = {'poles':       [-5.49779E+00  -5.60886E+00j, -5.49779E+00  +5.60886E+00j],
         'zeros':       [+0.00000E+00  +0.00000E+00j, +0.00000E+00  +0.00000E+00j],
         'gain':        2080.0}

# --- SET Reading parameters -------------------------------------------------------------------------------------------
kav_angle               = [-148, -113] #[-145, -115] #-100]                # opening angle for events from kavachi direction [deg]


hyp_dist                = {'KAV04': 36.207,
                           'KAV11': 27.687,
                           'KAV00': 36.600}

# hyp_dist                = {'KAV04': 36.197, 'KAV11': 27.674, 'KAV00': 36.605}

bbd_ids                 = ['KAV00']

triggerresttime         = [20.0] # 20. #.5 # seconds


# Plotting parameters
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
freqbandpklfile         = 'multimodal_freqanalysis_output_KDE_bw_0.2_hmin_0.3.pkl' #'multimodal_freqanalysis_output.pkl' # pkl file with analytically determined frequency bands
specnoverlap_factor     = .75
window_smooth_kavaprod  = 6 # number of points for smoothing the kava prod
