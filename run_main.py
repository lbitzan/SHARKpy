'''
>> run_main.py <<

Run main script. Either >> runPlotCatalog << or >>magnitude_estimation << can be called.
Set all parameters in kav_init.py.
Make sure to have all library files in the same directory as this script.

Required libraries:
numpy, pandas, matplotlib, obspy, scipy, os

L. Bitzan, 2025
'''

if __name__ == '__main__':
    import numpy as np, os as os, pandas as pd
    import kavutil_2 as kt
    from runplotcatalog1_6 import runPlotCatalog
    import kav_init as ini
    from magnitude_estimation import run_Mlv_v2
    import matplotlib.pyplot as plt
    from compute_frequencybands import freq_band_analysis_process
    vis = True

    if not os.getcwd() == 'C:/Users/Ludwig/projectsT14/GitHub/KavScripts':
        os.chdir('C:/Users/Ludwig/projectsT14/GitHub/KavScripts')
    
    start, end = '2023-02-01', '2023-08-01'
    start, end = '2023-04-05', '2023-04-06'

    # If frequency should be computed from reference catalog/data, set flags accordingly in kav_init.py.
    # Make sure the reference catalog is available in the specified path.
    if ini.flag_compute_freqbands:
        # 1. Preallocations
        referencecatalogfile = 'catalog_pl.txt'
        path          = ini.rootdata
        timevar       = 'Starttime'
        var           = 'BAZ'
        kav_angle     = [-145,-100] # 0 kav_angle # from init
        nrand         = 20
        linewidth     = 1.
        stationidlist = ['c0941','c0939','c0bdd']
        stationlist   = ['KAV11','KAV04','KAV00']
        pathout       = ini.rootouts + 'stacked_spectrum/' # <- set path to output directory for stacked spectrum

        bw_factor     = 0.2
        heightpeaks   = 0.3
        df_output_multimodal = freq_band_analysis_process(referencecatalogfile,
                                                            path,
                                                            timevar,
                                                            var,
                                                            kav_angle,
                                                            nrand,
                                                            linewidth,
                                                            stationidlist,
                                                            stationlist,
                                                            pathout,
                                                            bw_factor=bw_factor,
                                                            heightpeaks=heightpeaks,
                                                            showplots=False)
        fpickle = 'fbands_computed.pkl'
        df_output_multimodal.to_pickle(os.path.join(ini.rootdata, fpickle))
        ini.freqbandpklfile = fpickle
  
    breakpoint()
    # Run plot catalog
    runPlotCatalog(datetime_start=start, datetime_end=end)
    breakpoint()
    # Load catalog
    catalog = pd.read_pickle(ini.outputcatalogpkl)

    # Visualise catalog
    if vis:
        daily_counts = catalog['Eventtime(UTC)'].dt.date.value_counts().sort_index() 
        df_cnts = pd.DataFrame({'Date': daily_counts.index, 'Counts': daily_counts.values})
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df_cnts['Date'], df_cnts['Counts'], width=0.4, color='steelblue', alpha=0.7, label=f'Total number of events: {df_cnts["Counts"].sum()}')
        ax.set_ylabel('Number of Events per Day'); ax.set_xlabel('Date'); ax.set_title('Daily Event Counts')
        ax.legend(loc='upper left', fontsize=10)
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(kt.DateFormatter('%Y-%m-%d'))
        plt.show()

    # Run magnitude estimation
    _,_= run_Mlv_v2(ini.outputcatalogpkl,
               datetime_start=start,
               datetime_end=end,
               distance=None,
               stationdir=ini.stationdir[1],
               stationid=ini.stationid[1],
               drop_details=False,
               add_2_catalog=True,
               batch_size=200,
               )
    # Load magnitude catalog
    fcatalog_magnitudes = ini.outputcatalogpkl.replace('.pkl', '.mlv.pkl')

    # Visualise magnitude catalog
    if vis:
        catalog_mlv = pd.read_pickle(fcatalog_magnitudes)
        # scatter magnitudes
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(catalog_mlv['Eventtime(UTC)'], catalog_mlv['mlv'], s=10, c='steelblue', alpha=0.5, label=f'events: {len(catalog_mlv)}\n mean: {catalog_mlv["mlv"].mean():.2f}\n std: {catalog_mlv["mlv"].std():.2f}')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlabel('Event Time (UTC)'); ax.set_ylabel('Magnitude'); ax.set_title('Magnitude Catalog')
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(kt.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.show()
