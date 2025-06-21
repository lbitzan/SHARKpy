import numpy                as np
import pygmt
from   kav_init             import *

# --- Plot Solomon Islands overview
plot_solomon    = True
if plot_solomon == True:
    import  pygmt
    from    kav_init import *
    minlon, maxlon, minlat, maxlat = 154, 162, -10.5, -5
    font    = "Helvetica-Bold"
    gridsol = pygmt.datasets.load_earth_relief(
        data_source = "igpp",
        resolution  = "15s",
        region      = [minlon, maxlon, minlat, maxlat],
        use_srtm    = False
        )
    figsol  = pygmt.Figure()
    gridsol = gridsol #- 5.
    figsol.grdimage(
        grid        = gridsol,
        frame       = "a",
        projection  = "M10c",
        cmap        = "geo",
        shading     = True
        )
    figsol.colorbar(
        frame       = ["a1000", "x+lElevation", "y+lm"])
    figsol.grdcontour(
        grid        = gridsol,
        interval    = 1000,
        annotation  = "1000+f6p",
        limit       = "-8000/0",
        pen         = "a0.15p"
        )
    # Plot Kavachi
    figsol.plot(x=[157.979], y=[-8.991],  style='kvolcano/.3c', fill = 'red',  pen='black') # Kavachi
    # figsol.text(x=[157.979], y=[-9.2],    text="Kavachi", font = font)
    # Plot Solomon Islands
    figsol.text(x=[158.680], y=[-8.479],  text="Solomon Islands", angle=335, font=font)
    # Plot Nggatokae Island
    figsol.plot(x=[158.1818], y=[-8.771], style='c.2c', fill='violet', pen='black')
    # save figure
    figsol.savefig(rootdata+"/results/poster/Solomon_Islands_Overview.png", dpi=600)
    figsol.show()

# --- Plot map of Nggatokae Island with array locations
plot_local      = False
if plot_local   == True:
    import pygmt
    minlon, maxlon, minlat, maxlat = 158.1, 158.25, -8.84, -8.73
    proj            = "M10c"
    stationx        = [158.151, 158.212]
    stationy        = [-8.82,   -8.766]
    stationtxt      = ["Array 1", "Array 2"]
    gridloc         = pygmt.datasets.load_earth_relief(
        data_source = "igpp",
        resolution  = "01s",
        region      = [minlon, maxlon, minlat, maxlat],
        use_srtm    = False
        )
    gridloc         = gridloc - 5.
    figloc          = pygmt.Figure()
    figloc.grdimage(
        grid        = gridloc,
        frame       = "a",
        projection  = proj,
        cmap        = "geo",
        shading     = True
        )
    figloc.colorbar(
        frame       = ["a1000", "x+lElevation", "y+lm"])
    # figloc.coast(
    #     region      = [minlon, maxlon, minlat, maxlat], 
    #     projection  = proj, 
    #     shorelines  = True,
    #     frame       = True
    #     )
    figloc.grdcontour(
        grid        = gridloc,
        interval    = 100,
        annotation  = "100+f6p",
        limit       = "-8000/0",
        pen         = "a0.15p"
        )
    # Plot lines between stations 
    figloc.plot(x=stationx, y=stationy, pen="2p,black")
    figloc.text(x=stationx[0]+(stationx[1]-stationx[0])/2 + 0.01, y=stationy[0]+(stationy[1]-stationy[0])/2, text='~ 8 km', font="Helvetica-Bold")
    # Plot stations
    figloc.plot(x=stationx[0], y=stationy[0], style= 'c.3c', color= 'blue', pen= 'black')
    figloc.plot(x=stationx[1], y=stationy[1], style= 'c.3c', color= 'red',  pen= 'black')
    figloc.text(
        x           = stationx[0],
        y           = stationy[0]-.01,
        text        = 'Array 1 w "KAV11"',
        font        = "Helvetica-Bold")
    figloc.text(
        x           = stationx[1],
        y           = stationy[1]-.01,
        text        = 'Array 2 w "KAV04"',
        font        = "Helvetica-Bold")
    figloc.savefig(rootdata+"/results/poster/Nggatokae_Island.pdf")
    figloc.show()

# --- Plot new regional map
plot_regional   = False
if plot_regional == True:
    from kav_init import *
    import pygmt
    minlon, maxlon, minlat, maxlat = 157.9, 158.28, -9.05, -8.7
    proj            = "M10c"
    transpi         = 20
    stationx        = [158.151, 158.212]
    stationy        = [-8.82,   -8.766]
    stationtxt      = ["Array 1", "Array 2"]
    gridreg         = pygmt.datasets.load_earth_relief(
        data_source = "igpp",
        resolution  = "01s",
        region      = [minlon, maxlon, minlat, maxlat],
        use_srtm    = False
        )
    gridreg         = gridreg - 5.
    figreg          = pygmt.Figure()
    figreg.grdimage(
        grid        = gridreg,
        frame       = "a",
        projection  = proj,
        cmap        = "geo",
        shading     = True
        )
    figreg.colorbar( frame       = ["a1000", "x+lElevation", "y+lm"])
    figreg.grdcontour(
        grid        = gridreg,
        interval    = 100,
        annotation  = "100+f6p",
        limit       = "-8000/0",
        pen         = "a0.15p"
        )
    # Plot lines between stations 
    figreg.plot(x=stationx, y=stationy, pen="2p,black")
    figreg.text(x=stationx[0]+(stationx[1]-stationx[0])/2 + 0.035, y=stationy[0]+(stationy[1]-stationy[0])/2, text='~ 8 km', font="Helvetica-Bold")
    # Plot stations
    figreg.plot(x=stationx[0], y=stationy[0], style= 'c.3c', fill= 'blue', pen= 'black', transparency=transpi)
    figreg.plot(x=stationx[1], y=stationy[1], style= 'c.3c', fill= 'red',  pen= 'black', transparency=transpi)
    figreg.text(x=stationx[0], y=stationy[0]-.02, text='Array 1,  "KAV11"', font="Helvetica-Bold")
    figreg.text(x=stationx[1], y=stationy[1]+.02, text='Array 2,  "KAV04"', font="Helvetica-Bold")
    # Plot Kavachi
    figreg.plot(x=[157.979], y=[-8.991],  style='kvolcano/.4c', fill = 'red',  pen='black')
    figreg.text(x=[157.979], y=[-8.991+.02],  text="Kavachi", font="Helvetica-Bold")
    figreg.savefig(rootdata+"/results/poster/Kavachi_region_map_png.png", dpi=600)
    figreg.show()





# del /p "grdblend_resampled_a19936.nc"