import urllib2
import StringIO
import csv
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle as pickle

topo_file = '../data/iceland_uk.p' #mediterranean
downsample_degree = 5

def block_reduce(mat, degree):
    nx,ny = mat.shape
    newx,newy = (nx/degree, ny/degree)
    newmat = np.zeros((newx, newy), dtype=mat.dtype)
    for i in range(newx):
        for j in range(newy):
            newmat[i,j] = mat[i*degree:i*degree+degree,j*degree:j*degree+degree].mean()
    return newmat

try:
    with open(topo_file, 'rb') as fp:
        print "Loading pre-saved data file."
        TOPO = pickle.load(fp)
        grid_x = TOPO['lons']
        grid_y = TOPO['lats']
        grid_z = TOPO['topo']
        minlon = grid_x[0,0]
        maxlon = grid_x[-1,0]
        minlat = grid_y[0,0]
        maxlat = grid_y[0,-1]

except IOError:
    print "Topography file {0} not found, creating new...".format(topo_file)
    # Definine the domain of interest
    minlat = 54.0#29.0 #
    maxlat = 65.0#45.0 #
    minlon = -24.0#3.0 #
    maxlon = -4.0#38.0 #
        
    # Read data from: http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.html
    response = urllib2.urlopen('http://coastwatch.pfeg.noaa.gov/erddap/griddap/usgsCeSrtm30v6.csv?topo[(' \
                                +str(maxlat)+'):1:('+str(minlat)+')][('+str(minlon)+'):1:('+str(maxlon)+')]')
    
    data = StringIO.StringIO(response.read())
    
    r = csv.DictReader(data,dialect=csv.Sniffer().sniff(data.read(1000)))
    data.seek(0)
    
    # Initialize variables
    lat, lon, topo = [], [], []
    
    # Loop to parse 'data' into our variables
    # Note that the second row has the units (i.e. not numbers). Thus we implement a
    # try/except instance to prevent the loop for breaking in the second row (ugly fix)
    for row in r:
        try:
            lat.append(float(row['latitude']))
            lon.append(float(row['longitude']))
            topo.append(float(row['topo']))
        except:
            print 'Row '+str(row)+' is a bad...'
    
    # Convert 'lists' into 'numpy arrays'
    lat  = np.array(lat,  dtype='float')
    lon  = np.array(lon,  dtype='float')
    topo = np.array(topo, dtype='float')
    
    # Data resolution determined from here:
    # http://coastwatch.pfeg.noaa.gov/erddap/info/usgsCeSrtm30v6/index.html
    resolution = 0.008333333333333333
    
    # Determine the number of grid points in the x and y directions
    nx = complex(0,(max(lon)-min(lon))/resolution)
    ny = complex(0,(max(lat)-min(lat))/resolution)
    
    # Build 2 grids: One with lats and the other with lons
    grid_x, grid_y = np.mgrid[min(lon):max(lon):nx,min(lat):max(lat):ny]
    
    # Interpolate topo into a grid (x by y dimesions)
    grid_z = scipy.interpolate.griddata((lon,lat),topo,(grid_x,grid_y),method='linear')
    
    # Make an empty 'dictionary'... place the 3 grids in it.
    TOPO = {}
    TOPO['lats']=grid_y
    TOPO['lons']=grid_x
    TOPO['topo']=grid_z
    
    # Save (i.e. pickle) the data for later use
    # This saves the variable TOPO (with all its contents) into topo_file
    with open(topo_file, 'wb') as fp:
        pickle.dump(TOPO, fp)

if downsample_degree != 1:
    # Downsample:
    grid_x = block_reduce(grid_x, downsample_degree)
    grid_y = block_reduce(grid_y, downsample_degree)
    grid_z = block_reduce(grid_z, downsample_degree)
    minlon = grid_x[0,0]
    maxlon = grid_x[-1,0]
    minlat = grid_y[0,0]
    maxlat = grid_y[0,-1]
 
   
# Create map
m = Basemap(projection='mill', llcrnrlat=minlat,urcrnrlat=maxlat,llcrnrlon=minlon, urcrnrlon=maxlon,resolution='i')
x,y = m(grid_x,grid_y)

fig1 = plt.figure()
cs = m.pcolor(x,y,grid_z,cmap=plt.cm.jet)
m.drawcoastlines()
m.drawmapboundary()
plt.title('SMRT30 - Bathymetry/Topography')
cbar = plt.colorbar(orientation='horizontal', extend='both')
cbar.ax.set_xlabel('meters')

plt.show()
 
# Save figure (without 'white' borders)
# plt.savefig('../fig/topo.png', bbox_inches='tight')