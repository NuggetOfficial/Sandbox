'''
author : Tom van der Wielen*
last_edit : 2-01-2022

description:
    In this file I created a class that reads a DIR of .gpx coordinates and con-
    catinates the read data in a vector, containing waypoint names and a matrix
    containing the corresponidng waypoint coordinates in lat-long. These
    coordinates are then projected to an xy plane. In this plane all pairs of 
    euclidian distances are calculated to create a distance matrix. This matrix
    is then collapsed using multidimensional scaling (MDS). The predicted waypoint
    coordinates are rotated by 45 degrees and the result is plotted alongside the
    GEOdata.
    

* Bsc Student at Leiden University - NL, 5th semester exchange student at 
  UiT - NO;  primary affiliation : LeidenUniv
'''

""" ####========---------------- IMPORT AND INIT --------------========#### """
# DIR MANAGER
import os

# DEFINE DIRS
# get dir
DIR = os.path.dirname(os.path.realpath(__file__))

# define dirs
dataDIR   = DIR + '/data/'
rawDIR    = dataDIR + 'RAW/'
genDIR    = dataDIR + 'GENERATED/'
outputDIR = DIR + '/output/'
figDIR    = outputDIR + 'figures/'   

# IMPORT HOLIEST OF LIBS
import numpy as np

# IMPORT GRAPH THINGS
from matplotlib import pyplot as plt

# IMPORT GPX FILE LOADER
import gpxpy
import gpxpy.gpx

# IMPORT PORJECTION
import pyproj

# IMPORT SCIPY FOR CALC AND CONSTANTS
import scipy.spatial

""" ####========----------------- DEFINITIONS -----------------========#### """
# FILE MANIPULATION
class GPX:
    ''' This class is used to extract waypoint data from .gpx files given a DIR.
     This DIR should contain purely .gpx files to avoid crashes.'''
    
    def __init__(self, DIR, *,fname = None):
        ''' Initialise the GPX data extrator with a directory and (optionally)
         a filename. If no filename provided the entire directory will be sweeped
        
        IN: DIR   ; (str) Contains the realpath to directory containing .gpx files
            fname ; (str) the name of the file to be loaded [if not provided ->
                          None -> then sweep DIR]
        '''
        
        # SET INPUT ATTRIBUTES
        self.DIR = DIR
        self.fname = fname
        
        # GLOBAL LISTS : will be filled
        self.cities = list()
        self.coords = list()
    
    
    def _sweep_dir(self):
        ''' check method, if no filename is defined during __init__, then sweep
         all the files in provided DIR '''
        
        # TERNARY OPERATOR; if fname is none return False else True
        return False if self.fname else True
        
    
    def _append_data(self, waypoint):
        ''' Update output lists using current waypoint.
        
        IN: waypoint ; (object) a gpxpy waypoint object
        '''
        
        # APPEND TO OUTPUT
        self.cities.append(waypoint.name)
        self.coords.append(tuple([waypoint.latitude,waypoint.longitude]))
        
        
    def _extract(self):
        ''' extract waypoints from file and add them to output lists'''
        
        # CREATE PARSE OBJECT
        _data = gpxpy.parse(open(self.DIR + self.fname))
        
        # DISECT FILE : use list comprehension to disect waypoints
        [self._append_data(waypoint) for waypoint in _data.waypoints]
    
    
    def extract(self):
        ''' Use to extract data: checks self._sweep_dir, if True use OS to
         loop over files, else extract from file provided'''
        
        # ASSERT IF DATA AS ALEADY BEEN EXTRACTED
        if ( len(self.cities) != 0 ) or ( len(self.coords) != 0 ):
            raise AssertionError('Data has already been extracted')
        
        # IF SWEEP
        if self._sweep_dir():
            
            # WALK TO DIR
            for __,__,files in os.walk(self.DIR):
                
                # LOOP THROUGH FILES
                for file in files:
                    self.fname = file
                    self._extract()
                    
            # RESET FILENAME : here self.fname is used as a flag
            self.fname = None
            
        else:
            # EXTRACT FROM FILENAME
            self._extract()
        
        # OUTPUT THE LISTS AS <np.ndarray>
        return np.array(self.cities), np.array(self.coords)
   
    
#MDS
class MDS:
    ''' allows for mutlidimensioncal scaling on a dataset X '''
    
    def __init__(self, X, silent = True):
        ''' Initialise using the data
        
         
        IN: X      ; (matrix) contains data, of form [samples] x [features]
            silent ; (bool) if True -> no printing
        '''
        
        # INITIALISE PARSED VARIABLES
        self.X = X
        self.silent = silent
        
        # INITIALISE PREPARED MATRIX ADRESS
        self.B = None
        
        # INTIALISE OUTPUT ADRESS
        self.outputRAW = None
    
    
    def _prepare_matrix(self):
        ''' method that prepares self.X for MDS'''
        
        # Use TERNARY OPERATOR:
        if not self.B:
            
            # DECOMPOSE X
            r, c = self.X.shape
            
            # GET CENTERING MATRIX
            H = np.eye(r) - np.ones( (r, r) )/r
            
            # PREPARE MATRIX
            self.B = -H.dot(np.square(self.X)).dot(H)/2
        
        else:
            # IF B ALREADY EXISTS RAISE
            raise AssertionError(' Matrix has already been prepared. ')
            
    
    def collapse_to_dimensions(self, n):
        ''' main algorithm. collapse self.X to {n} dimensions
        
        IN: n ; (int) the amount of resulting dimension you want to collapse to.
        '''
        
        # PREPARE X
        self._prepare_matrix()
        
        # GET EIGENVALUES AND VECTORS
        _w,_v = np.linalg.eigh(self.B)
        
        # SORT EIGENVALUES FROM HIGH TO LOW
        idxs = _w.argsort()[::-1]
        
        # PASS EIGENVALUES AND VECTORS
        w = _w[idxs]  ;  v = _v.T[idxs]
        
        # USE POSITIVE EIGENVALUES ONLY
        whep = (w >= 0)
        wp = w[whep]  ;  vp = v[whep]
        
        # CALCULATE Z
        Z = np.dot(np.sqrt(np.diag(wp)),vp)
        
        # SAVE TO OUTPUT ADRESS
        self.outputRAW = Z
            
        # OUT
        return Z[:n]
    

    

# ALL FUNCTIONS
# projection
def ConvertToMapProjection(Coordinates):
    ''' Projects a mapped pair of coordinates onto a map. Extracted from
     https://stackoverflow.com/questions/21968924/pyproj-mapping-latitude-
     longitude-onto-rectangle-using-x-y-coordinates.'''

    # Define the projection
    settings = "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    RobinsonProjection = pyproj.Proj(settings)

    # EXECUTE PROJECTION
    East, North = RobinsonProjection(Coordinates[0],Coordinates[1])

    return np.array([East,North])

# rotation
def RotateByMatrixMultiplication(X, a):
    ''' Rotate the matrix X with angle a.'''
    
    # CALCULATE 2x2 ROTATION MATRIX
    c = np.cos(a)  ;  s = np.sin(a)
    R = np.array([[c,s],[-s,c]])
    
    # APPLY MATRIX TO DATA MATRIX
    return R.dot(X)


""" ####========---------------- EXECUTE CODE -----------------========#### """
# MAIN CODE BLOCK AND EXECUTION
def main(write = True,*,DIR = ''):
    ''' Intended programm , load .gpx data, project to xy, determine distance
     matrix, predict xy using multidimensional scaling (MDS)'''
     
    # EXTRACT DATA
    gpx = GPX(rawDIR)
    cities, coords = gpx.extract()
    
    # GET UNIQUES
    idxs = np.unique(cities, return_index = True)[1]
    
    # SELECT UNIQUES
    cities = cities[idxs]  ; coords = coords[idxs]
    
    # PROJECT LATLONG TO XY
    xy = ConvertToMapProjection(coords.T)
    
    # COMPUTE DISTANCE MATRIX
    D = scipy.spatial.distance_matrix(x = xy.T, y = xy.T)
    
    # MDS
    xy_hat = MDS(D).collapse_to_dimensions(2)
    
    # ROTATE xy_hat
    xy_hat = RotateByMatrixMultiplication(xy_hat, 45)
    
    # FIG
    # inni
    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (20,10), dpi = 400)
    
    # ax markup
    ax2.invert_xaxis()
    ax1.set_title('Robinson Projection GEOcoords')
    ax2.set_title('MSD Approximation: rotated 45 deg | inv-x')
    
    # plot
    ax1.scatter(xy[1],xy[0],alpha = .7)
    ax2.scatter(xy_hat[1],xy_hat[0], alpha = .7)
    
    # save if allowed to
    if write:
        plt.savefig(DIR + 'MDS_GEO_comparision.pdf')
        
    # flush mpl
    plt.show()
    
    return True

# IF RAN FROM FILE DIRECTLY
if __name__ == '__main__':
    
    # EXECUTE INTENDED PROGRAMM
    if main(DIR = figDIR):
        print(' Finished intended pogramm defined by <main>.')
    
