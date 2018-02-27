import cv2
import itertools
import numpy as np
from scipy.optimize import fsolve

from videoslicer import VideoFrame
from videoslicer.utils import *
from videoslicer.markers import *


class CRSConverter(object):
    '''Helper class for coordinate conversion

    Given the shape of a matrix this class provides an itemgetter that
    returns the coordinates of each item in the matrix in a given
    coordinate reference system. The coordinate reference system is
    defined by an origin and a resolution.

    The class can return x-coordinates (axis=0), y-coordinates
    (axis=1) or both (axis=-1).

    Examples
    --------
    >>> c = CRSConverter((100,50), (50,10), 5, axis=-1)
    >>> c[0,0]
        (-2.0, -10.0)
    >>> c[5,5]
        (-1.0, -9.0)
    >>> c[[(0,0),(0,5),(5,5)]]
        [(-2.0, -10.0), (-2.0, -9.0), (-1.0, -9.0)]
    >>> c[:,5]
        [array([[-2. , -1.8, ...,  7.6,  7.8]]),
         array([[-9., -9., ..., -9., -9.]])]

    >>> c = CRSConverter((100,50), (50,10), 5, axis=0)
    >>> c[0]
        -10.0
    >>> c[5]
        -9.0
    >>> c[[0,5]]
        [-10. -9.]
    >>> c[:]
        [-10. -9.8 ... 9.6 9.8]

    >>> c = CRSConverter((100,50), (50,10), 5, axis=1)
    >>> c[0]
        -2.0
    >>> c[5]
        -1.0
    >>> c[[0,5]]
        [-2. -1.]
    >>> c[:]
        [-2. -1.8 ... 7.6  7.8]

    '''
    
    def __init__(self, shape, origin, resolution, axis=0):
        '''Initialization

        Parameters
        ----------
        shape : tuple
          Shape of pixel matrix
        origin : tuple
          Origin of coordinate reference system in pixel coordinates
        resolution : float
          Resolution of pixel matrix in coordinate reference system in
          px/length
        axis : 0, 1, or -1
          Axis information to return (0, 1 or -1 for both)

        '''
        
        self.shape = shape
        self.origin = origin
        self.resolution = resolution
        self.axis = axis

        self.u = np.arange(self.shape[0])
        self.v = np.arange(self.shape[1])
        
    
    def __getitem__(self, s):
        '''Itemgetter'''
        
        s = preprocess_getitem_args(s, self.shape)

        if self.axis < 0:
            # return both axes
            if len(s) == 1:
                # assume a list of 2-tuples that need to be converted one-by-one
                s = list(zip(*s[0]))
                x = CRSConverter(self.shape, self.origin, self.resolution, axis=1)[list(s[0])]
                y = CRSConverter(self.shape, self.origin, self.resolution, axis=0)[list(s[1])]
                return list(zip(x, y))
            else:
                # assume two slices for which the meshgrid needs to be
                # returned, unless the result contains only a single
                # coordinate
                x = CRSConverter(self.shape, self.origin, self.resolution, axis=1)[s[0]]
                y = CRSConverter(self.shape, self.origin, self.resolution, axis=0)[s[1]]
                if isinstance(x, float) and isinstance(y, float):
                    return x, y
                else:
                    return np.meshgrid(x,y)
        elif self.axis == 0:
            # return first axis
            return (self.u[s] - self.origin[0]) / self.resolution
        elif self.axis == 1:
            # return second axis
            return (self.v[s] - self.origin[1]) / self.resolution
        else:
            raise ValueError('Unspported axis: %d' % self.axis)
        
        
class VideoFrameCRS(VideoFrame):
    '''VideoFrame class with built-in coordinate reference system (crs)

    Extension of the VideoFrame class that supports marker detection
    in the video frame. Distances between the markers and an
    arbitrarily chosen origin should be given to derive a coordinate
    reference system of the image frame. The markers are assumed to be
    in a plane perpendicular to the camera.

    A marker is assumed to be a white square with a red centered
    dot. The diameter of the dot is assumed to be half of the
    width/height of the square. Markers are numbered from top to
    bottom.

    '''
    
    def __new__(cls, arr, distances_markers=[], distances_origin=[],
                n_markers=4, method='redness', method_args={}, *args, **kwargs):
        '''Constructor

        See `VideoFrame` for more details.

        Parameters
        ----------
        distances_markers : list of 2-tuples
          List of 2-tuples where each item contains another 2-tuple
          and a float. The 2-tuple contains the marker numbers for
          which the interdistance is defined by the float.
        distance_origin : list of 2-tuples
          List of 2-tuples where each item contains an integer and a
          float. The integer is the marker number for which the
          distance to the origin of the coordinate reference system is
          defined by the float.
        n_markers : int
          Number of markers in the frame
        method : str
          Detection method (redness or template)
        method_args : dict
          Keyword-value arguments to the detection method (see
          `videoslicer.markers`)

        '''
        
        if n_markers < 0:
            raise ValueError('At least 2 markers are required.')
        
        obj = super(VideoFrameCRS, cls).__new__(cls, arr, *args, **kwargs)
        
        obj.n_markers = n_markers
        obj.method = method
        obj.method_args = method_args
        
        obj.marker_distance_to_marker_uv = []
        obj.marker_distance_to_marker_xy = distances_markers
        obj.marker_distance_to_origin_uv = []
        obj.marker_distance_to_origin_xy = distances_origin
        obj.marker_position_uv = []
        obj.marker_position_xy = []
        obj.origin_uv = None
        obj.origin_xy = (0, 0)
        obj.resolution = None
        obj.resolution_rmse = None
        
        obj._resolutions = []

        obj.find_markers()
        obj.determine_resolution()
        obj.determine_origin()
        obj.compute_derivatives()
        
        return obj

    
    def __repr__(self):
        s = 'VideoFrameCRS:\n'
        s += '  Frame:\n'
        s += '    %-12s %8d%8d px\n' % ('size:', *self.shape[:2])
        s += '    %-12s %8d\n' % ('depth:', self.shape[2])
        s += '  Pixel coordinates:\n'
        s += '    %-12s %8d%8d px\n' % ('origin:', *self.origin_uv)
        for i, uv in enumerate(self.marker_position_uv):
            s += '    %-12s %8d%8d px\n' % ('markers:' if i == 0 else '', *uv)
        s += '  Real-world coordinates:\n'
        s += '    %-12s %8.2f%8.2f cm\n' % ('origin:', *self.origin_xy)
        for i, xy in enumerate(self.marker_position_xy):
            s += '    %-12s %8.2f%8.2f cm\n' % ('markers:' if i == 0 else '', *xy)
        s += '  Conversion:\n'
        s += '    %-12s %8.4f px/cm\n' % ('resolution:', self.resolution)
        s += '    %-12s %8.4f px/cm\n' % ('accuracy:', self.resolution_rmse)
        return s
        

    def find_markers(self):
        '''Find markers in the video frame given a specific method'''
        if self.method == 'redness':
            self.marker_position_uv = find_markers_redness(self, **self.method_args)
        elif self.method == 'template':
            self.marker_position_uv = find_markers_template(self, **self.method_args)
        else:
            raise ValueError('Unsupported method: %s' % self.method)
        
        
    def determine_resolution(self):
        '''Determine the average resolution from the distances between detected markers'''
        
        if len(self.marker_distance_to_marker_xy) < 1:
            raise ValueError('Provide at least one distance in real-world '
                             'coordinates between two markers.')
        if len(self.marker_position_uv) < 2:
            raise ValueError('First determine the marker positions by '
                             'calling `find_markers`.')

        
        self._resolutions = []
        for (i1,i2), dst_xy in self.marker_distance_to_marker_xy:
            dst_uv = np.sqrt(np.sum((np.asarray(self.marker_position_uv[i1]) - 
                                     np.asarray(self.marker_position_uv[i2]))**2))
            self.marker_distance_to_marker_uv.append(((i1,i2), dst_uv))
            self._resolutions.append(dst_uv / dst_xy)
        self.resolution = np.mean(self._resolutions)
        self.resolution_rmse = np.sqrt(np.mean(np.asarray(self._resolutions - self.resolution)**2))
        
        
    def determine_origin(self, p0=[0,0]):
        '''Determine the origin in pixel coordinates from the distances between markers and origin

        Parameters
        ----------
        p0 : tuple
          Initial guess of origin location

        '''
        
        def equations(p, anchors):
            eqs = []
            for i, dst in anchors:
                eq = np.sqrt((p[0]-self.marker_position_uv[i][0])**2 + \
                             (p[1]-self.marker_position_uv[i][1])**2) - dst * self.resolution
                eqs.append(eq)
            return eqs
        
        if len(self.marker_distance_to_origin_xy) < 2:
            raise ValueError('Provide at least two distances in real-world '
                             'coordinates between a marker and the origin.')
        if self.resolution is None:
            raise ValueError('First determine the frame resolution by '
                             'calling `determine_resolution`.')

        positions = []
        for i1,i2 in itertools.combinations(range(len(self.marker_distance_to_origin_xy)), 2):
            args = [self.marker_distance_to_origin_xy[i1], 
                    self.marker_distance_to_origin_xy[i2]]
            positions.append(fsolve(equations, p0, args=args))
        p = np.median(positions, axis=0)

        self.origin_uv = tuple(p)
        
        
    def compute_derivatives(self):
        '''Compute derined values from detected values'''
        
        if self.origin_uv is None:
            raise ValueError('First determine the origin by '
                             'calling `determine_origin`.')

        self.marker_distance_to_origin_uv = [(i, dst * self.resolution) 
                                             for i,dst in self.marker_distance_to_origin_xy]
        self.marker_position_xy = [tuple((np.asarray(uv) - np.asarray(self.origin_uv)) / self.resolution) 
                                   for uv in self.marker_position_uv]

        
    def plot(self, ax=None, crs=True):
        '''Plot the video frame including detected coordinate reference system

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes, optional
          Axes to plot onto
        crs : bool
          Flag to disable plotting of the coordinate reference system

        Returns
        -------
        ax : matplotlib.pyplot.Axes
          Axes containing plot

        '''

        ax = super(VideoFrameCRS, self).plot(ax=ax)
        
        if crs:
            for i, dst in self.marker_distance_to_origin_uv:
                v1, u1 = self.marker_position_uv[i]
                v2, u2 = self.origin_uv
                ax.plot([u1,u2],[v1,v2],'-y')
                ax.annotate('%0.2f' % (dst / self.resolution),
                            xy=((u1+u2)/2,(v1+v2)/2), color='y', ha='center')
            ax.scatter(u2, v2, c='y')

            for (i1,i2), dst in self.marker_distance_to_marker_uv:
                v1, u1 = self.marker_position_uv[i1]
                v2, u2 = self.marker_position_uv[i2]
                ax.plot([u1,u2],[v1,v2],'-or')
                ax.annotate('%0.2f' % (dst / self.resolution),
                            xy=((u1+u2)/2,(v1+v2)/2), color='r', ha='center')
                
            for i, (v, u) in enumerate(self.marker_position_uv):
                ax.annotate(i, xy=(u,v), xytext=(5,5), 
                            textcoords='offset points', color='w')

        return ax

    
    @property
    def x(self):
        '''Returns x-coordinates of given indexes in the first dimension'''
        return CRSConverter(self.shape, self.origin_uv, self.resolution, axis=1)
    
    
    @property
    def y(self):
        '''Returns x-coordinates of given indexes in the second dimension'''
        return CRSConverter(self.shape, self.origin_uv, self.resolution, axis=0)
    
    
    @property
    def xy(self):
        '''Returns x- and y-coordinates of given indexes in the both dimensions'''
        return CRSConverter(self.shape, self.origin_uv, self.resolution, axis=-1)
        
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.n_markers = getattr(obj, 'n_markers', None)
        self.method = getattr(obj, 'method', None)
        self.method_args = getattr(obj, 'method_args', None)
        
        self.marker_distance_to_marker_uv = getattr(obj, 'marker_distance_to_marker_uv', [])
        self.marker_distance_to_marker_xy = getattr(obj, 'marker_distance_to_marker_xy', [])
        self.marker_distance_to_origin_uv = getattr(obj, 'marker_distance_to_origin_uv', [])
        self.marker_distance_to_origin_xy = getattr(obj, 'marker_distance_to_origin_xy', [])
        self.marker_position_uv = getattr(obj, 'marker_position_uv', [])
        self.marker_position_xy = getattr(obj, 'marker_position_xy', [])
        self.origin_uv = getattr(obj, 'origin_uv', None)
        self.origin_xy = getattr(obj, 'origin_xy', (0,0))
        self.resolution = getattr(obj, 'resolution', None)
        self.resolution_rmse = getattr(obj, 'resolution_rmse', None)
        
        self._resolutions = getattr(obj, '_resolutions', [])
