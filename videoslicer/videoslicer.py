import cv2
import numpy as np

from videoslicer.utils import *


class VideoSlicer(object):
    '''VideoSlicer class

    Reads dimensions from a video file and allows for slicing and
    iterating over the video data in arbitrary axis. Any slicing or
    iteration action will result in a VideoSlice object targeted at a
    specific part of the video data.

    Slicing or iterating over the time dimension is both optimized for
    speed and memory usage. Slicing and iterating over spatial
    dimensions can be either optimized for speed (run over video once,
    collecting all data), or for memory usage (run over video multiple
    times, each time collecting the required amount of data).

    Examples
    --------
    >>> slicer = VideoSlicer('movie.avi')
    >>> view = slicer[:10,::10,::10]
    >>> for frame in view:
          frame.save('frame{:06d}.jpg'.format(frame.index))

    >>> slicer = VideoSlicer('movie.avi')
    >>> frame = slicer[10,...]
    >>> frame.save('frame.jpg')
    >>> frame.plot()

    >>> slicer = VideoSlicer('movie.avi', axis=2) # loop over horizontal dimension
    >>> for frame in slicer[:,:,::10]:
          frame.T.save('timestack{:06d}.jpg'.format(frame.index)) # transpose to have time on the horizontal axis

    See Also
    --------
    VideoView

    '''
    
    
    def __init__(self, filename, axis=0, optimize='memory'):
        '''Initialization

        Parameters
        ----------
        filename : str
          Path to video file
        axis : int, optional
          Iteration axis (default: 0)
        optimize : str, optional
          Optimization method (speed or memory; default: memory)

        '''
        
        self.filename = filename
        self.axis = axis
        self.optimize = optimize
        
        self.buffer = cv2.VideoCapture(filename)
        if not self.buffer.isOpened():
            raise IOError('Cannot open video file: {}'.format(filename))
            
        self.fps = self.buffer.get(cv2.CAP_PROP_FPS)
        self.nt = int(self.buffer.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ny = int(self.buffer.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nx = int(self.buffer.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nd = 3
        
        
    def __repr__(self):
        return '{}(frames={:d}, shape={:d}x{:d}px, depth={:d}, fps={:0.0f}, axis={:d})'.format(
            self.__class__.__name__, *self.shape, self.fps, self.axis
        )
    
        
    def __str__(self):
        return self.__repr__()
    
    
    def __enter__(self):
        pass
    
    
    def __exit__(self):
        self.close()
        
        
    def __iter__(self):
        for frame in self.get_view():
            yield frame
        
        
    def __getitem__(self, s):
        '''Returns a VideoView object given the provided slicing

        See `get_view` for details.

        '''

        s = preprocess_getitem_args(s, self.shape)
        
        if (len(s) < 3 or len(s) > 4):
            raise IndexError('Invalid dimensions for video data ({:d}), '
                             'should be 3 or 4.'.format(len(s)))
            
        return self.get_view(s)
    
    
    def close(self):
        '''Release video file handler'''
        if self.buffer.isOpened():
            self.buffer.release()
            
            
    def get_view(self, slices=()):
        '''Returns a VideoView object given the provided slicing

        Parameters
        ----------
        slices : tuple, optional
          Tuple with slice objects and/or indices that define the
          required data view

        Returns
        -------
        VideoView
          VideoView object that adheres to the provided slices

        See Also
        --------
        VideoView

        '''
        
        return VideoView(self.buffer, self.shape, slices, 
                         fps=self.fps, axis=self.axis, optimize=self.optimize)
    
    
    @property
    def time(self):
        '''Returns the time axis in seconds for the open video file'''
        return self.get_view().time
    
    
    @property
    def shape(self):
        '''Returns the shape of the open video file'''
        return (
            self.nt,
            self.ny,
            self.nx,
            self.nd
        )
    
    
class VideoView(object):
    '''VideoView class

    Provides a generator over a specific part of a video file. The
    generator iterates over a given axis and returns individual frames
    in the remaining dimensions. By default, the generator iterates
    over time and returns standard video frames. Alternatively, the
    generator can iterate over space or depth, returning timestacks
    from the video file.

    If the slicing is chosen such that there is only one frame in the
    iteration axis, the VideoView falls back to being a VideoFrame.

    Slicing or iterating over the time dimension is both optimized for
    speed and memory usage. Slicing and iterating over spatial
    dimensions can be either optimized for speed (run over video once,
    collecting all data), or for memory usage (run over video multiple
    times, each time collecting the required amount of data).

    See Also
    --------
    VideoSlicer
    VideoFrame

    '''
    
    
    def __new__(cls, buffer, shape, slices, fps=None, axis=0, optimize='memory'):
        '''Constructor

        Returns a VideoView object, unless the view consists of a
        single frame in the iteration axis, then it falls back to
        being a VideoFrame object.

        Parameters
        ----------
        buffer : int
          Open video file buffer
        shape : tuple
          Shape of video file
        slices : tuple
          Tuple of slices definign the view
        fps : int, optional
          Frame rate of video
        axis : int, optional
          Iteration axis (default: 0)
        optimize : str, optional
          Optimization method (speed or memory; default: memory)

        '''
        
        if len(slices) > len(shape):
            raise ValueError('Number of slices should not exceed number of dimensions.')
        
        obj = super(VideoView, cls).__new__(cls)
        obj.buffer = buffer
        obj._shape = shape
        obj.axis = axis
        obj.optimize = optimize
        obj.fps = fps
    
        obj.slices = [slices[i] if len(slices)>i else None
                      for i in range(len(shape))]

        # if the slice in the iteration axis only contains a single
        # frame, fall back to being a VideoFrame object
        primary_slice = obj.iterable_slices()[axis]
        if len(primary_slice) == 1:
            return VideoFrame(list(obj)[0], index=primary_slice[0])
        else:
            return obj

        
    def __repr__(self):
        return '{}(frames={:d}, shape={:d}x{:d}px, depth={:d})'.format(self.__class__.__name__, *self.shape)
    
        
    def __str__(self):
        return self.__repr__()

    
    def __iter__(self):
        for frame in self.generator():
            yield frame
            

    def generator(self, slices=None, axis=None):
        '''Generator method that provides individual VideoFrame objects along a given axis

        Parameters
        ----------
        slices : tuple, optional
          Tuple with slices to overwrite the object's default slicing
        axis : int, optional
          Iteration axis to overwrite the object's default iteration
          axis

        See Also
        --------
        VideoFrame

        '''

        if slices is None:
            slices = self.slices
        if axis is None:
            axis = self.axis
            
        indices = self.iterable_slices(slices)

        if axis == 0:
            # iteration over time
            for i in indices[0]:
                self.buffer.set(cv2.CAP_PROP_POS_FRAMES, i)
                rc, frame = self.buffer.read()
                if rc:
                    # sequentially apply slicing
                    frame = frame[indices[1],:,:]
                    frame = frame[:,indices[2],:]
                    frame = frame[:,:,indices[3]]
                    yield VideoFrame(frame, index=i)
                else:
                    yield VideoFrame(self._empty(*indices[1:]), index=i)
        elif axis > 0:
            # iteration in space
            if self.optimize == 'speed':
                # speed optimized: first collect all data by iterating
                # over time, and then construct the generator
                frames = np.asarray(list(self.generator(axis=0)))
                indices2 = [slice(None)] * 4
                for i,n in enumerate(indices[axis]):
                    indices2[axis] = i
                    yield VideoFrame(frames[indices2], index=n)
            elif self.optimize == 'memory':
                # memory optimized: iterate over time, collecting only
                # the data required for the current video frame
                for i in indices[axis]:
                    slices[axis] = i
                    yield VideoFrame(list(self.generator(slices=slices, axis=0)), index=i)
            else:
                raise ValueError('Optimization not supported: {}'.format(self.optimize))
                
    
    def iterable_slices(self, slices=None):
        '''Converts arbitrary slices into iterable lists, given the video dimensions

        Parameters
        ----------
        slices : tuple, optional
          Tuple of slices to overwrite the object's default slicing

        Returns
        -------
        list
          List with lists with indices corresponding to the provided
          slices and shape of the video file

        '''

        if slices is None:
            slices = self.slices
        return [self._iterable_slice(s1, s2) 
                for s1, s2 in zip(self._shape, slices)]
    
    
    @property
    def time(self):
        '''Returns the time axis in seconds for the video view'''
        return np.asarray(self.iterable_slices()[0]) / self.fps

    
    @property
    def shape(self):
        '''Returns the shape of the video view'''
        return tuple([len(x)
                      for x in self.iterable_slices()
                      if not isinstance(x, int)])
    
    
    @staticmethod
    def _empty(*args):
        '''Creates and empty frame'''
        return np.nan + np.zeros([len(a) for a in args])
    
    
    @staticmethod
    def _iterable_slice(n, s):
        '''Returns an iterable from a slice and a length

        Parameters
        ----------
        n : int
          Length
        s : slice
          Slice

        Returns
        -------
        slice

        '''
        
        i = np.asarray(list(range(n)))
        if s is not None:
            i = i[s]
        return list(i)
    
    
class VideoFrame(np.ndarray):
    '''VideoFrame class

    A `numpy.ndarray` with video frame specific methods: `T`
    (transpose), `resize`, `plot` and `save`. The dimensions of the
    array need to be valid image dimensions (i.e. 2 or 3) and the
    depth dimension need to be 1, 3 or 4 depending on the number of
    color and alpha channels.

    With respect to the original `numpy.ndarray` an additional
    attribute `index` contains the index of the current frame in the
    original video file.

    See Also
    --------
    VideoView

    '''
    

    def __new__(cls, arr, index=None, *args, **kwargs):
        '''Constructor

        Parameters
        ----------
        arr : iterable
          Image data
        index : int, optional
          Index of video frame in original video file

        '''
        
        arr = np.asarray(arr)
        
        if arr.ndim < 2 or arr.ndim > 3:
            raise TypeError('Invalid dimensions for image data ({:d}), '
                            'should have 2 or 3 non-unity dimensions.'.format(arr.ndim))
        if arr.ndim == 3 and arr.shape[2] not in [1, 3, 4]:
            raise TypeError('Invalid third dimension for image data ({:d}), '
                            'should be 1, 3 or 4.'.format(arr.shape[2]))
            
        if arr.shape[-1] == 1:
            arr = arr.squeeze()

        obj = arr.view(cls, *args, **kwargs)
        obj.index = index

        return obj
    
    
    def set_index(self, index):
        '''Set index of video frame in original video file'''
        self.index = index
    
    
    def resize(self, shape):
        '''Resize video frame to given dimensions
        
        Parameters
        ----------
        shape : tuple
          Target image shape

        '''
        
        arr = cv2.resize(self, shape[::-1])
        obj = self.__class__(arr)
        obj.__array_finalize__(self)
        return obj

    
    @classmethod
    def read(cls, filename, **kwargs):
        '''Initialize VideoFrame object from image file

        Parameters
        ----------
        filename : str
          Path to image file

        Returns
        -------
        VideoFrame
          VideoFrame object with image data

        '''
        
        return cls(cv2.imread(filename), **kwargs)
        
        
    def write(self, filename):
        '''Write VideoFrame to image file

        Parameters
        ----------
        filename : str
          Path to image file

        '''
        
        cv2.imwrite(filename, self)
        
        
    def plot(self, ax=None):
        '''Plot image

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes, optional
          Axes to plot onto

        Returns
        -------
        ax : matplotlib.pyplot.Axes
          Axes containing plot

        '''
        
        import matplotlib.pyplot as plt
        if ax is None: fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(self, cv2.COLOR_BGR2RGB))
        return ax
        
        
    @property
    def T(self):
        '''Returns the transposed image'''
        return np.swapaxes(self, 0, 1)
    
    
    @property
    def shape(self):
        '''Returns the shape of the image (always 3 dimensions)'''
        shp = super(VideoFrame, self).shape
        if len(shp) < 3:
            return shp + (1,)
        else:
            return shp
        
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.index = getattr(obj, 'index', None)
