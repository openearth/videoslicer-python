VideoSlicer
===========

This package provides a `VideoSlicer` object that allows you to access
a video file as if it is a matrix. You can create `VideoView` objects
from a `VideoSlicer` object that target specific parts of a video file
using native Python slicing and indexing mechanisms. The `VideoView`
objects can be used to iterate over the frames in arbitrary
dimensions, resulting in `VideoFrame` or `VideoFrameCRS` objects. Both
objects provide a convenient interface to transport, resize, save and
reference frames from the original video file.

As slicing, indexing and iterating is supported in arbitrary
dimensions, the `VideoSlicer` object can be used to quickly extract
timestacks as well as ordinary frames in physical meaningful
coordinates.
