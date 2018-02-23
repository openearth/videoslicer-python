def preprocess_getitem_args(s, shape):
    '''Guarantees to return a slice tuple with ellipses expanded

    Parameters
    ----------
    s : tuple or slice
      Slice or tuple with slices
    shape : tuple
      Shape of the matrix to apply slices on

    Returns
    -------
    s : tuple
      Tuple with expanded slices and ellipses

    '''
    
    if not isinstance(s, tuple):
        s = s,
            
    for i, si in enumerate(s):
        if isinstance(si, type(Ellipsis)):
            n = len(shape) - len(s) + 1
            s = s[:i] + tuple([slice(None)] * n) + s[i+1:]
            break
            
    if any([isinstance(si, type(Ellipsis)) for si in s]):
        raise IndexError('Cannot use ellipsis (...) twice.')
        
    return s
