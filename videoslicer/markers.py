import cv2
import scipy.spatial
import sklearn.cluster
import numpy as np


def find_markers_template(frame, size_min=25, size_max=75,
                          n_markers=4, threshold=.6, max_variance=10):
    '''Detects markers in image frame based on template matching

    The template size is varied to find the best matching size. From
    the best matching template size, all matches are clustered using a
    K-means algorithm in the assumed number of clusters. The centers
    of the clusters are returned as the assumed marker centers.

    Matches are validated against the scatter within a single cluster
    and the interdistance between clusters. If the scatter is too
    large or the interdistance is too small, the match is ignored and
    the next template size is validated.

    Parameters
    ----------
    frame : VideoFrame or np.ndarray
      Image data
    size_min : int, optional
      Minimum size of the template (default: 25)
    size_max : int, optional
      Maximum size of the template (default: 75)
    n_markers : int, optional
      Number of markers in frame (default: 4)
    threshold : float, optional
      Matching threshold for `cv2.matchTemplate` (default: 0.6)
    max_variance : float, optional
      Maximum variance in a cluster of match locations in order to
      assume the match locations belong to the same marker. (default:
      10)

    Returns
    -------
    list of 2-tuples
      List with locations of the marker centers

    '''

    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    marker_position_uv = []
    for sz in range(size_max, size_min, -1):
            
        tmpl = np.zeros((sz,sz,3), dtype=np.uint8) + 255
        tmpl = cv2.circle(tmpl, (sz//2,sz//2), sz//4, (255,0,0), -1)
        matches = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)

        # skip if not enough marker are found
        if np.sum(matches >= threshold) < n_markers:
            continue
            
        # cluster matches based on interdistance
        locs = np.asarray(list(zip(*np.where(matches >= threshold)[::-1])))
        kmeans = sklearn.cluster.KMeans(n_clusters=n_markers).fit(locs)
        kmeans.predict(np.asarray([[0,0]] * n_markers))
        locsc = [locs[kmeans.labels_==i,:] for i in range(n_markers)]

        # skip if variance in distance to cluster center is too large
        var = [np.sum(np.var(locs, axis=0)) for locs in locsc]
        if sum(var) > max_variance:
            continue

        # skip if found markers are too close
        c = kmeans.cluster_centers_
        tree = scipy.spatial.KDTree(c)
        if not all(tree.query(c, k=2)[0][:,1] > sz):
            continue

        # get best match per cluster
        for i, locs in enumerate(locsc):
            vals = [matches[tuple(loc[::-1])] for loc in locs]
            locsc[i] = locs[np.argmax(vals)] + np.asarray([sz//2,sz//2])
        locsc = np.asarray(locsc)[:,::-1]
                
        ix = locsc[:,0].argsort()
        marker_position_uv = [tuple(c) for c in locsc[ix]]

    return marker_position_uv


def find_markers_redness(frame, n_markers=4, min_redness=.5, max_iter=10, max_distance=50):
    '''Detects markers in image frame based on pixel redness

    Red pixels are clustered in a predefined number of marker
    clusters. The centers of the clusters are returned as the assumed
    marker centers.

    After a first estimate is obtained, red pixels far from the
    cluster center that the pixel belongs to are discarded and the
    clustering is recomputed. This procedure is repeated until all red
    pixels are within a given distance from the cluster center and the
    solution converges.

    Parameters
    ----------
    frame : VideoFrame or np.ndarray
      Image data
    n_markers : int, optional
      Number of markers in frame (default: 4)
    min_redness : float, optional
      Minimal value for the redness needed to take a pixel into
      account (default: 0.5)
    max_iter : int, optional
      Maximum number of iterations (default: 10)
    max_distance : int, optional
      Maximum size of markers to consider (default: 50)

    Returns
    -------
    list of 2-tuples
      List with locations of the marker centers

    '''

    # convert image to redness
    redness = frame[...,-1] / frame.sum(axis=-1)

    # find red pixel coordinates
    ix = np.asarray(list(zip(*np.where(redness > min_redness))))

    marker_position_uv = []
    for i in range(max_iter):

        # find cluster centers using kmeans
        kmeans = sklearn.cluster.KMeans(n_clusters=n_markers).fit(ix)
        centers = list(zip(*kmeans.cluster_centers_))

        # find closest cluster center for each red pixel
        distances = []
        for i in range(n_markers):
            center = np.repeat(kmeans.cluster_centers_[i:i+1,:], ix.shape[0], axis=0)
            distances.append(np.sqrt(np.sum((center - ix)**2, axis=1)))
            ix_cluster = np.argmin(np.asarray(distances), axis=0)

        # discard red pixels that are too distant from cluster center
        ix2 = []
        for i in range(n_markers):
            i1 = ix_cluster == i
            i2 = distances[i] < max_distance
            ix2 += list(ix[i1&i2,:])
        ix2 = np.asarray(ix2)

        # exit iterations if nothing changed
        if ix.shape[0] == ix2.shape[0]:
            break

        # recompute cluster centers
        ix = ix2.copy()

    ix = kmeans.cluster_centers_[:,0].argsort()
    marker_position_uv = [tuple(c) for c in kmeans.cluster_centers_[ix]]

    return marker_position_uv

        
