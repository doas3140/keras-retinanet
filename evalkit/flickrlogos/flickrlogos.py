# -*- coding: utf-8 -*-
"""
 Tools for the FlickrLogos-32 dataset.
 See http://www.multimedia-computing.de/flickrlogos/ for details.

 Please cite the following paper in your work:
 Scalable Logo Recognition in Real-World Images
 Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
 ACM International Conference on Multimedia Retrieval 2011 (ICMR11), Trento, April 2011.

 Author:   Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de

 Notes:
  - Script was developed/tested on Windows with Python 2.7

 $Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $
 $Rev: 7621 $$Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $
 $HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/flickrlogos/flickrlogos.py $
 $Id: flickrlogos.py 7621 2013-11-18 10:15:33Z romberg $
"""
__version__ = "$Rev: 7621 $$Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $"

import sys
from os.path import exists
from collections import defaultdict
from collections import OrderedDict
from math import sqrt

#==============================================================================
#
#==============================================================================

def fl_msplit(s, delimiters=";,\t"):
    """Splits the given string by any of the given delimiters."""
    if len(s) == 0:
        return []

    delim = delimiters[0]
    for d in delimiters[1:]:
        s = s.replace(d, delim)

    tokens = s.split(delim)
    #print("tokens = "+str(tokens))
    return [ t.strip() for t in tokens if len(t) > 0 ]


def fl_read_csv(filename, delimiters=",\t;"):
    """Reads a CSV file and returns a list of list holding the data."""

    data = []
    with open(filename, "r") as f:
        for line in f:
            tokens = fl_msplit(line, delimiters)
            data.append(tokens)
    return data

def fl_read_groundtruth(groundtruth_file):
    """Reads the FlickrLogos-32 ground truth.

    Returns a map <image> -> <object class>.
    """
    if not exists(groundtruth_file):
        print("ERROR: fl_read_groundtruth(): Arg groundtruth_file: "\
              "File '"+groundtruth_file+"' does not exist!\n")
        exit(1)

    gt_data = fl_read_csv(groundtruth_file, delimiters=" \t,;")

    groundtruth = OrderedDict()
    for classname, image in gt_data:
        groundtruth[image] = classname.lower()

    class_names = sorted(list(set(groundtruth.values())))
    return groundtruth, class_names


def fl_read_groundtruth2(groundtruth_file):
    """Reads the FlickrLogos-32 ground truth.

    Returns a 3-tuple with
           (1) a map: <image> -> <object class>,
           (2) a map: <object class> -> [list of images],
           (3) a list: [sorted list of class names]
    """

    if not exists(groundtruth_file):
        print("ERROR: fl_read_groundtruth2(): Arg groundtruth_file: "\
              "File '" + groundtruth_file + "' does not exist!\n")
        exit(1)

    gt_data = fl_read_csv(groundtruth_file)

    gt_map_img2class  = OrderedDict()
    gt_map_class2imgs = defaultdict(list)

    for classname, image in gt_data:
        gt_map_img2class[image] = classname.lower()
        gt_map_class2imgs[classname].append(image)

    class_names = sorted(list(set(gt_map_img2class.values())))
    return gt_map_img2class, gt_map_class2imgs, class_names

#==============================================================================
#
#==============================================================================

def fl_ap(pos, amb, ranked_list):
    """Computes the average precision (AP) for an ordered list of retrieval results.

    pos: set of positives images expected (ground truth)
    amb: set of images ignores in this retrieval (query images + null images)
    ranked_list: list of images retrieved sorted by similarity, most similar images come first

    Code adapted from http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/

    Examples/Tests:

    >>> pos         = set([0, 1, 3])
    >>> amb         = set()
    >>> ranked_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> fl_ap(pos, amb, ranked_list)
    0.9027777777777777

    >>> pos         = set([0, 3, 7])
    >>> amb         = set()
    >>> ranked_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> fl_ap(pos, amb, ranked_list)
    0.5823412698412699

    >>> pos         = set([0, 3, 7])
    >>> amb         = set()
    >>> ranked_list = []
    >>> fl_ap(pos, amb, ranked_list)
    0.0

    >>> pos         = set([3])
    >>> amb         = set()
    >>> ranked_list = [3]
    >>> fl_ap(pos, amb, ranked_list)
    1.0

    >>> pos         = set([3])
    >>> amb         = set([3])
    >>> ranked_list = [3]
    >>> fl_ap(pos, amb, ranked_list)
    0.0
    """
    assert isinstance(pos, set)
    assert isinstance(amb, set)
    assert isinstance(ranked_list, list)

    old_recall    = 0.0
    old_precision = 1.0
    ap = 0.0

    intersect_size = 0
    j = 0

    for x in ranked_list:
        if x in amb:
            continue
        if x in pos:
            intersect_size += 1

        recall    = intersect_size / float(len(pos))
        precision = intersect_size / (j + 1.0)

        ap += (recall - old_recall) * ((old_precision + precision)/2.0)

        old_recall    = recall
        old_precision = precision
        j += 1

    return ap

def fl_mean(l):
    """Computes the mean.

    >>> import numpy as np
    >>> a = np.array([-1, 0, 3, 4])
    >>> np.mean(a)
    1.5
    >>> fl_mean(a.tolist())
    1.5
    >>> fl_mean([])
    0.0
    >>> fl_mean([7])
    7.0
    """
    mean = 0.0
    for x in l:
        mean += x

    N = float(len(l))
    if N != 0:
        mean /= N
    return mean

def fl_sdev(l):
    """Computes the standard deviation.

    >>> import numpy as np
    >>> a = np.array([-1, 0, 3, 4])
    >>> np.std(a)
    2.0615528128088303
    >>> np.std(a, ddof=1)
    2.3804761428476167
    >>> fl_sdev(a.tolist())
    2.3804761428476167
    >>> fl_sdev([4])
    0.0
    """
    mean = fl_mean(l)
    sdev = 0.0
    for x in l:
        diff = x - mean
        sdev += diff*diff

    N = float(len(l))
    if N > 1:
        sdev = sqrt( sdev / (N-1.0) )

    return sdev

#==============================================================================
# Misc
#==============================================================================

class Tee(object):
    """Simulates the behaviour of the unix program 'tee' to write output
    both to stdout *and* a file.
    """

    def __init__(self, filename, mode):
        self.file = None
        if filename is not None and filename != "" and filename != "-":
            self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass
        try:
            sys.stdout = sys.__stdout__
        except Exception:
            pass

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        sys.__stdout__.write(data)

#==============================================================================
#
#==============================================================================
if __name__ == "__main__":

    # run all doctests
    import doctest
    doctest.testmod()

    print("All doctests passed.")
