# -*- coding: utf-8 -*-
"""
 Tools & Utilities for the FlickrLogos-32 dataset.
 See http://www.multimedia-computing.de/flickrlogos/ for details.

 Please cite the following paper in your work:
 Scalable Logo Recognition in Real-World Images
 Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
 ACM International Conference on Multimedia Retrieval 2011 (ICMR11), Trento, April 2011.

 Author: Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de

 Notes:
  - Script was developed/tested on Windows with Python 2.7

 $Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $
 $Rev: 7621 $$Date: 2013-11-18 11:15:33 +0100 (Mo, 18 Nov 2013) $
 $HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/flickrlogos/__init__.py $
 $Id: __init__.py 7621 2013-11-18 10:15:33Z romberg $
"""
import sys

if sys.version_info >= (3,0,0):
    from .flickrlogos import *
    from .core_io import *
else:
    from flickrlogos import *
    from core_io import *
