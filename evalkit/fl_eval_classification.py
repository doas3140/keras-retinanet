# -*- coding: utf-8 -*-
"""
 Computes scores for recognition results on the FlickrLogos-32 dataset.
 See http://www.multimedia-computing.de/flickrlogos/ for details.

 Please cite the following paper in your work:
 Scalable Logo Recognition in Real-World Images
 Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
 ACM International Conference on Multimedia Retrieval 2011 (ICMR11), Trento, April 2011.

 Author:   Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de

 Notes:
  - Script was developed/tested on Windows with Python 2.7

 $Date: 2013-12-19 09:54:21 +0100 (Do, 19 Dez 2013) $
 $Rev: 7692 $$Date: 2013-12-19 09:54:21 +0100 (Do, 19 Dez 2013) $
 $HeadURL: https://137.250.173.47:8443/svn/romberg/trunk/romberg/research/FlickrLogos-32_SDK/FlickrLogos-32_SDK-1.0.4/scripts/fl_eval_classification.py $
 $Id: fl_eval_classification.py 7692 2013-12-19 08:54:21Z romberg $
"""
__version__ = "$Id: fl_eval_classification.py 7692 2013-12-19 08:54:21Z romberg $"
__author__  = "Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de"

import sys, time
from os.path import exists, isdir, normpath

from flickrlogos import fl_read_groundtruth2, fl_read_csv, Tee

#==============================================================================
#
#==============================================================================

def sround(x, arg):
    if isinstance(x, float):
        return round(x, arg)
    else:
        return x

def fl_eval_classification(flickrlogos_dir, detection_file, verbose):
    """Computes scores for classification/recognition results."""
    #==========================================================================
    # check input
    #==========================================================================
    flickrlogos_dir = normpath(flickrlogos_dir)
    detection_file  = normpath(detection_file)
    
    if not exists(flickrlogos_dir):
        print("ERROR: fl_eval_classification(): Directory given by --flickrlogos does not exist: '"+str(flickrlogos_dir)+"'")
        exit(1)

    if not exists(detection_file):
        print("ERROR: detection_file: File '"+detection_file+"' does not exist!\n")
        exit(1)

    if not flickrlogos_dir.endswith('/') and not flickrlogos_dir.endswith('\\'):
        flickrlogos_dir += '/'

    gt_all_file     = normpath(flickrlogos_dir + "all.txt")
    gt_train_file   = normpath(flickrlogos_dir + "trainset.txt")
    gt_valset_file  = normpath(flickrlogos_dir + "valset.txt")
    gt_testset_file = normpath(flickrlogos_dir + "testset.txt")

    if not exists(gt_all_file):
        print("ERROR: fl_eval_retrieval(): Ground truth file does not exist: '"+str(gt_all_file)+"'")
        exit(1)
    if not exists(gt_train_file):
        print("ERROR: fl_eval_retrieval(): Ground truth file does not exist: '"+str(gt_train_file)+"'")
        exit(1)
    if not exists(gt_valset_file):
        print("ERROR: fl_eval_retrieval(): Ground truth file does not exist: '"+str(gt_valset_file)+"'")
        exit(1)
    if not exists(gt_testset_file):
        print("ERROR: fl_eval_retrieval(): Ground truth file does not exist: '"+str(gt_testset_file)+"'")
        exit(1)

    #==========================================================================
    # load ground truth
    # NOTE: Not all ground truth files are actually required to compute the scores
    #==========================================================================
    # ALL: Training set + Validation set + Test set
    gt_all_map_img2class, gt_all_map_class2imgs, gt_all_class_names = fl_read_groundtruth2(gt_all_file)
    assert len(gt_all_class_names) == 33
    assert len(gt_all_map_img2class) == 8240
    assert len(gt_all_map_class2imgs) == 33

    # Training set
    gt_train_map_img2class, gt_train_map_class2imgs, gt_train_class_names = fl_read_groundtruth2(gt_train_file)
    assert len(gt_train_class_names) == 32
    assert len(gt_train_map_img2class) == 320
    assert len(gt_train_map_class2imgs) == 32

    # Validation set
    gt_val_map_img2class, gt_val_map_class2imgs, gt_val_val_class_names = fl_read_groundtruth2(gt_valset_file)
    assert len(gt_val_val_class_names) == 33
    assert len(gt_val_map_img2class) == 3960
    assert len(gt_val_map_class2imgs) == 33

    # Test set
    gt_test_map_img2class, gt_test_map_class2imgs, gt_test_class_names = fl_read_groundtruth2(gt_testset_file)
    assert len(gt_test_class_names) == 33
    assert len(gt_test_map_img2class) == 3960
    assert len(gt_test_map_class2imgs) == 33

    #==========================================================================
    # load detection results and normalize all class names to lower case
    #==========================================================================
    actual_data = fl_read_csv(detection_file)
    actual_data = [(img, detected_class.lower(), conf) for img, detected_class, conf in actual_data ]
        
    #==========================================================================
    # parse actual_data and compute scores
    #==========================================================================
    compute_scores(gt_all_map_img2class, actual_data, gt_all_file, detection_file, verbose)

#==============================================================================
#
#==============================================================================
def compute_scores(groundtruth, actual_data, groundtruth_file=None, detection_file=None, verbose=False):
    """Computes several scores for classification/recognition results."""

    assert len(groundtruth) > 0
    assert len(actual_data) > 0
    assert isinstance(groundtruth, dict)
    assert isinstance(actual_data, list)

    #==========================================================================
    # analyze results
    #==========================================================================
    count_bad                               = 0.0
    count_gt_logos                          = 0.0
    count_gt_nonlogos                       = 0.0

    # classification
    count_classified_nonlogo_as_nonlogo     = 0.0
    count_classified_nonlogo_as_logo        = 0.0
    count_classified_logo_as_nonlogo        = 0.0
    count_classified_logo_as_logo           = 0.0

    # detection
    count_identified_logo_x_as_x            = 0.0
    count_identified_logo_x_as_y            = 0.0

    # the confidence of the detection is not taken into account in this evaluation
    for img, detected_class, confidence in actual_data:
        im = img + ".jpg"
        if not im in groundtruth:
            print("ERROR: Ground truth does not contain key: '"+str(im)+"'. Skipped.")
            continue

        real_class = groundtruth[im]

        # count number of images that were actually used in this run
        if real_class == "no-logo":
            count_gt_nonlogos += 1
        else:
            count_gt_logos += 1

        if detected_class == "bad":
            count_bad += 1
            continue

        # ----------------------------------------------------
        if real_class == "no-logo":
            if detected_class == "no-logo":
                count_classified_nonlogo_as_nonlogo += 1
            else:
                count_classified_nonlogo_as_logo += 1

        # ----------------------------------------------------
        else: # real_class != "no-logo"
            if detected_class == "no-logo":
                count_classified_logo_as_nonlogo += 1
            else:
                count_classified_logo_as_logo += 1

                if detected_class == real_class:
                    count_identified_logo_x_as_x += 1
                else:
                    count_identified_logo_x_as_y += 1

    count_classified_as_nonlogos = count_classified_nonlogo_as_nonlogo + count_classified_logo_as_nonlogo
    count_classified_as_logos    = count_classified_logo_as_logo       + count_classified_nonlogo_as_logo

    #==========================================================================
    # compute scores for detection
    #==========================================================================
    #    Precision   = TP / (TP+FP)
    #    Recall      = TP / (TP+FN)
    #    Specificity = TN / (TN+FP)
    #    Accuracy    = (TP+TN) / (TP+FP+FN+TN)
    TP = float(count_classified_logo_as_logo)
    TN = float(count_classified_nonlogo_as_nonlogo)
    FP = float(count_classified_nonlogo_as_logo)
    FN = float(count_classified_logo_as_nonlogo)

    detection_precision    = "DivByZero"
    detection_recall       = "DivByZero"
    detection_specificity  = "DivByZero"
    detection_accuracy     = "DivByZero"
    FalsePositiveRate      = "DivByZero"
    FalseNegativeRate      = "DivByZero"

    if (TP+FP) > 0:       detection_precision   = TP / (TP+FP)
    if (TP+FN) > 0:       detection_recall      = TP / (TP+FN)
    if (TN+FP) > 0:       detection_specificity = TN / (TN+FP)
    if (TP+FP+FN+TN) > 0: detection_accuracy    = (TP+TN) / (TP+FP+FN+TN)
    if (FP + TN) > 0:     FalsePositiveRate     = FP / (FP + TN)
    if (TP + FN) > 0:     FalseNegativeRate     = FN / (TP + FN)

    #==========================================================================
    # compute scores for recognition
    #==========================================================================
    recognition_precision   = count_identified_logo_x_as_x / (count_identified_logo_x_as_x + count_identified_logo_x_as_y)
    recognition_recall      = count_identified_logo_x_as_x / count_gt_logos
    recognition_accuracy    = ( (count_identified_logo_x_as_x + 
                                 count_classified_nonlogo_as_nonlogo )
                               / (count_gt_logos + count_gt_nonlogos ) )

    #==========================================================================
    # output
    #==========================================================================
    print("---------------------------------------------------------------------------")
    print(" RESULTS ")
    print("---------------------------------------------------------------------------")
    print("Ground truth:")
    if groundtruth_file is not None:
        print("  Ground truth file: '"+groundtruth_file+"'")
    print("  Total number of images...............................:   "+str(len(actual_data)).rjust(5))
    print(" ")
    print("Input")
    if detection_file is not None:
        print("  Result file: '"+detection_file+"'")
    print("  Result file: Results for logo images ................:   "+str(int(count_gt_logos)).rjust(5))
    print("  Result file: Results for non-logo images.............:   "+str(int(count_gt_nonlogos)).rjust(5))
    print("  Bad images (excluded from computing scores)..........:   "+str(int(count_bad)).rjust(5))
    print(" ")
    
    if verbose: # usually these scores are not needed for evaluation
        print("Detection: (\"Is a logo present: Yes/No?\")")
        print("  Bad images (excluded from computing scores)..........:   "+str(int(count_bad)).rjust(5))
        print(" ")
        print("  TP = count_classified_logo_as_logo...................:   "+(str(int(count_classified_logo_as_logo)).rjust(5)))
        print("  TN = count_classified_nonlogo_as_nonlogo.............:   "+(str(int(count_classified_nonlogo_as_nonlogo)).rjust(5)))
        print("  FP = count_classified_nonlogo_as_logo................:   "+(str(int(count_classified_nonlogo_as_logo)).rjust(5)))
        print("  FN = count_classified_logo_as_nonlogo................:   "+(str(int(count_classified_logo_as_nonlogo)).rjust(5)))
        print(" ")
        print("  detection_precision..................................:   "+str(sround(detection_precision,3)))
        print("  detection_recall.....................................:   "+str(sround(detection_recall,3)))
        print("  detection_specificity................................:   "+str(sround(detection_specificity,3)))
        print("  detection_accuracy...................................:   "+str(sround(detection_accuracy,3)))
        print(" ")
        print("  True positive rate  = Recall ........................:   "+str(sround(detection_recall,3)))
        print("  True negative rate  = Specificity ...................:   "+str(sround(detection_specificity,3)))
        print("  False positive rate = FP / (FP + TN) ................:   "+str(sround(FalsePositiveRate,3)))
        print("  False negative rate = FN / (TP + FN) ................:   "+str(sround(FalseNegativeRate,3)))
        print(" ")
        
    # main scores
    print("Recognition: (\"If a logo is present of which class is it?\")")
    print("  recognition_precision................................:   "+str(sround(recognition_precision,3)))
    print("  recognition_recall...................................:   "+str(sround(recognition_recall,3)))
    print("  recognition_accuracy.................................:   "+str(sround(recognition_accuracy,3)))
    if verbose:
        print("\nDate/Time: " + time.asctime())
        
    print("---------------------------------------------------------------------------")

    return

#==============================================================================
if __name__ == '__main__': # MAIN
#============================================================================== 
    from optparse import OptionParser
    usage = "Usage: %prog --flickrlogos=<dataset root dir> --classification=<file with classification results> "
    parser = OptionParser(usage=usage)

    parser.add_option("--flickrlogos", dest="flickrlogos", type=str, default="",
                      help="Base (root) directory of the FlickrLogos-32 dataset\n")
    parser.add_option("--classification", dest="classification", type=str, default="",
                      help= """File classification results: "\
                      "contains the file names of the images and the corresponding "\
                      "detected classes in the following format: "\
                      "<image id>, <detected class name>, <confidence value or "\
                      "1 if class was detected, 0 otherwise> \n""")
    parser.add_option("-o","--output", dest="output", type=str, default="-",
                      help= "Output file, can be '-' for stdout. Default: stdout \n""")
    parser.add_option("-v","--verbose", dest="verbose", action="store_true", default=False, 
                      help="Optional: Flag for verbose output. Default: False\n""")
    (options, args) = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        exit(1)

    #==========================================================================
    # show passed args
    #==========================================================================
    if options.verbose:
        print("fl_eval_classification.py\n"+__version__)
        print("-"*79)
        print("ARGS:")
        print("FlickrLogos root dir (--flickrlogos):")
        print("  > '"+options.flickrlogos+"'")
        print("Result file (--classification):")
        print("  > '"+options.classification+"'")
        print("Output file ( --output):")
        print("  > '"+options.output+"'")
        print("-"*79)
    
    if options.flickrlogos is None or options.flickrlogos == "":
        print("Missing argument --flickrlogos=<file>")
        exit(1)
        
    if options.classification is None or options.classification == "":
        print("Missing argument --classification=<file with classification results>")
        exit(1)
        
    #==========================================================================
    # if output is a file and not "-" then all print() statements are redirected
    # to *both* stdout and a file.
    #==========================================================================
    if options.output is not None and options.output != "" and options.output != "-":
        if isdir(options.output):
            print("Invalid argument: Arg --output must denote a file. Exit.")
            exit(1)

        Tee(options.output, "w")

    #==========================================================================
    # compute scores
    #==========================================================================
    fl_eval_classification(options.flickrlogos, options.classification, options.verbose)

