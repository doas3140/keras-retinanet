# Example:
# python prepare.py -f ./../FlickrLogos-v2/ -l ./../Flickr32plus/ -c ./csvpaths/classes.csv -t ./csvpaths/retina-train.csv -v ./csvpaths/retina-valid.csv -s ./csvpaths/retina-test.csv

import numpy as np
import os
import sys
import getopt
import scipy.io
from scripts.data_manipulation import *

# flickr32 functions

# path to groundtruth path
def img_path2gt_path(img_path):
    path_split = img_path.split('/')
    path_split[4] = 'masks'
    path = '/'.join(path_split)
    path += '.bboxes.txt'
    return path

# groundtruth path to groundtruth coordinates
def gt_path2gt(txt_path):
    gt = []
    try:
        with open(txt_path, 'r') as f:
            data = f.readlines()
            for line in data[1:]:
                line = line[:-1]
                coords = [int(i) for i in line.split(' ')]
                gt.append(coords)
    except FileNotFoundError:
        gt = [ ['','','',''] ]
    return gt

# path to label
def path2label(img_path):
    label = img_path.split('/')[5]
    return label

# groundtruth check if isn't outside of image
def check_coords(x1,y1,x2,y2,img_h,img_w):
    if x2 == '' or y2 == '':
        return x1,y1,x2,y2
    if int(x2) >= img_w:
        x2 -= 1
    if int(y2) >= img_h:
        y2 -= 1
    if int(x1) >= img_w:
        x1 -= 1
    if int(y1) >= img_h:
        y1 -= 1
    return x1,y1,x2,y2

# create flickr32 csv
def flickr32_csv(txt_name='all.relpaths.txt', dataset_path = './../FlickrLogos-v2/', txt_out='retina.csv'):
    # get relpaths
    path = dataset_path + txt_name
    with open(path, 'r') as f:
        data = f.readlines()
        relpaths = []
        for line in data:
            relpaths.append(dataset_path+line[:-1])
    
    # get gt paths
    gt_paths = []
    for path in relpaths:
        gt_paths.append( img_path2gt_path(path) )
    # get groundtruths
    GT = []
    for path in gt_paths:
        GT.append( gt_path2gt(path) )
    # get labels
    labels = []
    for path in relpaths:
        labels.append( path2label(path) )
        
    with open(txt_out,'a') as f:
        for i,img_path in enumerate(relpaths):
            img_h,img_w,_ = read_image(img_path).shape
            for gt in GT[i]:
                x1,y1,w,h = gt
                x2,y2 = x1+w,y1+h
                x1,y1,x2,y2 = check_coords(x1,y1,x2,y2,img_h,img_w)
                coords = str(x1)+','+str(y1)+','+str(x2)+','+str(y2)
                if labels[i] == 'no-logo': labels[i] = ''
                f.write(relpaths[i]+','+coords+','+labels[i]+'\n')
            print('FlickLogos32: {}/{} '.format(i, len(relpaths)), end='\r')

# logos32-plus

# get paths
def get_paths_flickr32plus( flickr32plus_folder_path='./../Flickr32plus/' ):
    flickr32plus_folder_path = flickr32plus_folder_path + 'images/'
    class_path_list = os.listdir(flickr32plus_folder_path) # adidas, aldi, apple ...
    X_train_paths = []
    for i,class_folder in enumerate(class_path_list[:]):
        class_folder += '/'
        img_filename_list = os.listdir(flickr32plus_folder_path + class_folder) # 01.jpg, 02.jpg ...
        for j,img_filename in enumerate(img_filename_list[:]):
            img_path = flickr32plus_folder_path + class_folder + img_filename
            X_train_paths.append(img_path)
    return X_train_paths

# get groundtruth
def get_groundtruth_flickr32plus( flickr32plus_folder_path='./../Flickr32plus/' ):
    ''' Get groundtruth patches from .mat file
    Returns:
    GT - dictionary img_number -> groundtruth[]
    '''
    groundtruth_mat_path= flickr32plus_folder_path + 'groundtruth.mat'
    flickr32plus_folder_path = flickr32plus_folder_path + 'images/'
    file = scipy.io.loadmat(groundtruth_mat_path)
    images_array = file['groundtruth'][0]
    GT = dict()
    for i,img_info in enumerate(images_array[:]):
        img_path = img_info[0][0].replace('\\', '/')
        img_path = flickr32plus_folder_path + 'images/' + img_path
        img_number = img_info[0][0].split('\\')[1][:-4] # crops 'class\' and '.jpg'
        groundtruths = img_info[1]
        GT[str(img_number)] = groundtruths.tolist()
    return GT

# get labels
def get_labels_flickr32plus( flickr32plus_folder_path='./../Flickr32plus/' ):
    '''
    Returns:
    flickr32plus - dict (img_number -> label)
    '''
    flickr32plus_folder_path = flickr32plus_folder_path + 'images/'
    class_path_list = os.listdir(flickr32plus_folder_path)
    flickr32plus = dict()
    for i,class_folder in enumerate(class_path_list[:]):
        class_folder += '/'
        img_filename_list = os.listdir(flickr32plus_folder_path + class_folder)
        for j,img_filename in enumerate(img_filename_list[:]):
#             img_path = flickr32plus_folder_path + class_folder + img_filename
            img_number = img_filename[:-4]
            flickr32plus[str(img_number)] = class_folder[:-1]
    return flickr32plus

# create logos32plus csv
def flickr32plus_csv(flickr32plus_folder_path='./../Flickr32plus/', txt_out='retina.csv'):
    # get relpaths
    img_paths = get_paths_flickr32plus(flickr32plus_folder_path)
    # get GT
    GT = get_groundtruth_flickr32plus(flickr32plus_folder_path)
    # get number to label dictionary
    n2l = get_labels_flickr32plus(flickr32plus_folder_path)
    # bad imges
    bad_imgs_i = [815,831,3416,3421,3422,3425,5315,6954]
    
    with open(txt_out,'a') as f:
        for i,_ in enumerate(img_paths):
            if i in bad_imgs_i:
                continue
            img_number = img_paths[i].split('/')[-1][:-4]
            img_label = n2l[img_number]
            if img_label == 'guinness':
                img_label = 'guiness'
                
            img_h,img_w,_ = read_image(img_paths[i]).shape
            for gt in GT[img_number]:
                x1,y1,w,h = gt
                x2,y2 = x1+w,y1+h
                x1,y1,x2,y2 = check_coords(x1,y1,x2,y2,img_h,img_w)
                coords = str(int(x1))+','+str(int(y1))+','+str(int(x2))+','+str(int(y2))
                f.write(img_paths[i]+','+coords+','+img_label+'\n')

            print('Logos32plus: {}/{} '.format(i, len(img_paths)), end='\r')


# create class csv
def class_csv(dataset_path, txt_out):
    class_list = os.listdir(dataset_path+'classes/jpg/')
    # del no-logo
    for i,cl in enumerate(class_list):
        if cl == 'no-logo':
            del class_list[i]
    # create csv
    with open(txt_out,'w') as f:
        for i,class_name in enumerate(class_list):
            f.write(class_name+','+str(i)+'\n')

# parse argv
def parse(argv):
    p = dict()

    try:
        opts, args = getopt.getopt(argv,'f:l:c:t:v:s:',['-f=','-l=','-c=','-t=','-v=','-s='])
    except getopt.GetoptError:
        print('see readme for instructions') 
        sys.exit(2)

    for opt,arg in opts:
        if opt == '-h':
            print('args: read instructions...')
            sys.exit()
        elif opt in ('-f'):
            p['flicker32path'] = arg
        elif opt in ('-l'):
            p['logos32path'] = arg
        elif opt in ('-c'):
            p['class_name'] = arg
        elif opt in ('-t'):
            p['train_name'] = arg
        elif opt in ('-v'):
            p['valid_name'] = arg
        elif opt in ('-s'):
            p['test_name'] = arg
        
    return p

#
def main(argv):
    p = parse(argv)

    class_csv(dataset_path=p['flicker32path'], txt_out=p['class_name'])
    flickr32_csv(txt_name='valset.relpaths.txt', txt_out=p['valid_name'])
    print()
    flickr32_csv(txt_name='trainset.relpaths.txt', txt_out=p['train_name'])
    print()
    flickr32_csv(txt_name='testset.relpaths.txt', txt_out=p['test_name'])
    print()
    flickr32plus_csv(txt_out=p['train_name'])
    


if __name__ == '__main__':
    main(sys.argv[1:])




















































