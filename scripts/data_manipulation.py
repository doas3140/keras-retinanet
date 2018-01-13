'''
Image
'''
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def read_image( image_path ):
    return np.array(Image.open( image_path ))

def contrast_normalization( X, lmda=10, s=1, epsilon=0 ):
    X_avg = np.mean(X)
    X = X - X_avg
    contrast = np.sqrt( lmda + np.mean(X**2) )
    return s*X / max(contrast,epsilon)

def get_rects_from_image( image_path, rects, img_size=(32,32) ):
    '''
    Arguments:
    rects - [X,Y,W,H]
    Returns:
    rectangles - np.array ()
    '''
    image = read_image( image_path )
    rectangles = []
    for r in rects:
        x,y,w,h = r
        rectangle = image[ y:y+h , x:x+w ]
        rectangle = cv2.resize(rectangle, img_size) # resize
		# normalization
        #rectangle = rectangle/255.
        rectangle = contrast_normalization( rectangle, lmda=10, s=1, epsilon=0 )
        rectangles.append(rectangle)
    rectangles = np.array(rectangles)
    return rectangles

def show_image( image, rects=None ):
    """ Shows image w/ or w/out rectangles """
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    if rects is not None:
        for r in rects:
            X,Y,W,H = r[0],r[1],r[2],r[3]
            rectangle = patches.Rectangle((X,Y),W,H,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rectangle)
    plt.show()

'''
CSV
'''
import csv

def write_array_to_csv( csv_path, some_list ):
    '''
    Arguments:
    pos_instances - array of instance tuples ('label', [x,y,w,h])
    '''
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(some_list)

def read_rects_from_csv( csv_path ):
    '''
    Arguments:
    csv_file, that has list of rect coordinates [tl_x, tl_y, w, h] # top left
    Returns:
    rectangles - list of rects [X,Y,W,H]
    '''
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rectangle_coords = list(reader)
        rectangle_coords = [list(map(int,r)) for r in rectangle_coords] # convert strings to ints in list
        rectangles = [ [ r[0],r[1],r[2],r[3] ] for r in rectangle_coords ]
    return rectangles

'''
Flickr32plus
'''
import scipy.io

def get_train_paths_flickr32plus( flickr32plus_folder_path ):
    ''' Iterate through every class folder and create train data
    Returns:
    X_train_paths - list
    '''
    class_path_list = os.listdir(flickr32plus_folder_path) # adidas, aldi, apple ...
    X_train_paths = []
    for i,class_folder in enumerate(class_path_list[:]):
        class_folder += '/'
        img_filename_list = os.listdir(flickr32plus_folder_path + class_folder) # 01.jpg, 02.jpg ...
        for j,img_filename in enumerate(img_filename_list[:]):
            img_path = flickr32plus_folder_path + class_folder + img_filename
            X_train_paths.append(img_path)
    return X_train_paths

def get_flickr32plus_number2GT( flickr32plus_folder_path, groundtruth_mat_path='./groundtruth.mat' ):
    ''' Get groundtruth patches from .mat file
    Returns:
    GT - dictionary img_number -> groundtruth[]
    '''
    file = scipy.io.loadmat(groundtruth_mat_path)
    images_array = file['groundtruth'][0]
    GT = dict()
    for i,img_info in enumerate(images_array[:]):
        img_path = img_info[0][0].replace('\\', '/')
        img_path = flickr32plus_folder_path + img_path
        img_number = img_info[0][0].split('\\')[1][:-4] # crops 'class\' and '.jpg'
        groundtruths = img_info[1]
        GT[str(img_number)] = groundtruths.tolist()
    return GT

def get_flickr32plus_number2label( flickr32plus_folder_path ):
    '''
    Returns:
    flickr32plus - dict (img_number -> label)
    '''
    class_path_list = os.listdir(flickr32plus_folder_path)
    flickr32plus = dict()
    for i,class_folder in enumerate(class_path_list[:]):
        class_folder += '/'
        img_filename_list = os.listdir(flickr32plus_folder_path + class_folder)
        for j,img_filename in enumerate(img_filename_list[:]):
            img_number = img_filename[:-4]
            flickr32plus[str(img_number)] = class_folder[:-1]
    return flickr32plus

'''
Flickr32
'''

def get_test_paths_flickr32( flickr32_folder_path ):
    ''' Iterate through testset.relpaths.txt file and get paths of test set
    Returns:
    X_test_paths[] - array of test image paths
    '''
    X_test_paths = []
    with open(flickr32_folder_path+'/testset.relpaths.txt') as file:
        for line in file:
            X_test_paths.append(line[:-1])
    return X_test_paths

def get_valid_paths_flickr32( flickr32_folder_path ):
    ''' Iterate through valset.relpaths.txt file and get paths of test set
    Returns:
    X_valid_paths[] - array of validation image paths
    '''
    X_valid_paths = []
    with open(flickr32_folder_path+'/valset.relpaths.txt') as file:
        for line in file:
            X_valid_paths.append(line[:-1])
    return X_valid_paths

def get_train_paths_flickr32( flickr32_folder_path ):
    ''' Iterate through trainvalset.relpaths.txt file and get paths of test set
    Returns:
    X_train_paths[] - array of train image paths
    '''
    X_train_paths = []
    with open(flickr32_folder_path+'/trainvalset.relpaths.txt') as file:
        for line in file:
            X_train_paths.append(line[:-1])
    return X_train_paths

def get_flickr32_path2number( flickr32_folder_path ):
    '''
    '''
    path2number = dict()
    with open(flickr32_folder_path+'/all.relpaths.txt') as file:
        for img_path in file:
            img_path = img_path[:-1]
            img_number = img_path.split('/')[3][:-5]
            path2number[img_path] = img_number
    return path2number

def get_flickr32_number2label( flickr32_folder_path ):
    '''
    '''
    number2label = dict()
    with open(flickr32_folder_path+'/all.relpaths.txt') as file:
        for img_path in file:
            img_number = img_path.split('/')[3][:-5]
            img_label = img_path.split('/')[2]
            number2label[img_number] = img_label
    return number2label

def get_labels2numbers():
	'''
	returns labels2numbers, numbers2labels dictionaries
	'''
	labels = ['adidas', 'aldi', 'apple', 'becks', 'bmw', 'carlsberg', 'chimay', 'cocacola', 'corona', 'dhl', 'erdinger', 'esso', 'fedex', 'ferrari', 'ford', 'fosters', 'google', 'guiness', 'heineken', 'HP', 'milka', 'nvidia', 'paulaner', 'pepsi', 'rittersport', 'shell', 'singha', 'starbucks', 'stellaartois', 'texaco', 'tsingtao', 'ups', 'no-logo']
	
	l2n = dict()
	n2l = dict()
	for i, label in enumerate(labels):
		l2n[label] = i
		n2l[i] = label
	
	return l2n, n2l
	
	
	