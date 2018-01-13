# Example:
# python evaluate.py -w weights.h5 -c ./csvpaths/classes.csv -t ./csvpaths/retina-test.csv -o ./evalkit/classification.txt


import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from train import *

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import getopt

import numpy as np
import os
import scipy.io
from scripts.data_manipulation import *
import sys

import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    return tf.Session(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.backend.tensorflow_backend.set_session(get_session())
	
def parse(argv):
	p = dict()

	try:
		opts, args = getopt.getopt(argv,'w:c:t:o:',['-w=','-c=','-t=','-o='])
	except getopt.GetoptError:
		print('see readme for instructions') 
		sys.exit(2)

	for opt,arg in opts:
		if opt == '-h':
			print('args: -w = path to weights, -t = test csv path, -c = classes csv path, -o = output_path')
			sys.exit()
		elif opt in ('-w'):
			p['weights_path'] = arg
		elif opt in ('-c'):
			p['classescsv_path'] = arg
		elif opt in ('-t'):
			p['testcsv_path'] = arg
		elif opt in ('-o'):
			p['output_path'] = arg
	
	return p
	
def params(argv):
	args = parse(argv)
	p = dict()
	p['output_csv'] = args['output_path']
	p['test_csv'] = args['testcsv_path']
	p['classes_csv'] = args['classescsv_path'] 
	p['weights'] = args['weights_path']
	p['resnet'] = 50
	p['learning-rate'] = 1e-5
	p['valid-samples'] = 3960
	p['train-samples'] = 8150
	
	return p
	
def create_models_local(p):
	test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()
	test_generator = CSVGenerator(
		p['test_csv'],
		p['classes_csv'],
		test_image_data_generator,
		1
	)
	model, training_model, prediction_model = create_models(num_classes=test_generator.num_classes(), p=p)
	
	return model, training_model, prediction_model, test_generator

def main(argv):
	
	p = params(argv)
	
	model, training_model, prediction_model, test_generator = create_models_local(p)
	
	# this part can be changed, see eval folder for other options
	from eval.eval_logo_classification import evaluate_logo
	evaluate_logo(test_generator, model, p['output_csv'], threshold=0.53, iou_t=0.5)
	
	
	
if __name__ == '__main__':
    main(sys.argv[1:])