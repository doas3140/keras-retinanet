# Example:
# python train.py -n test -c ./csvpaths/classes.csv -t ./csvpaths/retina-train.csv -v ./csvpaths/retina-valid.csv


import argparse
import os
import sys
import json
import getopt

# Allow relative imports when being executed as script.
#if __name__ == "__main__" and __package__ is None:
#	print('lmao')
#    __package__ = "keras-retinanet.bin"
#    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

#sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

#__package__ = "keras_retinanet.bin"

from keras_retinanet import losses
from keras_retinanet import layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import ResNet50RetinaNet, ResNet101RetinaNet, ResNet152RetinaNet
from keras_retinanet.utils.keras_version import check_keras_version

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf

from eval.LogoEval import LogoEval
from eval.RedirectModel import RedirectModel


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(num_classes, p):
    # create "base" model (no NMS)
	image = keras.layers.Input((None, None, 3))

	if p['resnet'] == 101:
		model = ResNet101RetinaNet(image, num_classes=num_classes, weights=p['weights'], nms=False)
	elif p['resnet'] == 152:
		model = ResNet152RetinaNet(image, num_classes=num_classes, weights=p['weights'], nms=False)
	else: # 50
		model = ResNet50RetinaNet(image, num_classes=num_classes, weights=p['weights'], nms=False)

	training_model = model

    # append NMS for prediction only
	classification = model.outputs[1]
	detections = model.outputs[2]
	boxes = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
	detections = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])

	prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

	# compile model
	training_model.compile(
		loss={
			'regression'    : losses.smooth_l1(),
			'classification': losses.focal()
		},
		optimizer=keras.optimizers.adam(lr = p['learning-rate'], clipnorm=0.001),
		metrics = []
	)

	return model, training_model, prediction_model

def asd():
	pass

def create_callbacks(model, training_model, prediction_model, train_generator, validation_generator, p):
	callbacks = []

	# save the prediction model
	if p['snapshots']:
		checkpoint = keras.callbacks.ModelCheckpoint(
			os.path.join(
				p['snapshot-path'],
				'resnet'+str(p['resnet'])+'_{epoch:02d}.h5'
			),
			verbose=1
		)
		checkpoint = RedirectModel(checkpoint, prediction_model)
		callbacks.append(checkpoint)

	# from ..callbacks.coco import CocoEval
	# evaluation = CocoEval(validation_generator)
	# callbacks.append(evaluation)

	if p['lr-plateau'] == True:
		lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
		callbacks.append(lr_scheduler)

	if p['batch-log'] == True:
		global batch_log_file
		batch_log_file = open(p['batch-log-name'],'a')
		
		logging_callback = keras.callbacks.LambdaCallback(
			on_batch_end = lambda batch, logs: batch_log_file.write('{},{}\n'.format(batch,logs['loss']))
		)
		callbacks.append(logging_callback)
		
	if p['loss'] == True:
		global valid_loss_file
		global train_loss_file
		valid_loss_file = open(p['loss-folder-path']+'valid-loss.csv','a')
		train_loss_file = open(p['loss-folder-path']+'train-loss.csv','a')
		
		valid_loss_callback = keras.callbacks.LambdaCallback(
			on_epoch_end = lambda epoch, logs: valid_loss_file.write('{},{}\n'.format(epoch,logs['val_loss']))
		)
		train_loss_callback = keras.callbacks.LambdaCallback(
			on_epoch_end = lambda epoch, logs: train_loss_file.write('{},{}\n'.format(epoch,logs['loss']))
		)
		callbacks.append(valid_loss_callback)
		callbacks.append(train_loss_callback)
		
	if p['valid-acc'] == True:
		evaluation = LogoEval(validation_generator,filepath=p['valid-acc-path'],threshold=p['threshold'],iou_threshold=p['iou-threshold'])
		evaluation = RedirectModel(evaluation, prediction_model)
		callbacks.append(evaluation)
		
	if p['train-acc'] == True:
		evaluation = LogoEval(train_generator,filepath=p['train-acc-path'],threshold=p['threshold'],iou_threshold=p['iou-threshold'])
		evaluation = RedirectModel(evaluation, prediction_model)
		callbacks.append(evaluation)
		

	return callbacks



def create_generators(p):
	# create image data generator objects
	train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
		horizontal_flip=True,
	)
	val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

	train_generator = CSVGenerator(
		p['annotations'],
		p['classes'],
		train_image_data_generator,
		batch_size = p['batch-size']
	)

	if p['val-annotations']:
		validation_generator = CSVGenerator(
			p['val-annotations'],
			p['classes'],
			val_image_data_generator,
			batch_size = p['batch-size']
		)
	else:
		validation_generator = None

	return train_generator, validation_generator
	

def parse(argv):
	args_ = dict()
	
	try:
		opts, args = getopt.getopt(argv,'n:c:t:v:',['-n=','-c=','-t=','-v='])
	except getopt.GetoptError:
		print('see readme for instructions') 
		sys.exit(2)

	for opt,arg in opts:
		if opt == '-h':
			print('args: read instructions...')
			sys.exit()
		elif opt in ('-n'):
			args_['name'] = arg
		elif opt in ('-c'):
			args_['classes-path'] = arg
		elif opt in ('-t'):
			args_['train-path'] = arg
		elif opt in ('-v'):
			args_['valid-path'] = arg

	return args_

def params(argv):
	args = parse(argv)
	
	p = dict()
	
	folder = './snapshots/'+args['name']+'/'
	p['folder'] = folder
	
	# Hyperparams
	p['epochs'] = 50
	p['batch-size'] = 1 # Batch size must be equal to or higher than the number of GPUs
	p['train-samples'] = 8150 # total=8150 # = steps (cuz batch=1)
	p['valid-samples'] = 3960 # total=3960
	p['learning-rate'] = 1e-5
	p['resnet'] = 50 # 50,101,152
	p['lr-plateau'] = True
	p['threshold'] = 0.5
	p['iou-threshold'] = 0.5
	
	# Params
	p['acc-folder'] = folder+'acc/'
	
	p['valid-acc'] = True
	p['valid-acc-path'] = p['acc-folder']+'valid_acc.csv'
	
	p['train-acc'] = True
	p['train-acc-path'] = p['acc-folder']+'train_acc.csv'
	
	p['loss'] = True
	p['loss-folder-path'] = folder+'loss/'
	
	p['weights'] = 'imagenet' # default should be 'imagenet', can be path to weights .h5 file
	
	p['batch-log'] = True
	p['batch-log-name'] = folder+'loss_log.csv'
	
	p['snapshots'] = True
	p['snapshot-path'] = folder+'weights'
	
	p['classes'] = args['classes-path']
	p['annotations'] = args['train-path']
	p['val-annotations'] = args['valid-path']
	
	p['gpu'] = ''
	p['multi-gpu'] = 0
	
	return p
	
def createfolders(p):
	if not os.path.exists(p['folder']):
		os.makedirs(p['folder'])
	
	if not os.path.exists(p['acc-folder']):
		os.makedirs(p['acc-folder'])
	
	if not os.path.exists(p['snapshot-path']):
		os.makedirs(p['snapshot-path'])
	
	if not os.path.exists(p['loss-folder-path']):
		os.makedirs(p['loss-folder-path'])
	
def main(argv):

	p = params(argv)

	# make sure keras is the minimum required version
	check_keras_version()
	
	createfolders(p)

	# optionally choose specific GPU
	if p['gpu']:
		os.environ['CUDA_VISIBLE_DEVICES'] = p['gpu']
	keras.backend.tensorflow_backend.set_session(get_session())

	# create the generators
	train_generator, validation_generator = create_generators(p)

	# create the model
	print('Creating model, this may take a second...')
	model, training_model, prediction_model = create_models(num_classes = train_generator.num_classes(), p=p)

	# print model summary
	print(model.summary())

	# create the callbacks
	callbacks = create_callbacks(
		model,
		training_model,
		prediction_model,
		train_generator,
		validation_generator,
		p,
	)

	training_model.fit_generator(
		generator = train_generator,
		steps_per_epoch = p['train-samples'] // p['batch-size'],
		validation_data = validation_generator,
		validation_steps = p['valid-samples'] // p['batch-size'],
		epochs = p['epochs'],
		verbose = 1,
		#initial_epoch=9,
		callbacks = callbacks,
	)

if __name__ == '__main__':
	main(sys.argv[1:])