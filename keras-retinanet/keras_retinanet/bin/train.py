#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import json

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    __package__ = "keras_retinanet.bin"
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import losses
from .. import layers
from ..callbacks import RedirectModel
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..models.resnet import ResNet50RetinaNet, ResNet101RetinaNet, ResNet152RetinaNet
from ..utils.keras_version import check_keras_version


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(num_classes, p):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, 3))
	
	if p['resnet'] == 50:
		model = ResNet50RetinaNet(image, num_classes=num_classes, weights=p['weights'], nms=False)
	elif p['resnet'] == 101:
		model = ResNet101RetinaNet(image, num_classes=num_classes, weights=p['weights'], nms=False)
	elif p['resnet'] == 152:
		model = ResNet152RetinaNet(image, num_classes=num_classes, weights=p['weights'], nms=False)
	
	training_model = model

    # append NMS for prediction only
    classification   = model.outputs[1]
    detections       = model.outputs[2]
    boxes            = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
    detections       = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
    
	prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr = p['learning-rate'], clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, p):
    callbacks = []

    # save the prediction model
    if p['snapshots']:
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                p['snapshot-path'],
                'resnet50_{{epoch:02d}}.h5'
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
	
	if p['log'] == True:
		logger = keras.callbacks.CSVLogger( p['log-name'] )
		callbacks.append(logger)

	if p['json-log'] == True:
		json_log = open(p['json-name'], mode='wt', buffering=1)
		json_logging_callback = LambdaCallback(
			on_epoch_end=lambda epoch, logs: json_log.write(
				json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
			on_train_end=lambda logs: json_log.close()
		)
		callbacks.append(json_logging_callback)
		
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

def params():
	# Hyperparams
	p['epochs'] = 50
	p['batch-size'] = 1 # Batch size must be equal to or higher than the number of GPUs
	p['train-samples'] = 8150 # total=8150 # = steps (cuz batch=1)
	p['valid-samples'] = 3960 # total=3960
	p['learning-rate'] = 1e-5
	p['resnet'] = 50 # 50,101,152
	p['lr-plateau'] = True
	p[''] = 
	
	# Params
	
	p['measure-valid-acc'] = False
	
	p['weights'] = 'imagenet'
	p['gpu'] = True #?
	p['multi-gpu'] = 0
	
	p['log'] = True
	p['log-name'] = 'training.log'
	
	p['json-log'] = True
	p['json-name'] = 'loss_log.json'
	
	p['snapshots'] = True
	p['snapshot-path'] = './snapshots'
	
	p['classes'] = 
	p['annotations'] = 
	p['val-annotations'] = 
	
	return p
	
def main():
 
    p = params()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if p['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = p['gpu']
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(p)

    # create the model
    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(num_classes = train_generator.num_classes(), p, multi_gpu = p['multi-gpu'])

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        p,
    )

    # start training
	if p['measure-valid-acc'] == True:
		training_model.fit_generator(
			generator = train_generator,
			steps_per_epoch = p['train-samples'] // p['batch-size'],
			epochs = p['epochs'],
			validation_data = validation_generator,
			validation_steps = p['valid-samples'] // p['batch-size']
			verbose = 1,
			callbacks = callbacks,
		)
	else:
		training_model.fit_generator(
			generator = train_generator,
			steps_per_epoch = p['train-samples'] // p['batch-size'],
			epochs = p['epochs'],
			verbose = 1,
			callbacks = callbacks,
		)
    

if __name__ == '__main__':
    main()
