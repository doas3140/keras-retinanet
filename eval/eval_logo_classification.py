
import numpy as np
from .iou import *

def evaluate_logo(generator, model, name, threshold=0.05, iou_t=0.1):
	
	with open(name,'w') as f:
		results = []
		image_ids = []
		good_labeled,bad_labeled,not_labeled,bg_labeled = 0,0,0,0
		print()
		for i in range(len(generator.image_names)):

			image = generator.load_image(i)
			image = generator.preprocess_image(image)
			image, scale = generator.resize_image(image)


			_, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
			
			detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
			detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
			detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
			detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])
			

			detections[:, :4] /= scale
			
			detections[:, :, 2] -= detections[:, :, 0]
			detections[:, :, 3] -= detections[:, :, 1]
			
			highest_score = 0
			for detection in detections[0, ...]:
				positive_labels = np.where(detection[4:] > threshold)[0]


				for label in positive_labels:
					image_result = {
						'image_id'    : i,
						'category_id' : generator.label_to_name(label),
						'score'       : float(detection[4 + label]),
						'bbox'        : (detection[:4]).tolist(),
					}
					if image_result['score'] > highest_score:
						highest_score = image_result['score']
						hs_label = generator.label_to_name(label)
			if highest_score == 0:
				hs_label = 'no-logo'
				highest_score = 1.0

			img_nr = generator.image_path(i).split('/')[6][:-4]
			line = img_nr + '\t' + hs_label + '\t' + str(highest_score) + '\n'
			f.write(line)
			print('Evaluating: {}/{}'.format(i, len(generator.image_names)), end='\r')