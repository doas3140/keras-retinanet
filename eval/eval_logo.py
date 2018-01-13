
import numpy as np
from .iou import *

def evaluate_logo(generator, model, filepath, threshold=0.05, iou_t=0.1):
	
	results = []
	image_ids = []
	good_labeled,bad_labeled,not_labeled,bg_labeled = 0,0,0,0
	print()
	for i in range(len(generator.image_names)):
		image = generator.load_image(i)
		image = generator.preprocess_image(image)
		image, scale = generator.resize_image(image)

		annotations = generator.load_annotations(i)


		_, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

		detections[:, :4] /= scale
		
		predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
		scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
		

		no_detections = (scores>threshold).sum()
		
		path = generator.image_path(i)
		
		if len(annotations) == 0:
			bg_labeled += no_detections
		else:
			used_d = 0
			for a in annotations:
				gt = a[:4].astype(int)
				
				breaked = False
				for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
					if score > threshold:
						bbox = detections[0, idx, :4].astype(int)
						pred_label = generator.label_to_name(label)
						true_label = generator.image_data[path][0]['class']
						
						iou = IoU( xyxy2xywh(gt), xyxy2xywh(bbox) )
						if iou > iou_t:
							if pred_label == true_label:
								good_labeled += 1
								used_d += 1
								breaked = True
								break
							else:
								bad_labeled += 1
								used_d += 1
								breaked = True
								break
				if breaked == False:
					not_labeled += 1
			bg_labeled += no_detections - used_d
		
		
		acc = good_labeled / (good_labeled + bad_labeled + not_labeled + bg_labeled)


		print('Evaluating: {}/{} acc={:0.2f} G={} B={} N={} BG={}'.format(i, len(generator.image_names), acc, good_labeled,bad_labeled,not_labeled,bg_labeled), end='\r')

	
	acc = good_labeled / (good_labeled + bad_labeled + not_labeled + bg_labeled)
	
	return acc, good_labeled, bad_labeled, not_labeled, bg_labeled