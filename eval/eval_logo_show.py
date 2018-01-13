
import numpy as np
from .iou import *
import cv2
import matplotlib.pyplot as plt


def showlogo(test_generator,prediction_model,i):
	index = i
	image = test_generator.load_image(i)

	draw = image.copy()
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

	image = test_generator.preprocess_image(image)
	image, scale = test_generator.resize_image(image)
	annotations = test_generator.load_annotations(index)
	
	_, _, detections = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))

	predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
	scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

	detections[:, :4] /= scale
	
	path = test_generator.image_path(index)
	
	try:
		label = test_generator.image_data[path][0]['class']
	except:
		pass
	

	for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
		if score < 0.5:
			continue
		b = detections[0, idx, :4].astype(int)

		cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
		caption = "{} {:.3f}".format(test_generator.label_to_name(label), score)
		cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
		cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)



	for annotation in annotations:
		label = int(annotation[4])
		b = annotation[:4].astype(int)
		cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
		caption = "{}".format(test_generator.label_to_name(label))
		cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
		cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

	plt.figure(figsize=(15, 15))
	plt.axis('off')
	plt.imshow(draw)
	plt.show()

def evaluate_logo(asd,generator, model, filepath, threshold=0.05, iou_t=0.1):
	
	results = []
	image_ids = []
	good_labeled,bad_labeled,not_labeled,bg_labeled = 0,0,0,0
	print()

	for i in asd:
		image_o = generator.load_image(i)
		image = generator.preprocess_image(image_o)
		image, scale = generator.resize_image(image)

		annotations = generator.load_annotations(i)


		_, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
		
		predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
		scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
		

		detections[:, :4] /= scale
		

		no_detections = (scores>threshold).sum()
		
		path = generator.image_path(i)
		
		if len(annotations) == 0:
			bg_labeled += no_detections
			if no_detections > 0:
				#showlogo(generator,model,i)
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
								#showlogo(generator,model,i)
								used_d += 1
								breaked = True
								break
							else:
								bad_labeled += 1
								#showlogo(generator,model,i)
								used_d += 1
								breaked = True
								break
				if breaked == False:
					not_labeled += 1
					#showlogo(generator,model,i)
			bg_labeled += no_detections - used_d
		
		
		acc = good_labeled / (good_labeled + bad_labeled + not_labeled + bg_labeled)


		print('Evaluating: {}/{} acc={:0.2f} G={} B={} N={} BG={}'.format(i, len(generator.image_names), acc, good_labeled,bad_labeled,not_labeled,bg_labeled), end='\r')

	
	acc = good_labeled / (good_labeled + bad_labeled + not_labeled + bg_labeled)
	
	return acc, good_labeled, bad_labeled, not_labeled, bg_labeled