import keras
from .eval_logo import evaluate_logo

class LogoEval(keras.callbacks.Callback):

	def __init__(self, generator, filepath, threshold=0.5, iou_threshold=0.1):
		self.generator = generator
		self.filepath = filepath
		self.threshold = threshold
		self.iou_threshold = iou_threshold
		super(LogoEval, self).__init__()
		
	def on_epoch_end(self, epoch, logs={}):
		if epoch%5 == 0 and epoch != 0:
			acc,g,b,n,bg = evaluate_logo(self.generator, self.model, self.filepath, self.threshold, self.iou_threshold)
			with open(self.filepath,'a') as f:
				line = str(epoch)+','+str(acc)+','+str(g)+','+str(b)+','+str(n)+','+str(bg)+'\n'
				f.write(line)