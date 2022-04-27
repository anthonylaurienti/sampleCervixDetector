import numpy as np
import os
import datetime

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

#Specify our model architecture
spec = model_spec.get('efficientdet_lite0')
#get our data
#train_data, validation_data, test_data = object_detector.DataLoader.from_csv('dataset.csv')
image_dir = 'data/Type_1'
annotations_dir = 'data/train'
val_annotations_dir = 'data/val'
train_data = object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map={1: "cervix"})
val_data = object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map={1: "cervix"})
test_data = object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map={1: "cervix"})

#Train the model
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=val_data, epochs=100) #callbacks=[tensorboard_callback])
#eval our model with test data
f = open("evaldata.txt", "a")
f.write(str(model.evaluate(test_data)))
f.write("\n\n")
f.close()
#export model
model.export(export_dir='.')
#evaluate tflite model
f = open("evaldata.txt", "a")
f.write(str(model.evaluate_tflite('model.tflite', test_data)))
f.close()
