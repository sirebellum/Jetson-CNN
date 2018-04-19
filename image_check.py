from cnn_models import parse_record
import tensorflow as tf
import numpy as np
import matplotlib;
import cv2

train_filenames = ["COCO/train.record"]
batch_size=256
dataset = tf.data.TFRecordDataset(train_filenames)
dataset = tf.data.TFRecordDataset(train_filenames)
dataset = dataset.map(parse_record)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
features, labels = iterator.get_next()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  
  images = [0]
  while len(images) > 0:
    images = features.eval()
    labs = labels.eval()
  
    for image in images:
      if not image.any():
        cv2.imshow("bad object", np.asarray(image))
        cv2.waitKey(0) #wait for user input
      else:
        cv2.imshow("object", np.asarray(image))
        cv2.waitKey(5)
