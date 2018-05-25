from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
import COCO

import random
import math
import numpy as np
import cv2

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(encoded_jpg, label):

    width, height = (225, 225)
    image_format = b'jpg'

    feature={'image/encoded': _bytes_feature(encoded_jpg),
             'image/format': _bytes_feature(image_format),
             'image/label': _int64_feature(label)}

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


def main(_):
    ###Create Train tfrecord###
    trainDataset = COCO.dataset('train')
    num_objects = 5000
    path_to_write = os.path.join(os.getcwd()) + '/images/train.record'
    writer = tf.python_io.TFRecordWriter(path_to_write)
    
    train_data , train_labels = ([1], [1]) #start loop
    while len(train_data) > 0:
        train_data, train_labels = trainDataset.nextImages(num_objects)
        
        for x in range(0, len(train_data)):
            tf_example = create_tf_example(train_data[x], train_labels[x])
            writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the train TFRecord: {}'.format(path_to_write))
    del trainDataset #free up memory

    ###Create Test tfrecord###
    testDataset = COCO.dataset('test')
    num_objects = 5000
    path_to_write = os.path.join(os.getcwd()) + '/images/test.record'
    writer = tf.python_io.TFRecordWriter(path_to_write)
    
    test_data , test_labels = ([1], [1]) #start loop
    while len(test_data) > 0:
        test_data, test_labels = testDataset.nextImages(num_objects)

        for x in range(0, len(test_data)):
            tf_example = create_tf_example(test_data[x], test_labels[x])
            writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the test TFRecord: {}'.format(path_to_write))


if __name__ == '__main__':
    tf.app.run()
