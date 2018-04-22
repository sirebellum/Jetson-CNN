import tensorflow as tf

def CNN_Model(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  print("Mode:", mode)
  input_layer = features

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[11, 11],
      strides=(4, 4),
      padding="valid",
      activation=tf.nn.relu)
  
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
  
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[5, 5],
      strides=(1, 1),
      padding="valid",
      activation=tf.nn.relu)
      
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
  
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=384,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="valid",
      activation=tf.nn.relu)
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="valid",
      activation=tf.nn.relu)
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      strides=(1, 1),
      padding="valid",
      activation=tf.nn.relu)
      
  # Dense Layer
  _, height, width, depth = conv5.get_shape()
  print("CNN with final feature maps:", height, "x", width, "x", depth)
  conv5_flat = tf.reshape(conv5, [-1, height * width * depth])
  dense = tf.layers.dense(inputs=conv5_flat, units=2048, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=80)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=80)
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def parse_record(serialized_example): #parse a single binary example
  """Parses a single tf.Example into image and label tensors."""
  features = {'image/encoded': tf.FixedLenFeature([], tf.string),
             'image/format':  tf.FixedLenFeature([], tf.string),
             'image/label':   tf.FixedLenFeature([], tf.int64)}
  features = tf.parse_single_example(serialized_example, features)
  
  #print("JPG:", features['image/encoded'])
  image = tf.image.decode_jpeg(features['image/encoded'], channels=0)
  #print("image:", image)
  image = tf.reshape(image, [225, 225, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  
  label = tf.cast(features['image/label'], tf.int32)
  
  return (image, label)
  
# Define the input function for training
def train_input_fn():

  # Keep list of filenames, so you can input directory of tfrecords easily
  train_filenames = ["COCO/train.record"]
  test_filenames = ["COCO/test.record"]
  batch_size=256

  # Import data
  dataset = tf.data.TFRecordDataset(train_filenames)

  # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(parse_record)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat()
  #print("Dataset:", dataset.output_shapes, ":::", dataset.output_types)
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  #print("Iterator:", features)

  return (features, labels)

# Our application logic will be added here

# Define the input function for evaluating
def eval_input_fn():

  # Keep list of filenames, so you can input directory of tfrecords easily
  train_filenames = ["COCO/train.record"]
  test_filenames = ["COCO/test.record"]
  batch_size = 24

  # Import data
  dataset = tf.data.TFRecordDataset(test_filenames)

  # Map the parser over dataset, and batch results by up to batch_size
  dataset = dataset.map(parse_record)
  dataset = dataset.batch(batch_size)
  #print("Dataset:", dataset.output_shapes, ":::", dataset.output_types)
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  #print("Iterator:", features)

  return (features, labels)
