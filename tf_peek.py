import tensorflow as tf

path = "/Users/seifeddinereguige/Documents/tfds_Dataset/linear_movement_rotate_bar/1.0.0/flow_dataset-train.tfrecord-00000-of-00001"

raw_ds = tf.data.TFRecordDataset([path])

for raw in raw_ds.take(1):
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())
    print("num features:", len(ex.features.feature))
    print("first 25 keys:", list(ex.features.feature.keys())[:25])