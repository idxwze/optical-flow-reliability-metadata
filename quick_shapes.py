from tfrecord.reader import tfrecord_loader
import numpy as np

path = "/Users/seifeddinereguige/Documents/tfds_Dataset/linear_movement_rotate_bar/1.0.0/flow_dataset-train.tfrecord-00000-of-00001"
ex = next(iter(tfrecord_loader(path, index_path=None)))

print("video:", ex["video"].shape, ex["video"].dtype)
print("forward_flow:", ex["forward_flow"].shape, ex["forward_flow"].dtype)
print("forward_flow_range:", ex["metadata/forward_flow_range"])
print("height,width:", int(ex["metadata/height"][0]), int(ex["metadata/width"][0]))