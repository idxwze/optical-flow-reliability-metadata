from tfrecord.reader import tfrecord_loader

tfrecord_path = "/Users/seifeddinereguige/Documents/tfds_Dataset/linear_movement_rotate_bar/1.0.0/flow_dataset-train.tfrecord-00000-of-00001"
index_path = None  # usually None; only needed if you have a .index file

dataset = tfrecord_loader(tfrecord_path, index_path=index_path)

ex = next(iter(dataset))
print("Keys:", list(ex.keys())[:20])
for k, v in ex.items():
    try:
        print(k, type(v), getattr(v, "shape", None))
    except Exception:
        print(k, type(v))