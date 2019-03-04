# FAQ

### How can I use KittiSeg on my own data?

Have a look at [inputs.md](inputs.md) and this [issue](https://github.com/MarvinTeichmann/KittiSeg/issues/8). Feel free to open a further issue or comment on [issue 8](https://github.com/MarvinTeichmann/KittiSeg/issues/8) if your question is not covered so far. 

Also, once you figured out how to make it work, feel free to add some lines of explanation to [inputs.md](inputs.md). As owner of this code, it is not easy get an idea of which conceptual aspects needs more explanation. 

### I would like to train on greyscale images, is this possible?

Yes, since commit f7fdb24, all images will be converted to RGB upon load. Greyscale is those supported out-of-the box. This means, that even for greyscale data each pixel will be represented by a tripel. This is important when specifying the input format in [your hype file](../hypes/KittiSeg.json). Black will be stored as [0,0,0], white [255,255,255] and some light grey can be [200, 200, 200].

### Can I use KittiSeg for multi-class segmentation?

Yes, I had an earlier version run on Cityscapes data. Unfortunatly, my Cityscapes code is not compatible with the current TensorFlow and TensorVision version anymore and I did not find the time to port it, yet.

However making this run is not to much of an afford. You will need to adapt `_make_data_gen` in the [input_producer](../inputs/kitti_seg_input.py) to produce an `gt_image` tensor with more then two channels. In addition, you will need to write new evaluation code. The current [evaluator file](../evals/kitti_evals.py) computes kitti scores which are only defined on binary segmentation problems. 

Feel free to open a pull request if you find the time to implement those changes. I am also happy to help with any issues you might encounter.

### How can I make a model trained on Kitti data perform better on non-kitti street images? ([Issue #14](https://github.com/MarvinTeichmann/KittiSeg/issues/14))

Turn data augmentation on. The current version has all data augmentation turned of on default to perform well on the benchmark. This makes the trained model very sensitive to various aspects including lighting conditions and sharpness. Distortions, like random brightness, random resizing (including the change of aspect ratio) and even fancier thinks will force the ignore camera depended hints. Many common distortions are already in the [input-producer](https://github.com/MarvinTeichmann/KittiSeg/blob/master/inputs/kitti_seg_input.py), but turned of on default. 

Alternative, consider training on your data (if possible) or apply fine-tuning using view labeled images of your data.

### How can I test a picture of myself with a trained model (not in the kitti database)？
I want to test a picture that doesn't exist in kitti with the trained model, and see the results.The command used is[ python demo.py --input_image data/demo/mypicture. PNG] , but I encounter such problem [Traceback (most recent call last):
  File "demo.py", line 246, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 48, in run
    _sys.exit(main(_sys.argv[:1] + flags_passthrough))
  File "demo.py", line 194, in main
    output = sess.run([softmax], feed_dict=feed)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 1336, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Number of ways to split should evenly divide the split dimension, but got split_dim 3 (size = 4) and num_split 3
	 [[Node: Validation/Processing/split = Split[T=DT_FLOAT, num_split=3, _device="/job:localhost/replica:0/task:0/device:GPU:0"](Validation/Processing/split/split_dim, ExpandDims)]]
	 [[Node: Validation/decoder/Softmax/_81 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_266_Validation/decoder/Softmax", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

Caused by op u'Validation/Processing/split', defined at:
  File "demo.py", line 246, in <module>
    tf.app.run()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 48, in run
    _sys.exit(main(_sys.argv[:1] + flags_passthrough))
  File "demo.py", line 163, in main
    image=image)
  File "incl/tensorvision/core.py", line 137, in build_inference_graph
    logits = modules['arch'].inference(hypes, image, train=False)
  File "RUNS/KittiSeg_pretrained/model_files/architecture.py", line 27, in inference
    vgg_fcn.build(images, train=train, num_classes=2, random_init_fc8=True)
  File "/home/zsh/MT/KittiSeg/incl/tensorflow_fcn/fcn8_vgg.py", line 59, in build
    red, green, blue = tf.split(rgb, 3, 3)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/array_ops.py", line 1265, in split
    split_dim=axis, num_split=num_or_size_splits, value=value, name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_array_ops.py", line 5094, in _split
    name=name)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 2956, in create_op
    op_def=op_def)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/ops.py", line 1470, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InvalidArgumentError (see above for traceback): Number of ways to split should evenly divide the split dimension, but got split_dim 3 (size = 4) and num_split 3
	 [[Node: Validation/Processing/split = Split[T=DT_FLOAT, num_split=3, _device="/job:localhost/replica:0/task:0/device:GPU:0"](Validation/Processing/split/split_dim, ExpandDims)]]
	 [[Node: Validation/decoder/Softmax/_81 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_266_Validation/decoder/Softmax", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
], I don't know how to deal with it




