import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# tf.enable_eager_execution()

def mnist_x(x_orig, mdl_input_dims, is_training):
    '''
        x_orig: image; , [28,28,1]
        mdl_input_dims: size of wanted input ; (24, 24, 1)
        is_training
    '''
    # rescale to [0, 1]
    x_orig = tf.cast(x_orig, dtype=tf.float32) / x_orig.dtype.max

    tmp = tf.concat((tf.shape(x_orig)[:1], [20, 20], [1]), axis=0)
    print(" x_orig shape {} ---{}".format(x_orig.shape.as_list(), tmp))
    print(" shape -- xorg ", tf.shape(x_orig)[1])
    # get common shapes ; mdl_input_dims = (w, h, channel)

    height_width = mdl_input_dims[:-1] # [24, 24]
    n_chans = mdl_input_dims[-1]

    # training transformations
    if is_training:
        #crop ; 20/24
        x1 = tf.image.central_crop(x_orig, np.mean(20 / np.array(x_orig.shape.as_list()[1:-1])))
        # print(" shape x1 ,", x1.shape)
        # print(" gia tri mean: ", np.mean(20 / np.array(x_orig.shape.as_list()[1:-1])))
        #random crop ; [None, 20, 20, 1]
        x2 = tf.image.random_crop(x_orig, tf.concat((tf.shape(x_orig)[:1], [20, 20], [n_chans]), axis=0))
        # print("Shape x2 : ", x2.shape)
        x = tf.stack([x1, x2])
        print("shape x after stack : ", x.shape)
        x = tf.transpose(x, [1, 0, 2, 3, 4])
        print("Shape x after tranpose: ", x.shape) 

        i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(x)[0]))
        print("i = : ", i)
        print("shpe x[0] ", tf.shape(x)[0])
        # de lam gi?
        x = tf.map_fn(lambda y: y[0][y[1]], (x, i), dtype=tf.float32)
        # print("shape x after map_fun :", x.shape)
        x = tf.image.resize(x, height_width)

    # testing transformations
    else:
        x = tf.image.central_crop(x_orig, np.mean(20 / np.array(x_orig.shape.as_list()[1:-1])))
        x = tf.image.resize(x, height_width)
    # print(x.shape)
    return x


def mnist_gx(x_orig, mdl_input_dims, is_training, sample_repeats):
    # x_orig : 28, 28 , 1
    # if not training, return a constant value--it will unused but needs to be same shape to avoid TensorFlow errors
    if not is_training:
        return tf.zeros([0] + mdl_input_dims)

    # rescale to [0, 1]
    x_orig = tf.cast(x_orig, dtype=tf.float32) / x_orig.dtype.max

    # repeat samples accordingly
    x_orig = tf.tile(x_orig, [sample_repeats] + [1] * len(x_orig.shape.as_list()[1:]))

    # get common shapes
    height_width = mdl_input_dims[:-1] # 24 24
    n_chans = mdl_input_dims[-1] # 1

    # random rotation
    rad = 2 * np.pi * 25 / 360 # 25 degree
    x_rot = tf.contrib.image.rotate(x_orig, tf.random.uniform(shape=tf.shape(x_orig)[:1], minval=-rad, maxval=rad))
    # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    gx = tf.stack([x_orig, x_rot]) # axis = 1 is ok ; [2, None, 28, 28, 1] ; None means free to select the length
    gx = tf.transpose(gx, [1, 0, 2, 3, 4]) #[None, 2, 28, 28, 1]
    # create a tf.tensor size tf.shape(gx)[0] -- number row of gx with [0, 1]
    i = tf.squeeze(tf.random.categorical([[1., 1.]], tf.shape(gx)[0]))
    # lay tat ca dong 0 hoac dong 1
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)
    # lay moi noi mot cai

    # random crops
    x1 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [16, 16], [n_chans]), axis=0))
    x2 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [20, 20], [n_chans]), axis=0))
    x3 = tf.image.random_crop(gx, tf.concat((tf.shape(x_orig)[:1], [24, 24], [n_chans]), axis=0))
    gx = tf.stack([tf.image.resize(x1, height_width),
                   tf.image.resize(x2, height_width),
                   tf.image.resize(x3, height_width)])
    gx = tf.transpose(gx, [1, 0, 2, 3, 4])
    i = tf.squeeze(tf.random.categorical([[1., 1., 1.]], tf.shape(gx)[0]))
    gx = tf.map_fn(lambda y: y[0][y[1]], (gx, i), dtype=tf.float32)

    # apply random adjustments
    def rand_adjust(img):
        img = tf.image.random_brightness(img, 0.4)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        if img.shape.as_list()[-1] == 3:
            img = tf.image.random_saturation(img, 0.6, 1.4)
            img = tf.image.random_hue(img, 0.125)
        return img

    gx = tf.map_fn(lambda y: rand_adjust(y), gx, dtype=tf.float32)

    return gx


def pre_process_data(ds, info, is_training, **kwargs):
    """
    :param ds: TensorFlow Dataset object
    :param info: TensorFlow DatasetInfo object
    :param is_training: indicator to pre-processing function
    :return: the passed in data set with map pre-processing applied
    """
    # apply pre-processing function for given data set and run-time conditions
    if info.name == 'mnist':
        return ds.map(lambda d: {'x': mnist_x(d['image'],
                                              mdl_input_dims=kwargs['mdl_input_dims'],
                                              is_training=is_training),
                                 'gx': mnist_gx(d['image'],
                                                mdl_input_dims=kwargs['mdl_input_dims'],
                                                is_training=is_training,
                                                sample_repeats=kwargs['num_repeats']),
                                 'label': d['label']},
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        raise Exception('Unsupported data set: ' + info.name)


def configure_data_set(ds, info, batch_size, is_training, **kwargs):
    """
    :param ds: TensorFlow data set object
    :param info: TensorFlow DatasetInfo object
    :param batch_size: batch size
    :param is_training: indicator to pre-processing function
    :return: a configured TensorFlow data set object
    """
    # enable shuffling and repeats
        # ds: 10*N, 28, 28
    ds = ds.shuffle(10 * batch_size, reshuffle_each_iteration=True).repeat(1)

    # batch the data before pre-processing
    ds = ds.batch(batch_size)

    # pre-process the data set ; specific the device for ops created/executed in this context
    with tf.device('/cpu:0'):
        ds = pre_process_data(ds, info, is_training, **kwargs)

    # enable prefetch
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load(data_set_name, **kwargs):
    """
    :param data_set_name: data set name--call tfds.list_builders() for options
    :return:
        train_ds: TensorFlow Dataset object for the training data
        test_ds: TensorFlow Dataset object for the testing data
        info: data set info object
    """
    # get data and its info
    ds, info = tfds.load(name=data_set_name, split='train + test', with_info=True)
        # ds: N , 28, 28, 1 ; shapes: {image: (28, 28, 1), label: ()}, types: {image: tf.uint8, label: tf.int64}>
        # tf.data.Dataset
    # configure the data sets
    if 'train' in info.splits:
        train_ds = configure_data_set(ds=ds, info=info, is_training=True, **kwargs)
    else:
        train_ds = None
    if 'test' in info.splits:
        test_ds = configure_data_set(ds=ds, info=info, is_training=False, **kwargs)
    else:
        test_ds = None

    return train_ds, test_ds, info

