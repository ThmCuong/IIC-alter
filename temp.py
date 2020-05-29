import tensorflow as tf 
from data import load
print(tf.__version__)
# pick a data set
DATA_SET = 'mnist'

# define splits
DS_CONFIG = {
    # mnist data set parameters
    'mnist': {
        'batch_size': 5,
        'num_repeats': 5,
        'mdl_input_dims': [24, 24, 1]}
}

    # load the data set
TRAIN_SET, TEST_SET, SET_INFO = load(data_set_name=DATA_SET, **DS_CONFIG[DATA_SET])
# x = 3
# def adds(x):
#     x +=1
#     return x 

# print(adds(x))
# print("x = ",x)

# x = tf.constant([[1, 4], [1.5, 1.4]])
# y = tf.constant([[2, 5], [5.4, 3.2]]) 
# z = tf.constant([[3, 6], [2.3, 9.8]])
# print("shape x : ",tf.shape(x))
# t = tf.stack([x, y, z])
# print("shape t : ",tf.shape(t))
# print("t = ",t)
# q = tf.transpose(t,[1,0,2])
# print("q = ",q)
# print("Shape q: ",tf.shape(q))
# print("use stack x, y, z : ",tf.stack([x, y, z], axis= 1))

# j = tf.random.categorical([[1., 1.]], tf.shape(q)[0])
# print("j = ",j)
# i = tf.squeeze(j)
# print("after squeeze: ",i)

# lm = tf.map_fn(lambda y: y[1], (q, i), dtype= tf.int64)
# print("after use map_fn: ",lm)

# # print("q0 = ", q[0][0])
# print("---------------")
# # for elm in (q, i):
# #     print("elm q, i: ", elm)