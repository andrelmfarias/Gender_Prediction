import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import DataSets as ds
import Layers
from PIL import Image
import sys


image_path = sys.argv[1]

image = Image.open(image_path).convert('L')
image = image.resize((48, 48), Image.ANTIALIAS)
plt.imshow(image, cmap='gray')
image = np.reshape(image, (1, 48*48))


with tf.Session() as sess:
    # Restore variables from Model
    saver = tf.train.import_meta_graph('./ckpt/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./ckpt/'))
    #saver.restore(sess, "./ckpt/model.ckpt")
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("input/x:0")
    y = graph.get_tensor_by_name("CNN/fc_2/LogSoftmax:0")
    ITM = graph.get_tensor_by_name("input/Is_Training_Mode:0")

    feed_dict = {x: image, ITM: False}
    pred = sess.run(y, feed_dict)
    print(pred[0])
    label = np.argmax(pred[0])
    if label == 0:
        print('Label = MEN')
    else:
        print('Label = WOMAN')

# plt.show()
