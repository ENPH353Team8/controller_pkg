#!/usr/bin/env python3
import os

import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/src/node/my_model.h5')
new_model.summary()