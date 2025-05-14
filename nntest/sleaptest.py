# In this file I would tailing *sleap* to a minimal example for centroid prediction in a forward pass 
# Contain 2 parts: 1. model loading 2. forward pass with model and input data.
# prepare input data: a verified screenshot of previous video.
# prepare model: a centroid model from sleap. (什么格式？)

import numpy as np
import tensorflow as tf
# Import from parent folder
# This is a workaround for this simple test.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sleap_peak_postprocessing import find_local_peaks

# utils
import matplotlib
import matplotlib.pyplot as plt
import time # for profiling
from typing import Tuple


start_time = time.time()
model = tf.keras.models.load_model('nntest/best_model.h5', compile=False)
load_time = time.time() - start_time
print(f'Load time: {load_time:.3f}s')

# Profile the model inference time.
start_time = time.time()
# Read test.bmp and predict
data = matplotlib.image.imread('nntest/test.bmp')
# Pad the data to 1024x1280x3 with black, the image is kept at upper left corner.
data = np.pad(data, ((0, 1024 - data.shape[0]), (0, 1280 - data.shape[1]), (0, 0)), 'constant', constant_values=0)
data = np.expand_dims(data, axis=0)
# Normalize the data to 0-1.
data = data / 255.0
# Resize image to a half resolution.
data = tf.image.resize(data, (512, 640))
preprocessing_simple_time = time.time() - start_time
start_time = time.time()
out = model.predict(data)
predict_time = time.time() - start_time
start_time = time.time()
peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(out)
postprocessing_time = time.time() - start_time
print(f'Preprocessing time: {preprocessing_simple_time:.3f}s, Inference time: {predict_time:.3f}s, Postprocessing time: {postprocessing_time:.3f}s')
# Predict another 10 cycles to check if it has warmed up.
for i in range(10):
    # Add a random noise to the data.
    data_disrupt = data + np.random.normal(0, 0.01, data.shape)
    start_time = time.time()
    out = model.predict(data_disrupt)
    predict_time_again = time.time() - start_time
    print(f'Inference time again: {predict_time_again:.3f}s')

# Post-processing: find local peak
start_time = time.time()
peak_points, peak_vals, peak_sample_inds, peak_channel_inds = find_local_peaks(out)
postprocessing_time = time.time() - start_time
print(f'Postprocessing time: {postprocessing_time:.3f}s')

print(f'Peak points: {peak_points}')
print(f'Peak values: {peak_vals}')
print(f'Peak sample indices: {peak_sample_inds}')

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(out[0])
plt.plot(peak_points[:, 0], peak_points[:, 1], label='Peaks', color='red', marker='x', linestyle='None', markersize=8)
plt.title('Predicted Centroid')
plt.subplot(1, 2, 2)
plt.imshow(data[0])
peak_points = peak_points * 2
# In this example, the input of model is 1/2 of the ori image, 
# and the model will further pooling to 1/2 of the input. Here data[0] is the input image.
plt.plot(peak_points[:, 0], peak_points[:, 1], label='Peaks', color='red', marker='x', linestyle='None', markersize=8)
plt.title('Input Image')
plt.show()

# Test Result (Typical): i9-14900K: 
# Preprocessing 16 ms, First Inference 170 ms, 
# Subsequent Inference 90 ms.
# TODO: test post-processing time.

# 60fps x264 encoding would not affect the inference time.
