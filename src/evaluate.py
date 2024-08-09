import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('../results/cnn_model.h5')

# Load the test data
_, (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images = test_images / 255.0

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

