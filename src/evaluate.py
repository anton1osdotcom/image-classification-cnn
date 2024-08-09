# Your existing imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10  # Import the CIFAR-10 dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
test_images = test_images / 255.0

# Load your model
model = load_model('../results/cnn_model.keras')

# Evaluate your model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')

# Generate the confusion matrix
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_labels)
cmd = ConfusionMatrixDisplay(cm, display_labels=range(10))

cmd.plot()
plt.show()
