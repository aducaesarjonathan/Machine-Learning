import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Show first 9 training images
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.tight_layout()
plt.show()

# Normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build ANN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model summary
model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Show predictions for 10 random test images
print("\n--- Predictions on 10 Random Test Images ---\n")
random_indices = np.random.choice(len(x_test), 10, replace=False)

for i, idx in enumerate(random_indices):
    x_sample = x_test[idx]
    y_true = y_test[idx]
    y_pred_class = y_pred_classes[idx]

    plt.figure()
    plt.imshow(x_sample.reshape(28, 28), cmap='gray')
    plt.title(f"Image {i+1} → Predicted: {y_pred_class}, True: {y_true}", fontsize=14)
    plt.axis('off')
    plt.show()

# Confusion Matrix
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Error analysis
errors = (y_pred_classes != y_test)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_test[errors]
x_test_errors = x_test[errors]

# Prediction confidence
y_pred_errors_probability = np.max(y_pred_errors, axis=1)
true_probability_errors = np.array([y_pred_errors[i, label] for i, label in enumerate(y_true_errors)])
diff_errors_pred_true = y_pred_errors_probability - true_probability_errors

# Top 5 biggest mistakes
sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
top_idx_diff_errors = sorted_idx_diff_errors[-5:]

# Display top 5 mistakes
num = len(top_idx_diff_errors)
f, ax = plt.subplots(1, num, figsize=(20, 4))

for i in range(num):
    idx = top_idx_diff_errors[i]
    sample = x_test_errors[idx].reshape(28, 28)
    y_t = y_true_errors[idx]
    y_p = y_pred_classes_errors[idx]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title(f"Pred: {y_p}, True: {y_t}", fontsize=14)
    ax[i].axis('off')

plt.tight_layout()
plt.show()

# Save model
model.save('handwritten.model.keras')
print("Model saved as 'handwritten.model.keras'")

# Plot accuracy graph
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#plot_model(model, 
    #       to_file='model_architecture.png', 
   #        show_shapes=True, 
  #         show_layer_names=True,
 #          show_layer_activations=True) # This shows the activation functions

#print("Model architecture diagram saved as 'model_architecture.png'")