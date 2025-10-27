# Step 1: Setting Up Our Workspace
# ********************************************************************
# Import TensorFlow - our main machine learning library
import tensorflow as tf

# Import matplotlib for creating visualizations and plots
import matplotlib.pyplot as plt

# Import numpy for numerical operations
import numpy as np

# Import time for measuring training duration
import time

# Import specific components we'll need from TensorFlow's Keras API
from tensorflow.keras import Sequential # For building our neural network layer by layer
from tensorflow.keras.layers import Flatten, Dense # The specific types of layers we'll use

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up MNIST digit classification...")


# Step 2: Loading and Exploring Our Data
# ********************************************************************
# Define constants for better code clarity
NUM_CLASSES = 10  # Digits 0-9
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Load the MNIST dataset - this downloads it automatically the first time
# It returns training data (for learning) and test data (for final evaluation)
print("Loading MNIST dataset...")
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.mnist.load_data()

# Let's see what we're working with
print(f"Training images shape: {training_images.shape}") # Should be (60000, 28, 28)
print(f"Training labels shape: {training_labels.shape}") # Should be (60000,)
print(f"Test images shape: {testing_images.shape}") # Should be (10000, 28, 28)
print(f"Test labels shape: {testing_labels.shape}") # Should be (10000,)

# Let's examine the first few training labels to see what digits we have
print(f"First 10 training labels: {training_labels[:10]}")

# Check the range of pixel values in our images
print(f"Pixel value range: {training_images.min()} to {training_images.max()}")

# Let's visualize the first training image to see what we're dealing with
plt.figure(figsize=(6, 6))
plt.imshow(training_images[0], cmap='gray') # Display in grayscale
plt.title(f"First training image - Label: {training_labels[0]}")
plt.colorbar() # Show the color scale
plt.show()


# Step 3: Preparing Our Data for Training
# ********************************************************************
# Normalize pixel values from 0-255 range to 0-1 range
# This helps the neural network learn more efficiently
print("Normalizing pixel values...")
print(f"Before normalization - pixel range: {training_images.min()} to {training_images.max()}")

training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Verify the normalization worked
print(f"After normalization - pixel range: {training_images.min()} to {training_images.max()}")
print("Data normalization complete!")


# Step 4: Building Our Neural Network
# ********************************************************************
# Define architecture constants for clarity
HIDDEN_LAYER_SIZE = 128  # Number of neurons in hidden layer
INPUT_FEATURES = IMAGE_HEIGHT * IMAGE_WIDTH  # 28 * 28 = 784 features

# Create a Sequential model - this means layers are stacked one after another
print("Building neural network architecture...")
print(f"Input features: {INPUT_FEATURES}")
print(f"Hidden layer size: {HIDDEN_LAYER_SIZE}")
print(f"Output classes: {NUM_CLASSES}")

digit_classifier = tf.keras.models.Sequential()

# Layer 1: Flatten the 28x28 image into a 1D array of 784 values
# Think of this as unrolling the 2D image into a single line of pixels
digit_classifier.add(Flatten(input_shape=INPUT_SHAPE))

# Layer 2: Dense (fully connected) layer with 128 neurons
# Each neuron connects to all 784 input values and learns to detect patterns
# ReLU activation helps the network learn complex, non-linear patterns
digit_classifier.add(Dense(HIDDEN_LAYER_SIZE, activation='relu'))

# Layer 3: Output layer with 10 neurons (one for each digit 0-9)
# Softmax activation converts the outputs to probabilities that sum to 1
# The highest probability tells us which digit the network thinks it sees
digit_classifier.add(Dense(NUM_CLASSES, activation='softmax'))

# Let's see a summary of our network architecture
print("\nModel Architecture:")
digit_classifier.summary()


# Step 5: Configuring Our Model for Training
# ********************************************************************
# Configure the model for training
# Think of this as giving the model its learning strategy
print("Configuring model for training...")
print("Using Adam optimizer with sparse categorical crossentropy loss...")

digit_classifier.compile(
    optimizer='adam', # Adam optimizer - a smart way to adjust learning
    loss='sparse_categorical_crossentropy', # How to measure prediction errors
    metrics=['accuracy'] # Track accuracy during training
)

print("Model compiled and ready for training!")
print("Expected training accuracy: 85-95%")


# Step 6: Training Our Model
# ********************************************************************
# Training configuration
NUM_EPOCHS = 10  # Number of complete passes through training data
VALIDATION_SPLIT = 0.1  # Use 10% of training data for validation

# Train the model on our training data
print("Starting training...")
print("This will take a few minutes - watch the accuracy improve with each epoch!")
print(f"Training for {NUM_EPOCHS} epochs with {VALIDATION_SPLIT*100}% validation split...")

# Record training start time
training_start_time = time.time()

# Train for specified epochs
training_history = digit_classifier.fit(
    training_images,  # Input images
    training_labels,  # Correct labels
    epochs=NUM_EPOCHS,  # Number of times to go through all training data
    validation_split=VALIDATION_SPLIT,  # Use 10% of training data for validation
    verbose=1  # Show progress during training
)

# Calculate training duration
training_duration = time.time() - training_start_time
print(f"Training completed in {training_duration:.2f} seconds!")


# Step 7: Evaluating Our Model
# ********************************************************************
# Evaluate the model on test data it has never seen
print("Evaluating model on test data...")
print("Testing on data the model has never seen before...")

final_test_loss, final_test_accuracy = digit_classifier.evaluate(testing_images, testing_labels, verbose=0)

print(f"Final test accuracy: {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)")
print(f"Final test loss: {final_test_loss:.4f}")

# Let's also make some predictions and see what the model thinks
print("\nMaking predictions on first 5 test images...")
NUM_SAMPLE_PREDICTIONS = 5
model_predictions = digit_classifier.predict(testing_images[:NUM_SAMPLE_PREDICTIONS])

for sample_idx in range(NUM_SAMPLE_PREDICTIONS):
    predicted_digit = model_predictions[sample_idx].argmax()  # Get the digit with highest probability
    actual_digit = testing_labels[sample_idx]
    prediction_confidence = model_predictions[sample_idx].max()  # Get the confidence level

    print(f"Sample {sample_idx+1}: Predicted={predicted_digit}, Actual={actual_digit}, Confidence={prediction_confidence:.3f}")


# Step 8: Visualizing Our Results
# ********************************************************************
# Create a visualization of training progress
print("Creating training progress visualization...")
results_figure = plt.figure(figsize=(12, 4))

# Plot training accuracy over time
plt.subplot(1, 2, 1)
plt.plot(training_history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(training_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training loss over time
plt.subplot(1, 2, 2)
plt.plot(training_history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(training_history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training visualization complete!")


# Step 9: Testing Individual Predictions
# ********************************************************************
# Let's examine some predictions in detail
print("Creating detailed prediction analysis...")
prediction_analysis_figure = plt.figure(figsize=(15, 6))

for sample_idx in range(NUM_SAMPLE_PREDICTIONS):
    # Show the actual test image
    plt.subplot(2, 5, sample_idx+1)
    plt.imshow(testing_images[sample_idx], cmap='gray')
    plt.title(f"Actual: {testing_labels[sample_idx]}")
    plt.axis('off')

    # Show the prediction probabilities
    plt.subplot(2, 5, sample_idx+6)
    digit_probabilities = model_predictions[sample_idx]
    plt.bar(range(NUM_CLASSES), digit_probabilities)
    plt.title(f"Predicted: {digit_probabilities.argmax()}")
    plt.xlabel('Digit')
    plt.ylabel('Probability')

plt.tight_layout()
plt.show()

print("Prediction analysis complete!")