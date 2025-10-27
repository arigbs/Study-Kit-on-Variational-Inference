# Tutorial 03: CNN Implementation for Image Processing
# Step 1: Setting Up Our CNN Learning Environment
# ********************************************************************
# Import our essential libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sys

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up CNN implementation environment...")
print("Today we'll learn how CNNs 'see' images!")

# Step 2: Loading and Exploring Our Data (CNN Perspective)
# ********************************************************************
# Load MNIST data - same as before, but now we'll reshape it for CNNs
print("Loading MNIST dataset for CNN processing...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# CNN-specific preprocessing: reshape to include channel dimension
# CNNs expect images in format (height, width, channels)
# MNIST is grayscale, so we have 1 channel
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Normalize pixel values to 0-1 range
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

print(f"Training images shape: {train_images.shape}")  # Should be (60000, 28, 28, 1)
print(f"Test images shape: {test_images.shape}")       # Should be (10000, 28, 28, 1)
print("Data prepared for CNN processing!")


# Step 3: Understanding Convolution Through Visualization
# ********************************************************************
def visualize_convolution_effect(image_index=0):
	"""
	Visualize how different convolution filters affect an image
	This helps us understand what CNNs are actually 'seeing'
	"""
	# Get a sample image
	sample_image = train_images[image_index].reshape(28, 28)
	
	# Define some common edge detection filters
	# These are simplified versions of what CNNs learn automatically
	filters = {
		'Original': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
		'Horizontal Edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
		'Vertical Edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
		'Diagonal Edge': np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
	}
	
	# Apply each filter and visualize the results
	plt.figure(figsize=(15, 4))
	
	for i, (name, filter_kernel) in enumerate(filters.items()):
		plt.subplot(1, 4, i+1)
		
		if name == 'Original':
			# Show original image
			plt.imshow(sample_image, cmap='gray')
		else:
			# Apply convolution manually (simplified)
			from scipy import ndimage
			filtered_image = ndimage.convolve(sample_image, filter_kernel)
			plt.imshow(filtered_image, cmap='gray')
		
		plt.title(f'{name}')
		plt.axis('off')
	
	plt.suptitle(f'How Different Filters See Digit: {train_labels[image_index]}')
	plt.tight_layout()
	plt.show()
	
	print(f"Notice how each filter highlights different features!")
	print(f"CNNs learn these filters automatically to detect patterns!")

# Let's see convolution in action
print("\n" + "="*50)
print("UNDERSTANDING CONVOLUTION THROUGH VISUALIZATION")
print("="*50)

# Install scipy if needed for the convolution visualization
try:
	from scipy import ndimage
	visualize_convolution_effect(0)
except ImportError:
	print("Installing scipy for convolution visualization...")
	import subprocess
	subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
	visualize_convolution_effect(0)


# Step 4: Building Your First CNN Architecture
# ********************************************************************
def create_cnn_model(model_name="Basic CNN"):
	"""
	Create a CNN model with detailed explanations of each layer
	"""
	print(f"\n--- Building {model_name} ---")
	
	# Create sequential model
	model = Sequential(name=model_name)
	
	# First Convolutional Layer
	# 32 filters, each 3x3 pixels, with ReLU activation
	# This layer learns to detect basic features like edges and curves
	print("Adding first convolutional layer: 32 filters, 3x3 size")
	model.add(Conv2D(
		filters=32,             # Number of different patterns to learn
		kernel_size=(3, 3),     # Size of each filter (3x3 pixels)
		activation='relu',      # ReLU activation for non-linearity
		input_shape=(28, 28, 1) # Input shape: 28x28 grayscale images
	))
	
	# First Pooling Layer
	# Reduces image size while keeping important features
	# 2x2 pooling reduces 28x28 to 14x14
	print("Adding first pooling layer: 2x2 max pooling")
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Second Convolutional Layer
	# 64 filters to learn more complex patterns
	# At this point, the network learns combinations of basic features
	print("Adding second convolutional layer: 64 filters, 3x3 size")
	model.add(Conv2D(
		filters=64,
		kernel_size=(3, 3),
		activation='relu'
	))
	
	# Second Pooling Layer
	# Further reduces size and focuses on most important features
	print("Adding second pooling layer: 2x2 max pooling")
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# Third Convolutional Layer
	# 64 filters for even more complex pattern recognition
	print("Adding third convolutional layer: 64 filters, 3x3 size")
	model.add(Conv2D(
		filters=64,
		kernel_size=(3, 3),
		activation='relu'
	))
	
	# Flatten Layer
	# Converts 2D feature maps into 1D vector for dense layers
	print("Adding flatten layer: converting 2D features to 1D")
	model.add(Flatten())
	
	# Dense Layer
	# 64 neurons for high-level feature combination
	print("Adding dense layer: 64 neurons with ReLU")
	model.add(Dense(64, activation='relu'))
	
	# Dropout Layer
	# Randomly sets 50% of neurons to 0 during training
	# This prevents overfitting by forcing the network to be more robust
	print("Adding dropout layer: 50% dropout rate")
	model.add(Dropout(0.5))
	
	# Output Layer
	# 10 neurons for digit classification (0-9)
	print("Adding output layer: 10 neurons with softmax")
	model.add(Dense(10, activation='softmax'))
	
	print(f"{model_name} architecture complete!")
	return model

# Create our first CNN
print("\n" + "="*50)
print("BUILDING YOUR FIRST CNN")
print("="*50)

cnn_model = create_cnn_model("My_First_CNN")

# Display the model architecture
print("\n" + "="*50)
print("CNN MODEL ARCHITECTURE SUMMARY")
print("="*50)
cnn_model.summary()

# Let's understand what each layer does to our image dimensions
print("\n" + "="*50)
print("UNDERSTANDING LAYER TRANSFORMATIONS")
print("="*50)
print("Input: 28x28x1 (height x width x channels)")
print("Conv2D(32, 3x3) → 26x26x32 (lost 2 pixels on each side)")
print("MaxPooling2D(2x2) → 13x13x32 (halved dimensions)")
print("Conv2D(64, 3x3) → 11x11x64 (lost 2 pixels on each side)")
print("MaxPooling2D(2x2) → 5x5x64 (halved dimensions)")
print("Conv2D(64, 3x3) → 3x3x64 (lost 2 pixels on each side)")
print("Flatten() → 576 (3x3x64 = 576 features)")
print("Dense(64) → 64 neurons")
print("Dense(10) → 10 output probabilities")


# Step 5: Compiling and Training Your CNN
# ********************************************************************
# Compile the CNN model
print("\n" + "="*50)
print("COMPILING CNN MODEL")
print("="*50)

cnn_model.compile(
	optimizer='adam',                       # Adam optimizer works well for CNNs
	loss='sparse_categorical_crossentropy', # For multi-class classification
	metrics=['accuracy']                    # Track accuracy during training
)

print("CNN model compiled successfully!")

# Train the CNN model
print("\n" + "="*50)
print("TRAINING CNN MODEL")
print("="*50)
print("Training will take longer than dense networks, but results will be better!")

# Start training with timing
import time
start_time = time.time()

# Train for 10 epochs with validation split
cnn_history = cnn_model.fit(
	train_images,
	train_labels,
	epochs=10,
	batch_size=32,          # Process 32 images at a time
	validation_split=0.1,   # Use 10% of training data for validation
	verbose=1               # Show progress
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Evaluate the CNN model
print("\n" + "="*50)
print("EVALUATING CNN PERFORMANCE")
print("="*50)

cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(test_images, test_labels, verbose=0)
print(f"CNN Test accuracy: {cnn_test_accuracy:.4f} ({cnn_test_accuracy*100:.2f}%)")


# Step 6: Comparing CNN vs Dense Network Performance
# ********************************************************************
# Build a comparable dense network for comparison
print("\n" + "="*50)
print("BUILDING DENSE NETWORK FOR COMPARISON")
print("="*50)

dense_model = Sequential([
	Flatten(input_shape=(28, 28, 1)),
	Dense(128, activation='relu'),
	Dense(10, activation='softmax')
])

dense_model.compile(
	optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy']
)

print("Training dense network for comparison...")
dense_history = dense_model.fit(
	train_images,
	train_labels,
	epochs=10,
	batch_size=32,
	validation_split=0.1,
	verbose=1
)

dense_test_loss, dense_test_accuracy = dense_model.evaluate(test_images, test_labels, verbose=0)

print("\n" + "="*50)
print("PERFORMANCE COMPARISON")
print("="*50)
print(f"Dense Network Accuracy: {dense_test_accuracy:.4f} ({dense_test_accuracy*100:.2f}%)")
print(f"CNN Accuracy: {cnn_test_accuracy:.4f} ({cnn_test_accuracy*100:.2f}%)")
print(f"Improvement: {(cnn_test_accuracy - dense_test_accuracy)*100:.2f} percentage points")

if cnn_test_accuracy > dense_test_accuracy:
	print("🎉 CNN performs better! This shows the power of spatial feature learning.")
else:
	print("Interesting! Sometimes the improvement is subtle, but CNNs are more robust.")


# Step 7: Visualizing Training Progress
# ********************************************************************
# Create detailed training visualizations
print("\n" + "="*50)
print("VISUALIZING TRAINING PROGRESS")
print("="*50)

plt.figure(figsize=(15, 10))

# Plot 1: CNN Training Accuracy
plt.subplot(2, 3, 1)
plt.plot(cnn_history.history['accuracy'], label='CNN Training', linewidth=2)
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation', linewidth=2)
plt.title('CNN: Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: CNN Training Loss
plt.subplot(2, 3, 2)
plt.plot(cnn_history.history['loss'], label='CNN Training', linewidth=2)
plt.plot(cnn_history.history['val_loss'], label='CNN Validation', linewidth=2)
plt.title('CNN: Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Dense Network Training Accuracy
plt.subplot(2, 3, 3)
plt.plot(dense_history.history['accuracy'], label='Dense Training', linewidth=2)
plt.plot(dense_history.history['val_accuracy'], label='Dense Validation', linewidth=2)
plt.title('Dense Network: Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Direct Accuracy Comparison
plt.subplot(2, 3, 4)
plt.plot(cnn_history.history['accuracy'], label='CNN Training', linewidth=2)
plt.plot(dense_history.history['accuracy'], label='Dense Training', linewidth=2)
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Final Performance Comparison
plt.subplot(2, 3, 5)
models = ['Dense Network', 'CNN']
accuracies = [dense_test_accuracy, cnn_test_accuracy]
colors = ['blue', 'red']
bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
plt.title('Final Test Accuracy Comparison')
plt.ylabel('Test Accuracy')
plt.ylim(0.95, 1.0)  # Zoom in on the high accuracy range
for bar, acc in zip(bars, accuracies):
	plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
					 f'{acc:.3f}', ha='center', va='bottom')

# Plot 6: Model Parameter Comparison
plt.subplot(2, 3, 6)
cnn_params = cnn_model.count_params()
dense_params = dense_model.count_params()
models = ['Dense Network', 'CNN']
params = [dense_params, cnn_params]
bars = plt.bar(models, params, color=['blue', 'red'], alpha=0.7)
plt.title('Model Parameter Count')
plt.ylabel('Number of Parameters')
for bar, param in zip(bars, params):
	plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
					 f'{param:,}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"CNN Parameters: {cnn_params:,}")
print(f"Dense Parameters: {dense_params:,}")
print(f"Parameter difference: {abs(cnn_params - dense_params):,}")


# Step 8: Visualizing What the CNN Learned
# ********************************************************************
def visualize_cnn_filters(model, layer_name, num_filters=8):
	"""
	Visualize the filters learned by a convolutional layer
	"""
	# Get the layer
	layer = model.get_layer(layer_name)
	weights = layer.get_weights()[0]  # Get filter weights
	
	# Normalize weights for visualization
	weights = (weights - weights.min()) / (weights.max() - weights.min())
	
	# Plot filters
	plt.figure(figsize=(12, 6))
	for i in range(min(num_filters, weights.shape[3])):
		plt.subplot(2, 4, i+1)
		# Show the filter
		filter_img = weights[:, :, 0, i]  # Get the i-th filter
		plt.imshow(filter_img, cmap='gray')
		plt.title(f'Filter {i+1}')
		plt.axis('off')
	
	plt.suptitle(f'Learned Filters in {layer_name}')
	plt.tight_layout()
	plt.show()

def visualize_feature_maps(model, image_index=0, layer_name=None):
	"""
	Visualize feature maps produced by a convolutional layer
	"""
	# Get a sample image
	sample_image = test_images[image_index:image_index+1]
	
	# Create a model that outputs feature maps
	if layer_name is None:
		layer_name = 'conv2d'  # First conv layer
	
	feature_model = tf.keras.Model(
		inputs=model.inputs,
		outputs=model.get_layer(layer_name).output
	)
	
	# Get feature maps
	feature_maps = feature_model.predict(sample_image)
	
	# Visualize first 8 feature maps
	plt.figure(figsize=(12, 6))
	for i in range(min(8, feature_maps.shape[3])):
		plt.subplot(2, 4, i+1)
		plt.imshow(feature_maps[0, :, :, i], cmap='gray')
		plt.title(f'Feature Map {i+1}')
		plt.axis('off')
	
	plt.suptitle(f'Feature Maps from {layer_name} - Input: {test_labels[image_index]}')
	plt.tight_layout()
	plt.show()

# Visualize what our CNN learned
print("\n" + "="*50)
print("VISUALIZING CNN LEARNED FEATURES")
print("="*50)

# Show original image
plt.figure(figsize=(4, 4))
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f'Original Image - Label: {test_labels[0]}')
plt.axis('off')
plt.show()

# Visualize learned filters
print("Learned filters in first convolutional layer:")
visualize_cnn_filters(cnn_model, 'conv2d', num_filters=8)

# Visualize feature maps
print("Feature maps produced by first convolutional layer:")
visualize_feature_maps(cnn_model, image_index=0, layer_name='conv2d')