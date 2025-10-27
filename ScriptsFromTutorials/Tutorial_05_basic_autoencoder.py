# Tutorial 05: Basic Autoencoder Construction
# Step 1: Setting Up Our Autoencoder Laboratory
# ********************************************************************
# Import essential libraries for autoencoder implementation
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Input
from tensorflow.keras.layers import Conv2DTranspose

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up autoencoder construction laboratory...")
print("Today we'll learn how to compress and reconstruct images!")

# Configure matplotlib for better visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)


# Step 2: Understanding Autoencoder Architecture Through Visualization
# ********************************************************************
def visualize_autoencoder_concept():
	"""
	Create a conceptual visualization of how autoencoders work
	This helps understand the compression and reconstruction process
	"""
	print("\n" + "="*60)
	print("UNDERSTANDING AUTOENCODER ARCHITECTURE")
	print("="*60)
	
	# Load and prepare a sample image for demonstration
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	
	# Reshape and normalize for autoencoder processing
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255.0
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255.0
	
	# Create conceptual visualization
	plt.figure(figsize=(18, 6))
	
	# Original image
	plt.subplot(1, 6, 1)
	plt.imshow(test_images[0].squeeze(), cmap='gray')
	plt.title('Original Image\n28x28 = 784 pixels')
	plt.axis('off')
	
	# Simulated compression stages
	compressed_sizes = [196, 49, 16, 49, 196]
	stage_names = ['Compress 1\n14x14', 'Compress 2\n7x7', 'Latent Space\n4x4', 'Expand 1\n7x7', 'Expand 2\n14x14']
	
	for i, (size, name) in enumerate(zip(compressed_sizes, stage_names)):
		plt.subplot(1, 6, i + 2)
		
		# Create a simple visualization of compression/expansion
		if i < 2:  # Compression stages
			# Simulate compression by downsampling
			downsampled = tf.image.resize(test_images[0], [int(np.sqrt(size)), int(np.sqrt(size))])
			plt.imshow(downsampled.numpy().squeeze(), cmap='gray')
		elif i == 2:  # Latent space
			# Show the most compressed representation
			latent_viz = tf.image.resize(test_images[0], [4, 4])
			plt.imshow(latent_viz.numpy().squeeze(), cmap='viridis')
		else:  # Reconstruction stages
			# Simulate reconstruction by upsampling
			upsampled = tf.image.resize(test_images[0], [int(np.sqrt(size)), int(np.sqrt(size))])
			plt.imshow(upsampled.numpy().squeeze(), cmap='gray')
		
		plt.title(name)
		plt.axis('off')
	
	plt.suptitle('Autoencoder: Compression → Latent Space → Reconstruction')
	plt.tight_layout()
	plt.show()
	
	print("Key Concept: Autoencoders learn to compress data into a latent space")
	print("and then reconstruct it back to the original form.")
	print("The latent space captures the most important features!")
	
	return train_images, test_images, train_labels, test_labels

# Visualize the autoencoder concept
train_images, test_images, train_labels, test_labels = visualize_autoencoder_concept()


# Step 3: Building Your First Dense Autoencoder
# ********************************************************************
def build_dense_autoencoder(latent_dim=32):
	"""
	Build a simple dense autoencoder
	This helps understand the basic encoder-decoder pattern
	"""
	print(f"\n--- Building Dense Autoencoder with {latent_dim}D latent space ---")
	
	# Define the encoder
	encoder = Sequential([
		Flatten(input_shape=(28, 28, 1)),               # Flatten 28x28 image to 784 features
		Dense(128, activation='relu'),                  # First compression layer
		Dense(64, activation='relu'),                   # Second compression layer  
		Dense(latent_dim, activation='relu')    # Latent space representation
	], name='encoder')
	
	# Define the decoder
	decoder = Sequential([
		Dense(64, activation='relu', input_shape=(latent_dim,)),        # Start expanding
		Dense(128, activation='relu'),                                                  # Continue expanding
		Dense(784, activation='sigmoid'),                                               # Output layer (784 = 28*28)
		Reshape((28, 28, 1))                                                                    # Reshape back to image
	], name='decoder')
	
	# Combine encoder and decoder into autoencoder
	autoencoder_input = Input(shape=(28, 28, 1))
	encoded = encoder(autoencoder_input)
	decoded = decoder(encoded)
	
	autoencoder = Model(autoencoder_input, decoded, name='dense_autoencoder')
	
	print("Dense Autoencoder Architecture:")
	print(f"Input: 28x28x1 = {28*28} pixels")
	print(f"Encoder: {28*28} → 128 → 64 → {latent_dim}")
	print(f"Decoder: {latent_dim} → 64 → 128 → {28*28}")
	print(f"Output: 28x28x1 = {28*28} pixels")
	print(f"Compression ratio: {(28*28)/latent_dim:.1f}:1")
	
	return autoencoder, encoder, decoder

# Build and examine the dense autoencoder
dense_autoencoder, dense_encoder, dense_decoder = build_dense_autoencoder(latent_dim=32)

# Display model architectures
print("\n" + "="*60)
print("DENSE AUTOENCODER MODEL SUMMARIES")
print("="*60)

print("\nEncoder Summary:")
dense_encoder.summary()

print("\nDecoder Summary:")
dense_decoder.summary()

print("\nComplete Autoencoder Summary:")
dense_autoencoder.summary()

# Step 4: Training Your First Autoencoder
# ********************************************************************
def train_autoencoder(autoencoder, train_data, test_data, epochs=20):
	"""
	Train the autoencoder and monitor reconstruction quality
	"""
	print(f"\n--- Training Autoencoder for {epochs} epochs ---")
	
	# Compile the autoencoder
	# We use 'binary_crossentropy' because our images are normalized to [0,1]
	autoencoder.compile(
		optimizer='adam',
		loss='binary_crossentropy',             # Good for normalized images
		metrics=['mse']                         # Also track mean squared error
	)
	
	# Train the autoencoder
	# Important: For autoencoders, input and target are the same!
	history = autoencoder.fit(
		train_data, train_data,                 # Input = target for reconstruction
		epochs=epochs,
		batch_size=32,
		shuffle=True,
		validation_data=(test_data, test_data), # Same for validation
		verbose=1
	)
	
	print("Training completed!")
	return history

# Train the dense autoencoder
print("\n" + "="*60)
print("TRAINING DENSE AUTOENCODER")
print("="*60)

dense_history = train_autoencoder(dense_autoencoder, train_images, test_images, epochs=20)

# Visualize training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(dense_history.history['loss'], label='Training Loss')
plt.plot(dense_history.history['val_loss'], label='Validation Loss')
plt.title('Dense Autoencoder: Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(dense_history.history['mse'], label='Training MSE')
plt.plot(dense_history.history['val_mse'], label='Validation MSE')
plt.title('Dense Autoencoder: MSE During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Step 5: Evaluating Reconstruction Quality
# ********************************************************************
def evaluate_reconstruction_quality(autoencoder, test_data, test_labels, num_samples=10):
	"""
	Evaluate and visualize reconstruction quality
	"""
	print(f"\n--- Evaluating Reconstruction Quality ---")
	
	# Generate reconstructions
	reconstructions = autoencoder.predict(test_data[:num_samples])
	
	# Calculate reconstruction errors
	mse_errors = np.mean((test_data[:num_samples] - reconstructions) ** 2, axis=(1, 2, 3))
	
	# Create visualization
	plt.figure(figsize=(15, 6))
	
	for i in range(num_samples):
		# Original image
		plt.subplot(3, num_samples, i + 1)
		plt.imshow(test_data[i].squeeze(), cmap='gray')
		plt.title(f'Original\nDigit: {test_labels[i]}')
		plt.axis('off')
		
		# Reconstructed image
		plt.subplot(3, num_samples, i + 1 + num_samples)
		plt.imshow(reconstructions[i].squeeze(), cmap='gray')
		plt.title(f'Reconstructed\nMSE: {mse_errors[i]:.4f}')
		plt.axis('off')
		
		# Difference (error) image
		plt.subplot(3, num_samples, i + 1 + 2*num_samples)
		difference = np.abs(test_data[i] - reconstructions[i])
		plt.imshow(difference.squeeze(), cmap='hot')
		plt.title(f'Difference\nMax: {difference.max():.3f}')
		plt.axis('off')
	
	plt.suptitle('Autoencoder Reconstruction Quality Analysis')
	plt.tight_layout()
	plt.show()
	
	# Print statistics
	print(f"Average MSE: {np.mean(mse_errors):.4f}")
	print(f"Best reconstruction (lowest MSE): {np.min(mse_errors):.4f}")
	print(f"Worst reconstruction (highest MSE): {np.max(mse_errors):.4f}")
	
	return mse_errors

# Evaluate reconstruction quality
reconstruction_errors = evaluate_reconstruction_quality(dense_autoencoder, test_images, test_labels)


# Step 6: Building a Convolutional Autoencoder
# ********************************************************************
def build_convolutional_autoencoder():
	"""
	Build a convolutional autoencoder for better image reconstruction
	This preserves spatial structure better than dense autoencoders
	"""
	print(f"\n--- Building Convolutional Autoencoder ---")
	
	# Define the encoder (downsampling path)
	encoder = Sequential([
		Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
		MaxPooling2D((2, 2), padding='same'),           # 28x28 -> 14x14
		Conv2D(64, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),           # 14x14 -> 7x7
		Conv2D(128, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),           # 7x7 -> 4x4 (with padding)
	], name='conv_encoder')
	
	# Define the decoder (upsampling path)
	decoder = Sequential([
		Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(4, 4, 128)),
		UpSampling2D((2, 2)),                                   # 4x4 -> 8x8
		Conv2D(64, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),                                   # 8x8 -> 16x16
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),                                   # 16x16 -> 32x32
		Conv2D(1, (3, 3), activation='sigmoid', padding='same'),        # Output layer
	], name='conv_decoder')
	
	# Combine encoder and decoder
	autoencoder_input = Input(shape=(28, 28, 1))
	encoded = encoder(autoencoder_input)
	decoded = decoder(encoded)
	
	# Crop the output to match input size (32x32 -> 28x28)
	cropped = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(decoded)
	
	conv_autoencoder = Model(autoencoder_input, cropped, name='conv_autoencoder')
	
	print("Convolutional Autoencoder Architecture:")
	print("Encoder: 28x28x1 → 14x14x32 → 7x7x64 → 4x4x128")
	print("Decoder: 4x4x128 → 8x8x128 → 16x16x64 → 32x32x32 → 28x28x1")
	
	return conv_autoencoder, encoder, decoder

# Build the convolutional autoencoder
conv_autoencoder, conv_encoder, conv_decoder = build_convolutional_autoencoder()


# Step 7: Training and Comparing Both Autoencoders
# ********************************************************************
def compare_autoencoder_performance():
	"""
	Train convolutional autoencoder and compare with dense version
	"""
	print("\n" + "="*60)
	print("TRAINING CONVOLUTIONAL AUTOENCODER")
	print("="*60)
	
	# Train convolutional autoencoder
	conv_history = train_autoencoder(conv_autoencoder, train_images, test_images, epochs=20)
	
	# Compare reconstruction quality
	print("\n" + "="*60)
	print("COMPARING AUTOENCODER PERFORMANCE")
	print("="*60)
	
	# Generate reconstructions from both models
	num_samples = 10
	test_sample = test_images[:num_samples]
	
	dense_reconstructions = dense_autoencoder.predict(test_sample)
	conv_reconstructions = conv_autoencoder.predict(test_sample)
	
	# Calculate reconstruction errors
	dense_mse = np.mean((test_sample - dense_reconstructions) ** 2, axis=(1, 2, 3))
	conv_mse = np.mean((test_sample - conv_reconstructions) ** 2, axis=(1, 2, 3))
	
	# Create comparison visualization
	plt.figure(figsize=(15, 12))
	
	for i in range(num_samples):
		# Original
		plt.subplot(4, num_samples, i + 1)
		plt.imshow(test_sample[i].squeeze(), cmap='gray')
		plt.title(f'Original\nDigit: {test_labels[i]}')
		plt.axis('off')
		
		# Dense reconstruction
		plt.subplot(4, num_samples, i + 1 + num_samples)
		plt.imshow(dense_reconstructions[i].squeeze(), cmap='gray')
		plt.title(f'Dense\nMSE: {dense_mse[i]:.4f}')
		plt.axis('off')
		
		# Convolutional reconstruction
		plt.subplot(4, num_samples, i + 1 + 2*num_samples)
		plt.imshow(conv_reconstructions[i].squeeze(), cmap='gray')
		plt.title(f'Conv\nMSE: {conv_mse[i]:.4f}')
		plt.axis('off')
		
		# Difference between the two reconstructions
		plt.subplot(4, num_samples, i + 1 + 3*num_samples)
		diff = np.abs(dense_reconstructions[i] - conv_reconstructions[i])
		plt.imshow(diff.squeeze(), cmap='hot')
		plt.title(f'Diff\nMax: {diff.max():.3f}')
		plt.axis('off')
	
	plt.suptitle('Dense vs Convolutional Autoencoder Comparison')
	plt.tight_layout()
	plt.show()
	
	# Print comparison statistics
	print(f"Dense Autoencoder - Average MSE: {np.mean(dense_mse):.4f}")
	print(f"Convolutional Autoencoder - Average MSE: {np.mean(conv_mse):.4f}")
	print(f"Improvement: {((np.mean(dense_mse) - np.mean(conv_mse)) / np.mean(dense_mse)) * 100:.1f}%")
	
	return conv_history

# Compare autoencoder performance
conv_training_history = compare_autoencoder_performance()