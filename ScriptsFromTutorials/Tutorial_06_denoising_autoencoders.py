# Tutorial 06: Denoising Autoencoders & Robust Learning
# Step 1: Setting Up Our Denoising Laboratory
# ********************************************************************
# Import essential libraries for denoising autoencoder implementation
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Input
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization
import random

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up denoising autoencoder laboratory...")
print("Today we'll learn how to clean corrupted images and build robust representations!")

# Configure matplotlib for better visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 8)


# Step 2: Understanding Noise Types Through Visualization
# ********************************************************************
def load_and_prepare_data():
	"""
	Load MNIST data and prepare it for denoising experiments
	"""
	print("\n" + "="*60)
	print("LOADING AND PREPARING DATA FOR DENOISING")
	print("="*60)
	
	# Load MNIST dataset
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	
	# Reshape and normalize for autoencoder processing
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255.0
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255.0
	
	print(f"Training images: {train_images.shape}")
	print(f"Test images: {test_images.shape}")
	print("Data normalized to [0, 1] range for optimal autoencoder training")
	
	return train_images, test_images, train_labels, test_labels

def create_noise_types(clean_images, noise_type='gaussian', intensity=0.3):
	"""
	Create different types of noise and corruption for training denoising autoencoders
	This function demonstrates various real-world corruption scenarios
	"""
	print(f"\n--- Creating {noise_type} noise with intensity {intensity} ---")
	
	if noise_type == 'gaussian':
		# Gaussian noise: adds random values from normal distribution
		noise = np.random.normal(0, intensity, clean_images.shape)
		noisy_images = clean_images + noise
		noisy_images = np.clip(noisy_images, 0, 1)  # Keep values in [0,1] range
		
	elif noise_type == 'salt_pepper':
		# Salt and pepper noise: random black and white pixels
		noisy_images = clean_images.copy()
		num_pixels = int(intensity * clean_images.size)
		
		# Add salt (white pixels)
		salt_coords = [np.random.randint(0, i, num_pixels//2) for i in clean_images.shape]
		noisy_images[tuple(salt_coords)] = 1
		
		# Add pepper (black pixels)  
		pepper_coords = [np.random.randint(0, i, num_pixels//2) for i in clean_images.shape]
		noisy_images[tuple(pepper_coords)] = 0
		
	elif noise_type == 'masking':
		# Random masking: blocks of missing data
		noisy_images = clean_images.copy()
		mask = np.random.random(clean_images.shape) > intensity
		noisy_images = noisy_images * mask
		
	elif noise_type == 'dropout':
		# Dropout noise: randomly zero out pixels
		mask = np.random.random(clean_images.shape) > intensity
		noisy_images = clean_images * mask
		
	else:
		raise ValueError(f"Unknown noise type: {noise_type}")
	
	return noisy_images

# Load the data
train_images, test_images, train_labels, test_labels = load_and_prepare_data()

# Create different types of corrupted data for experimentation
print("\n" + "="*60)
print("CREATING DIFFERENT TYPES OF CORRUPTION")
print("="*60)

gaussian_noisy = create_noise_types(test_images[:100], 'gaussian', 0.3)
salt_pepper_noisy = create_noise_types(test_images[:100], 'salt_pepper', 0.1)
masked_noisy = create_noise_types(test_images[:100], 'masking', 0.4)
dropout_noisy = create_noise_types(test_images[:100], 'dropout', 0.3)


# Step 3: Visualizing Corruption Types
# ********************************************************************
def visualize_corruption_types():
	"""
	Create a comprehensive visualization of different corruption types
	This helps understand what challenges our denoising autoencoder must solve
	"""
	print("\n" + "="*60)
	print("VISUALIZING DIFFERENT CORRUPTION TYPES")
	print("="*60)
	
	# Select a few sample images for demonstration
	sample_indices = [0, 1, 2, 3, 4]
	
	# Create corrupted versions
	clean_samples = test_images[sample_indices]
	gaussian_samples = create_noise_types(clean_samples, 'gaussian', 0.3)
	salt_pepper_samples = create_noise_types(clean_samples, 'salt_pepper', 0.1)
	masked_samples = create_noise_types(clean_samples, 'masking', 0.4)
	dropout_samples = create_noise_types(clean_samples, 'dropout', 0.3)
	
	# Create comprehensive visualization
	plt.figure(figsize=(18, 12))
	
	corruption_types = [
		('Clean Original', clean_samples),
		('Gaussian Noise', gaussian_samples),
		('Salt & Pepper', salt_pepper_samples),
		('Random Masking', masked_samples),
		('Dropout Noise', dropout_samples)
	]
	
	for row, (corruption_name, corrupted_data) in enumerate(corruption_types):
		for col in range(5):
			plt.subplot(5, 5, row * 5 + col + 1)
			plt.imshow(corrupted_data[col].squeeze(), cmap='gray')
			
			if col == 0:  # Add corruption type label on first column
				plt.ylabel(corruption_name, fontsize=12, fontweight='bold')
			
			if row == 0:  # Add digit labels on first row
				plt.title(f'Digit: {test_labels[sample_indices[col]]}')
			
			plt.axis('off')
	
	plt.suptitle('Different Types of Image Corruption for Denoising Training', fontsize=16)
	plt.tight_layout()
	plt.show()
	
	print("Notice how each corruption type presents different challenges:")
	print("• Gaussian Noise: Overall image degradation, needs smoothing")
	print("• Salt & Pepper: Isolated pixel errors, needs local correction")
	print("• Random Masking: Missing regions, needs inpainting")
	print("• Dropout Noise: Sparse missing data, needs interpolation")

# Visualize all corruption types
visualize_corruption_types()


# Step 4: Building Your First Denoising Autoencoder
# ********************************************************************
def build_denoising_autoencoder():
	"""
	Build a convolutional denoising autoencoder
	This architecture is specifically designed for noise removal and image restoration
	"""
	print("\n" + "="*60)
	print("BUILDING DENOISING AUTOENCODER")
	print("="*60)
	
	# Encoder: Learns robust features from noisy input
	encoder = Sequential([
		# First conv block: Extract low-level features
		Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
		BatchNormalization(),                                           # Stabilize training
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),           # 28x28 -> 14x14
		
		# Second conv block: Learn mid-level features
		Conv2D(64, (3, 3), activation='relu', padding='same'),
		BatchNormalization(),
		Conv2D(64, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),           # 14x14 -> 7x7
		
		# Third conv block: High-level feature extraction
		Conv2D(128, (3, 3), activation='relu', padding='same'),
		BatchNormalization(),
		Conv2D(128, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),           # 7x7 -> 4x4 (with padding)
	], name='denoising_encoder')
	
	# Decoder: Reconstructs clean image from robust features
	decoder = Sequential([
		# First upsampling block: Begin reconstruction
		Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(4, 4, 128)),
		BatchNormalization(),
		Conv2D(128, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),                                   # 4x4 -> 8x8
		
		# Second upsampling block: Recover spatial details
		Conv2D(64, (3, 3), activation='relu', padding='same'),
		BatchNormalization(),
		Conv2D(64, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),                                   # 8x8 -> 16x16
		
		# Third upsampling block: Final reconstruction
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		BatchNormalization(),
		Conv2D(32, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),                                   # 16x16 -> 32x32
		
		# Output layer: Generate clean image
		Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
		# Crop to original size
		tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))  # 32x32 -> 28x28
	], name='denoising_decoder')
	
	# Combine encoder and decoder
	noisy_input = Input(shape=(28, 28, 1), name='noisy_input')
	encoded = encoder(noisy_input)
	clean_output = decoder(encoded)
	
	denoising_autoencoder = Model(noisy_input, clean_output, name='denoising_autoencoder')
	
	print("Denoising Autoencoder Architecture:")
	print("Input: Noisy/corrupted 28x28 image")
	print("Encoder: 28x28 → 14x14 → 7x7 → 4x4 (with 32→64→128 filters)")
	print("Decoder: 4x4 → 8x8 → 16x16 → 32x32 → 28x28 (with 128→64→32→1 filters)")
	
	return denoising_autoencoder, encoder, decoder

# Build the denoising autoencoder
denoising_model, denoising_encoder, denoising_decoder = build_denoising_autoencoder()


# Step 5: Training the Denoising Autoencoder
# ********************************************************************
def train_denoising_autoencoder(model, clean_images, noise_type='gaussian', noise_intensity=0.3, epochs=5):
	"""
	Train the denoising autoencoder with corrupted inputs and clean targets
	This is the key difference: input ≠ target (unlike regular autoencoders)
	"""
	print(f"\n--- Training Denoising Autoencoder with {noise_type} noise ---")
	
	# Create corrupted training data
	print("Creating corrupted training data...")
	noisy_train_images = create_noise_types(clean_images, noise_type, noise_intensity)
	
	# Create corrupted validation data  
	noisy_test_images = create_noise_types(test_images, noise_type, noise_intensity)
	
	# Compile the model
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
		loss='binary_crossentropy',                     # Good for [0,1] normalized images
		metrics=['mse', 'mae']                          # Track multiple metrics
	)
	
	# Setup training callbacks for better training
	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=5,
			restore_best_weights=True,
			verbose=1
		),
		tf.keras.callbacks.ReduceLROnPlateau(
			monitor='val_loss',
			factor=0.5,
			patience=3,
			min_lr=0.00001,
			verbose=1
		)
	]
	
	print(f"Training for up to {epochs} epochs with early stopping...")
	print("Key training concept: Input = noisy images, Target = clean images")
	
	# Train the model
	# CRITICAL: noisy_train_images as input, clean_images as target!
	history = model.fit(
		noisy_train_images, clean_images,               # Input corrupted, target clean
		epochs=epochs,
		batch_size=32,
		shuffle=True,
		validation_data=(noisy_test_images, test_images),  # Same for validation
		callbacks=callbacks,
		verbose=1
	)
	
	print("Training completed!")
	return history

# Train the denoising autoencoder with Gaussian noise
print("\n" + "="*60)
print("TRAINING DENOISING AUTOENCODER WITH GAUSSIAN NOISE")
print("="*60)


def visualize_denoising_effect(model, clean_images, noise_type='gaussian', noise_intensity=0.3, n=10):
    """
    Visualize how the denoising autoencoder removes noise from images.
    Shows original, noisy, and denoised images side by side.
    """
    # Select n random test images
    idx = np.random.choice(len(clean_images), n, replace=False)
    original = clean_images[idx]
    noisy = create_noise_types(original, noise_type, noise_intensity)
    denoised = model.predict(noisy)

    plt.figure(figsize=(18, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i].squeeze(), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')

    plt.suptitle("Denoising Autoencoder Results", fontsize=16)
    plt.tight_layout()
    plt.show()

denoising_history = train_denoising_autoencoder(
	denoising_model, 
	train_images, 
	noise_type='gaussian', 
	noise_intensity=0.3, 
	epochs=5
)

print("\n" + "="*60)
print("VISUALIZING DENOISING AUTOENCODER WITH GAUSSIAN NOISE")
print("="*60)

visualize_denoising_effect(
    denoising_model,
    test_images,
    noise_type='gaussian',     # Use the same noise type as training
    noise_intensity=0.3,       # Use the same intensity as training
    n=10                       # Number of images to display
)


# ********************************************************************


