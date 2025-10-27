# Tutorial 04: Understanding Feature Extraction & Convolution
# Step 1: Setting Up Our Advanced Convolution Laboratory
# ********************************************************************
# Import our essential libraries for advanced convolution analysis
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, SeparableConv2D
import seaborn as sns

# Set random seeds for reproducible experiments
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up advanced convolution analysis environment...")
print("Today we'll master the art and science of feature extraction!")

# Configure plotting for better visualizations
plt.style.use('default')
sns.set_palette("husl")


# Step 2: Loading and Preparing Our Data for Advanced Analysis
# ********************************************************************
# Load MNIST data with enhanced preprocessing for feature analysis
print("Loading MNIST dataset for advanced convolution analysis...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Reshape for CNN processing (add channel dimension)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalize to 0-1 range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a subset for detailed analysis (faster experimentation)
analysis_images = test_images[:100]  # 100 images for detailed analysis
analysis_labels = test_labels[:100]

print(f"Training images: {train_images.shape}")
print(f"Test images: {test_images.shape}")
print(f"Analysis subset: {analysis_images.shape}")
print("Data preparation complete!")

# Let's examine some sample images for our analysis
plt.figure(figsize=(12, 4))
for i in range(6):
	plt.subplot(2, 3, i+1)
	plt.imshow(test_images[i].squeeze(), cmap='gray')
	plt.title(f'Digit: {test_labels[i]}')
	plt.axis('off')
plt.suptitle('Sample Images for Feature Extraction Analysis')
plt.tight_layout()
plt.show()


# Step 3: Understanding Filter Sizes and Their Impact
# ********************************************************************
def analyze_filter_sizes():
	"""
	Comprehensive analysis of different filter sizes and their effects
	This function will help you understand the trade-offs between filter sizes
	"""
	print("\n" + "="*60)
	print("ANALYZING FILTER SIZES AND THEIR IMPACT")
	print("="*60)
	
	# Define different filter sizes to test
	filter_sizes = [(1, 1), (3, 3), (5, 5), (7, 7)]
	
	# Store results for comparison
	filter_results = {}
	
	for filter_size in filter_sizes:
		print(f"\n--- Testing {filter_size[0]}x{filter_size[1]} filters ---")
		
		# Create a simple CNN with specified filter size
		model = Sequential([
			Conv2D(32, filter_size, activation='relu', input_shape=(28, 28, 1)),
			MaxPooling2D((2, 2)),
			Conv2D(64, filter_size, activation='relu'),
			MaxPooling2D((2, 2)),
			Flatten(),
			Dense(64, activation='relu'),
			Dense(10, activation='softmax')
		])
		
		# Compile the model
		model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)
		
		# Train briefly to see immediate effects
		print(f"Training model with {filter_size[0]}x{filter_size[1]} filters...")
		history = model.fit(
			train_images[:5000],    # Use subset for faster training
			train_labels[:5000],
			epochs=3,                               # Just a few epochs for comparison
			validation_split=0.2,
			verbose=1
		)
		
		# Evaluate on test data
		test_loss, test_accuracy = model.evaluate(test_images[:1000], test_labels[:1000], verbose=0)
		
		# Calculate model parameters
		total_params = model.count_params()
		
		# Store results
		filter_results[filter_size] = {
			'accuracy': test_accuracy,
			'parameters': total_params,
			'model': model,
			'history': history
		}
		
		print(f"Filter {filter_size[0]}x{filter_size[1]}: Accuracy={test_accuracy:.4f}, Parameters={total_params:,}")
	
	return filter_results

# Run the filter size analysis
filter_analysis = analyze_filter_sizes()


# Step 4: Exploring Padding Strategies - SAME vs VALID
# ********************************************************************
def demonstrate_padding_effects():
	"""
	Visual demonstration of how SAME and VALID padding affect feature maps
	This is crucial for understanding how to preserve or reduce spatial dimensions
	"""
	print("\n" + "="*60)
	print("DEMONSTRATING PADDING EFFECTS")
	print("="*60)
	
	# Create input tensor for demonstration
	sample_input = test_images[0:1]  # Single image batch
	print(f"Input shape: {sample_input.shape}")
	
	# Test different padding strategies
	padding_types = ['valid', 'same']
	
	results = {}
	
	for padding in padding_types:
		print(f"\n--- Testing '{padding.upper()}' padding ---")
		
		# Create layers with different padding
		conv_layer = Conv2D(1, (3, 3), padding=padding, activation='relu')
		
		# Apply convolution
		output = conv_layer(sample_input)
		
		print(f"Input shape: {sample_input.shape}")
		print(f"Output shape with {padding} padding: {output.shape}")
		
		# Calculate dimension change
		input_height, input_width = sample_input.shape[1], sample_input.shape[2]
		output_height, output_width = output.shape[1], output.shape[2]
		
		print(f"Dimension change: ({input_height}, {input_width}) → ({output_height}, {output_width})")
		
		# Store results for visualization
		results[padding] = {
			'input_shape': sample_input.shape,
			'output_shape': output.shape,
			'output': output.numpy()
		}
	
	# Visualize the effects
	plt.figure(figsize=(15, 5))
	
	# Original image
	plt.subplot(1, 3, 1)
	plt.imshow(sample_input[0].squeeze(), cmap='gray')
	plt.title(f'Original Image\n{sample_input.shape[1:3]}')
	plt.axis('off')
	
	# VALID padding result
	plt.subplot(1, 3, 2)
	plt.imshow(results['valid']['output'][0].squeeze(), cmap='gray')
	plt.title(f'VALID Padding\n{results["valid"]["output_shape"][1:3]}')
	plt.axis('off')
	
	# SAME padding result  
	plt.subplot(1, 3, 3)
	plt.imshow(results['same']['output'][0].squeeze(), cmap='gray')
	plt.title(f'SAME Padding\n{results["same"]["output_shape"][1:3]}')
	plt.axis('off')
	
	plt.suptitle('Effect of Different Padding Strategies')
	plt.tight_layout()
	plt.show()
	
	return results

# Demonstrate padding effects
padding_results = demonstrate_padding_effects()


# Step 5: Understanding Stride and Its Impact on Feature Maps
# ********************************************************************
def analyze_stride_effects():
	"""
	Comprehensive analysis of different stride values and their effects
	Understanding stride is crucial for controlling spatial downsampling
	"""
	print("\n" + "="*60)
	print("ANALYZING STRIDE EFFECTS ON FEATURE EXTRACTION")
	print("="*60)
	
	# Test different stride values
	stride_values = [1, 2, 3]
	stride_results = {}
	
	sample_input = test_images[0:1]
	
	for stride in stride_values:
		print(f"\n--- Testing stride = {stride} ---")
		
		# Create convolution layer with specified stride
		conv_layer = Conv2D(
			filters=16,
			kernel_size=(3, 3),
			strides=(stride, stride),
			padding='same',
			activation='relu'
		)
		
		# Apply convolution
		output = conv_layer(sample_input)
		
		print(f"Input shape: {sample_input.shape}")
		print(f"Output shape with stride {stride}: {output.shape}")
		
		# Calculate reduction factor
		reduction_factor = (sample_input.shape[1] * sample_input.shape[2]) / (output.shape[1] * output.shape[2])
		print(f"Spatial reduction factor: {reduction_factor:.2f}x")
		
		stride_results[stride] = {
			'output_shape': output.shape,
			'output': output.numpy(),
			'reduction_factor': reduction_factor
		}
	
	# Visualize stride effects
	plt.figure(figsize=(12, 8))
	
	# Original image
	plt.subplot(2, 3, 1)
	plt.imshow(sample_input[0].squeeze(), cmap='gray')
	plt.title(f'Original Image\n{sample_input.shape[1:3]}')
	plt.axis('off')
	
	# Show first feature map for each stride
	for i, stride in enumerate(stride_values):
		plt.subplot(2, 3, i+2)
		plt.imshow(stride_results[stride]['output'][0, :, :, 0], cmap='gray')
		plt.title(f'Stride {stride}\n{stride_results[stride]["output_shape"][1:3]}\n{stride_results[stride]["reduction_factor"]:.1f}x reduction')
		plt.axis('off')
	
	# Create a comparison chart
	plt.subplot(2, 3, 5)
	strides = list(stride_results.keys())
	reductions = [stride_results[s]['reduction_factor'] for s in strides]
	
	plt.bar(range(len(strides)), reductions, color=['blue', 'green', 'red'])
	plt.xlabel('Stride Value')
	plt.ylabel('Spatial Reduction Factor')
	plt.title('Spatial Reduction by Stride')
	plt.xticks(range(len(strides)), strides)
	
	for i, v in enumerate(reductions):
		plt.text(i, v + 0.1, f'{v:.1f}x', ha='center')
	
	plt.tight_layout()
	plt.show()
	
	return stride_results

# Analyze stride effects
stride_analysis = analyze_stride_effects()