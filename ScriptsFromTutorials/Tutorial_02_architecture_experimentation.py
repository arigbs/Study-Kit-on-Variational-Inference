# Tutorial 02: Architecture Experimentation & Model Variations
# Setting Up Our Experiment Framework
# ********************************************************************
# Uncomment any of the commented experiments 
# Import our essential libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import time

# Set random seeds for reproducible results across experiments
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up architecture experimentation framework...")


# Step 2: Loading and Preparing Our Data (Quick Setup)
# ********************************************************************
# Load and prepare MNIST data (same as Tutorial 01)
print("Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to 0-1 range
train_images = train_images / 255.0
test_images = test_images / 255.0

print(f"Training data shape: {train_images.shape}")
print(f"Test data shape: {test_images.shape}")
print("Data preparation complete!")


# Step 3: Creating Our Experiment Function
# ********************************************************************
def create_and_test_model(hidden_layers, activation='relu', optimizer='adam', learning_rate=0.001, epochs=3):
	"""
	Creates, trains, and evaluates a neural network with specified architecture
	
	Args:
		hidden_layers: List of integers specifying neurons in each hidden layer
		activation: Activation function to use ('relu', 'tanh', 'sigmoid')
		optimizer: Optimizer to use ('adam', 'sgd', 'rmsprop')
		learning_rate: Learning rate for training
		epochs: Number of training epochs
	
	Returns:
		Dictionary containing model, history, and test accuracy
	"""
	print(f"\n{'='*50}")
	print(f"Testing architecture: {hidden_layers}")
	print(f"Activation: {activation}, Optimizer: {optimizer}, LR: {learning_rate}")
	print(f"{'='*50}")
	
	# Build the model architecture
	model = Sequential()
	
	# Input layer: flatten 28x28 images to 784 features
	model.add(Flatten(input_shape=(28, 28)))
	
	# Add hidden layers based on our specification
	for i, neurons in enumerate(hidden_layers):
		print(f"Adding hidden layer {i+1}: {neurons} neurons with {activation} activation")
		model.add(Dense(neurons, activation=activation))
	
	# Output layer: 10 neurons for digits 0-9 with softmax
	model.add(Dense(10, activation='softmax'))
	
	# Configure the optimizer with specified learning rate
	if optimizer == 'adam':
		opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	elif optimizer == 'sgd':
		opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	elif optimizer == 'rmsprop':
		opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
	
	# Compile the model
	model.compile(
		optimizer=opt,
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	
	# Show model architecture
	print("\nModel Architecture:")
	model.summary()
	
	# Train the model and time the training
	print(f"\nStarting training for {epochs} epochs...")
	start_time = time.time()
	
	history = model.fit(
		train_images, 
		train_labels,
		epochs=epochs,
		validation_split=0.1,           # Use 10% of training data for validation
		verbose=1                                       # Show progress during training
	)
	
	training_time = time.time() - start_time
	print(f"Training completed in {training_time:.2f} seconds")
	
	# Evaluate on test data
	print("Evaluating on test data...")
	test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
	print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
	
	return {
		'model': model,
		'history': history,
		'test_accuracy': test_accuracy,
		'training_time': training_time,
		'architecture': hidden_layers,
		'activation': activation,
		'optimizer': optimizer,
		'learning_rate': learning_rate
	}

# Step 4: Experiment 1 - Different Layer Sizes
# ********************************************************************
print("\n" + "="*60)
print("EXPERIMENT 1: TESTING DIFFERENT LAYER SIZES")
print("="*60)

# Test different numbers of neurons in a single hidden layer
layer_size_experiments = [
	[32],           # Small layer
	[64],           # Medium-small layer  
	[128],          # Medium layer (our baseline from Tutorial 01)
	[256],          # Large layer
	[512],          # Very large layer
]

# Store results for comparison
layer_size_results = []

for layers in layer_size_experiments:
	result = create_and_test_model(
		hidden_layers=layers,
		activation='relu',
		optimizer='adam',
		learning_rate=0.001,
		epochs=3
	)
	layer_size_results.append(result)
	
	# Brief pause between experiments
	print("Waiting 2 seconds before next experiment...")
	time.sleep(2)

print("\n" + "="*60)
print("LAYER SIZE EXPERIMENT RESULTS:")
print("="*60)
for result in layer_size_results:
	print(f"Neurons: {result['architecture'][0]:3d} | "
			  f"Accuracy: {result['test_accuracy']:.4f} | "
			  f"Time: {result['training_time']:.1f}s")


# Step 5: Experiment 2 - Different Activation Functions
# ********************************************************************
print("\n" + "="*60)
print("EXPERIMENT 2: TESTING DIFFERENT ACTIVATION FUNCTIONS")
print("="*60)

# Test different activation functions with the same architecture
activation_experiments = ['relu', 'tanh', 'sigmoid']
activation_results = []

for activation in activation_experiments:
	print(f"\n--- Testing {activation.upper()} activation function ---")
	result = create_and_test_model(
		hidden_layers=[128],            # Use our baseline architecture
		activation=activation,
		optimizer='adam',
		learning_rate=0.001,
		epochs=3
	)
	activation_results.append(result)
	
	# Brief pause between experiments
	print("Waiting 2 seconds before next experiment...")
	time.sleep(2)

print("\n" + "="*60)
print("ACTIVATION FUNCTION EXPERIMENT RESULTS:")
print("="*60)
for result in activation_results:
	print(f"Activation: {result['activation']:7s} | "
			  f"Accuracy: {result['test_accuracy']:.4f} | "
			  f"Time: {result['training_time']:.1f}s")


# Step 6: Experiment 3 - Multiple Hidden Layers (Going Deeper)
# ********************************************************************
print("\n" + "="*60)
print("EXPERIMENT 3: TESTING DIFFERENT NETWORK DEPTHS")
print("="*60)

# Test different numbers of hidden layers
depth_experiments = [
	[128],                          # 1 hidden layer (baseline)
	[128, 64],                      # 2 hidden layers (decreasing size)
	[128, 64, 128],          # 3 hidden layers (hourglass shape)
	[128, 128],                     # 2 hidden layers (same size)
	[256, 128, 64],         # 3 hidden layers (pyramid shape)
]

depth_results = []

for layers in depth_experiments:
	print(f"\n--- Testing {len(layers)} hidden layer(s): {layers} ---")
	result = create_and_test_model(
		hidden_layers=layers,
		activation='relu',
		optimizer='adam',
		learning_rate=0.001,
		epochs=3
	)
	depth_results.append(result)
	
	# Brief pause between experiments
	print("Waiting 2 seconds before next experiment...")
	time.sleep(2)

print("\n" + "="*60)
print("NETWORK DEPTH EXPERIMENT RESULTS:")
print("="*60)
for result in depth_results:
	arch_str = str(result['architecture']).replace('[', '').replace(']', '')
	print(f"Architecture: {arch_str:15s} | "
			  f"Accuracy: {result['test_accuracy']:.4f} | "
			  f"Time: {result['training_time']:.1f}s")


# Step 7: Experiment 4 - Different Optimizers
# ********************************************************************
print("\n" + "="*60)
print("EXPERIMENT 4: TESTING DIFFERENT OPTIMIZERS")
print("="*60)

# Test different optimizers with the same architecture
optimizer_experiments = ['adam', 'sgd', 'rmsprop']
optimizer_results = []

for optimizer in optimizer_experiments:
	print(f"\n--- Testing {optimizer.upper()} optimizer ---")
	
	# Use different learning rates for different optimizers
	# SGD typically needs a higher learning rate than Adam
	lr = 0.01 if optimizer == 'sgd' else 0.001
	
	result = create_and_test_model(
		hidden_layers=[128],
		activation='relu',
		optimizer=optimizer,
		learning_rate=lr,
		epochs=3
	)
	optimizer_results.append(result)
	
	# Brief pause between experiments
	print("Waiting 2 seconds before next experiment...")
	time.sleep(2)

print("\n" + "="*60)
print("OPTIMIZER EXPERIMENT RESULTS:")
print("="*60)
for result in optimizer_results:
	print(f"Optimizer: {result['optimizer']:7s} | "
			  f"LR: {result['learning_rate']:.3f} | "
			  f"Accuracy: {result['test_accuracy']:.4f} | "
			  f"Time: {result['training_time']:.1f}s")
	


# Step 8: Visualizing Our Experiment Results
# ********************************************************************
# Create comprehensive comparison plots
plt.figure(figsize=(15, 10))

# Plot 1: Layer Size vs Accuracy
plt.subplot(2, 3, 1)
layer_sizes = [result['architecture'][0] for result in layer_size_results]
layer_accuracies = [result['test_accuracy'] for result in layer_size_results]
plt.plot(layer_sizes, layer_accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Neurons')
plt.ylabel('Test Accuracy')
plt.title('Layer Size vs Accuracy')
plt.grid(True, alpha=0.3)

# Plot 2: Layer Size vs Training Time
plt.subplot(2, 3, 2)
layer_times = [result['training_time'] for result in layer_size_results]
plt.plot(layer_sizes, layer_times, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Neurons')
plt.ylabel('Training Time (seconds)')
plt.title('Layer Size vs Training Time')
plt.grid(True, alpha=0.3)

# Plot 3: Activation Function Comparison
plt.subplot(2, 3, 3)
activations = [result['activation'] for result in activation_results]
activation_accuracies = [result['test_accuracy'] for result in activation_results]
bars = plt.bar(activations, activation_accuracies, color=['blue', 'green', 'red'], alpha=0.7)
plt.ylabel('Test Accuracy')
plt.title('Activation Function Comparison')
plt.grid(True, alpha=0.3)
# Add value labels on bars
for bar, acc in zip(bars, activation_accuracies):
	plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
					 f'{acc:.3f}', ha='center', va='bottom')

# Plot 4: Network Depth Comparison
plt.subplot(2, 3, 4)
depth_names = [f"{len(result['architecture'])} layers" for result in depth_results]
depth_accuracies = [result['test_accuracy'] for result in depth_results]
plt.bar(range(len(depth_names)), depth_accuracies, color='purple', alpha=0.7)
plt.xticks(range(len(depth_names)), depth_names, rotation=45)
plt.ylabel('Test Accuracy')
plt.title('Network Depth Comparison')
plt.grid(True, alpha=0.3)

# Plot 5: Optimizer Comparison
plt.subplot(2, 3, 5)
optimizer_names = [result['optimizer'] for result in optimizer_results]
optimizer_accuracies = [result['test_accuracy'] for result in optimizer_results]
bars = plt.bar(optimizer_names, optimizer_accuracies, color=['orange', 'cyan', 'magenta'], alpha=0.7)
plt.ylabel('Test Accuracy')
plt.title('Optimizer Comparison')
plt.grid(True, alpha=0.3)
# Add value labels on bars
for bar, acc in zip(bars, optimizer_accuracies):
	plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
					 f'{acc:.3f}', ha='center', va='bottom')

# Plot 6: Overall Training Time Comparison
plt.subplot(2, 3, 6)
all_experiments = ['32', '64', '128', '256', '512', 'ReLU', 'Tanh', 'Sigmoid']
all_times = (layer_times + [result['training_time'] for result in activation_results])
plt.bar(range(len(all_times)), all_times, 
		color=['blue']*5 + ['red']*3, alpha=0.7)
plt.xticks(range(len(all_experiments)), all_experiments, rotation=45)
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Step 9: Finding Your Best Architecture
# ********************************************************************
print("\n" + "="*60)
print("FINDING THE BEST ARCHITECTURE")
print("="*60)

# Combine all results for comparison
all_results = layer_size_results + activation_results + depth_results + optimizer_results

# Find the best performing model
best_result = max(all_results, key=lambda x: x['test_accuracy'])

print("🏆 BEST PERFORMING CONFIGURATION:")
print(f"Architecture: {best_result['architecture']}")
print(f"Activation: {best_result['activation']}")
print(f"Optimizer: {best_result['optimizer']}")
print(f"Learning Rate: {best_result['learning_rate']}")
print(f"Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
print(f"Training Time: {best_result['training_time']:.2f} seconds")

# Compare with our Tutorial 01 baseline
baseline_accuracy = 0.975  # Approximate expected accuracy from Tutorial 01
improvement = (best_result['test_accuracy'] - baseline_accuracy) * 100

print(f"\n📈 IMPROVEMENT ANALYSIS:")
print(f"Tutorial 01 baseline: ~97.5%")
print(f"Best experiment result: {best_result['test_accuracy']*100:.2f}%")
if improvement > 0:
	print(f"Improvement: +{improvement:.2f} percentage points! 🎉")
else:
	print(f"Change: {improvement:.2f} percentage points")

print(f"\n💡 KEY INSIGHTS FROM YOUR EXPERIMENTS:")
print("="*60)

# Analyze layer size results
best_layer_size = max(layer_size_results, key=lambda x: x['test_accuracy'])
print(f"• Optimal layer size: {best_layer_size['architecture'][0]} neurons")

# Analyze activation functions
best_activation = max(activation_results, key=lambda x: x['test_accuracy'])
print(f"• Best activation function: {best_activation['activation']}")

# Analyze depth
best_depth = max(depth_results, key=lambda x: x['test_accuracy'])
print(f"• Best architecture depth: {len(best_depth['architecture'])} layers {best_depth['architecture']}")

# Analyze optimizers
best_optimizer = max(optimizer_results, key=lambda x: x['test_accuracy'])
print(f"• Best optimizer: {best_optimizer['optimizer']}")


# Step 10: Testing Your Optimal Architecture
# ********************************************************************
print("\n" + "="*60)
print("TRAINING OPTIMAL ARCHITECTURE WITH MORE EPOCHS")
print("="*60)

print("Training your best architecture for 20 epochs to see full potential...")

final_result = create_and_test_model(
	hidden_layers=best_result['architecture'],
	activation=best_result['activation'],
	optimizer=best_result['optimizer'],
	learning_rate=best_result['learning_rate'],
	epochs=12               # Reduced for tutorial speed - use 20+ for production quality
)

print(f"\n🎯 FINAL OPTIMIZED MODEL RESULTS:")
print(f"Test Accuracy: {final_result['test_accuracy']:.4f} ({final_result['test_accuracy']*100:.2f}%)")
print(f"Training Time: {final_result['training_time']:.2f} seconds")

# Plot the training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(final_result['history'].history['accuracy'], label='Training Accuracy')
plt.plot(final_result['history'].history['val_accuracy'], label='Validation Accuracy')
plt.title('Final Model: Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(final_result['history'].history['loss'], label='Training Loss')
plt.plot(final_result['history'].history['val_loss'], label='Validation Loss')
plt.title('Final Model: Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Tutorial 02 completed successfully! 🎉")