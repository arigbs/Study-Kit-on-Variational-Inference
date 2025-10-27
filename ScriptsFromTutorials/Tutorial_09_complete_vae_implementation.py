# Tutorial 09: Complete VAE Implementation
# Step 1: Import essential libraries for complete VAE system
# ********************************************************************

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Input
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Lambda
from tensorflow.keras.callbacks import Callback
import json
import os
from datetime import datetime

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up complete VAE implementation laboratory...")
print("Today we'll build a production-ready VAE system with all advanced features!")


# Step 2: Building the Advanced VAE Architecture - Production-Ready Components
# ********************************************************************
def load_and_prepare_data(validation_split=0.1):
    """
    Load and prepare MNIST data with proper train/validation/test splits
    """
    print("Loading and preparing MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Create validation split
    val_size = int(len(train_images) * validation_split)
    val_images = train_images[:val_size]
    val_labels = train_labels[:val_size]
    train_images = train_images[val_size:]
    train_labels = train_labels[val_size:]
    
    print(f"Training set: {train_images.shape}")
    print(f"Validation set: {val_images.shape}")
    print(f"Test set: {test_images.shape}")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

class AdvancedReparameterizationLayer(tf.keras.layers.Layer):
    """
    Enhanced reparameterization layer with numerical stability
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs, training=None):
        mu, log_var = inputs
        # Clamp log_var for numerical stability
        log_var = tf.clip_by_value(log_var, -20.0, 10.0)
        
        # Sample epsilon from standard normal
        epsilon = tf.random.normal(shape=tf.shape(mu))
        
        # Reparameterization trick
        z = mu + tf.exp(0.5 * log_var) * epsilon
        
        # Note: KL loss is handled in the training step, not here
        # Removing automatic loss to prevent double-counting
        
        return z

def build_advanced_vae_encoder(latent_dim=20, architecture='deep', dropout_rate=0.2):
    """
    Build advanced VAE encoder with configurable architecture
    """
    print(f"Building advanced VAE encoder (latent_dim={latent_dim}, architecture={architecture})")
    
    encoder_input = Input(shape=(28, 28, 1))
    
    # First convolutional block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Second convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Third convolutional block (for deep architecture)
    if architecture == 'deep':
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Dense layers for latent space
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output distribution parameters
    mu = Dense(latent_dim, name='mu')(x)
    log_var = Dense(latent_dim, name='log_var')(x)
    
    # Reparameterization
    z = AdvancedReparameterizationLayer()([mu, log_var])
    
    encoder = Model(encoder_input, [mu, log_var, z], name='advanced_encoder')
    return encoder

def build_advanced_vae_decoder(latent_dim=20, architecture='deep', dropout_rate=0.2):
    """
    Build advanced VAE decoder with configurable architecture
    """
    print(f"Building advanced VAE decoder (latent_dim={latent_dim}, architecture={architecture})")
    
    decoder_input = Input(shape=(latent_dim,))
    
    # Dense layers to expand latent space
    x = Dense(256, activation='relu')(decoder_input)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Determine reshape size based on architecture
    if architecture == 'deep':
        reshape_size = 4 * 4 * 128
        x = Dense(reshape_size, activation='relu')(x)
        x = Reshape((4, 4, 128))(x)
        
        # Upsampling blocks
        x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    else:  # shallow architecture
        reshape_size = 7 * 7 * 64
        x = Dense(reshape_size, activation='relu')(x)
        x = Reshape((7, 7, 64))(x)
        
        # Upsampling blocks
        x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
    
    # Output layer
    decoder_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Ensure output is exactly 28x28 using proper Keras layers
    # For deep architecture: 32x32 -> 28x28 (crop 2 pixels from each side)
    # For shallow architecture: 28x28 -> 28x28 (no cropping needed)
    if architecture == 'deep':
        # Deep architecture outputs 32x32, crop to 28x28
        decoder_output = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)), name='crop_to_28x28')(decoder_output)
    # Shallow architecture should already output 28x28, no cropping needed
    
    decoder = Model(decoder_input, decoder_output, name='advanced_decoder')
    return decoder

# Load data and build advanced architecture
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_and_prepare_data()

# Build models with configurable parameters
latent_dim = 20
architecture = 'deep'  # 'shallow' or 'deep'
dropout_rate = 0.2

advanced_encoder = build_advanced_vae_encoder(latent_dim, architecture, dropout_rate)
advanced_decoder = build_advanced_vae_decoder(latent_dim, architecture, dropout_rate)

# Display model summaries
print("\n" + "="*50)
print("ADVANCED VAE ENCODER SUMMARY")
print("="*50)
advanced_encoder.summary()

print("\n" + "="*50)
print("ADVANCED VAE DECODER SUMMARY")
print("="*50)
advanced_decoder.summary()

print(f"\nAdvanced VAE architecture built successfully!")
print(f"Configuration: {architecture} architecture, {latent_dim}D latent space, {dropout_rate} dropout")


# Step 3: Building the Advanced VAE Architecture - Production-Ready Components
# ********************************************************************
class BetaScheduler(Callback):
    """
    Advanced beta scheduling for β-VAE training with multiple schedule types
    """
    def __init__(self, schedule_type='linear_warmup', initial_beta=0.01, final_beta=0.1, warmup_epochs=4):
        super().__init__()
        self.schedule_type = schedule_type
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.warmup_epochs = warmup_epochs
        self.current_beta = initial_beta
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.schedule_type == 'linear_warmup':
            if epoch < self.warmup_epochs:
                self.current_beta = self.initial_beta + (self.final_beta - self.initial_beta) * (epoch / self.warmup_epochs)
            else:
                self.current_beta = self.final_beta
                
        elif self.schedule_type == 'cosine_annealing':
            self.current_beta = self.initial_beta + (self.final_beta - self.initial_beta) * \
                               (1 - np.cos(np.pi * min(epoch / self.warmup_epochs, 1.0))) / 2
                               
        elif self.schedule_type == 'exponential':
            decay_rate = np.log(self.final_beta / self.initial_beta) / self.warmup_epochs
            self.current_beta = self.initial_beta * np.exp(decay_rate * min(epoch, self.warmup_epochs))
            
        else:  # constant
            self.current_beta = self.final_beta
            
        # Update model's beta value
        if hasattr(self.model, 'beta'):
            self.model.beta = self.current_beta
            
        print(f"Epoch {epoch}: Beta = {self.current_beta:.4f}")
    
    def get_beta(self):
        return self.current_beta

class AdvancedVAETrainer(Model):
    """
    Advanced VAE trainer with beta scheduling, comprehensive monitoring, and production features
    """
    def __init__(self, encoder, decoder, latent_dim, beta_scheduler=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta_scheduler = beta_scheduler
        self.beta = 0.1  # Default beta value (optimal for reconstruction quality)
        
        # Comprehensive metrics tracking
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.beta_tracker = tf.keras.metrics.Mean(name="beta")
        
        # Additional metrics for monitoring
        self.mu_mean_tracker = tf.keras.metrics.Mean(name="mu_mean")
        self.mu_std_tracker = tf.keras.metrics.Mean(name="mu_std")
        self.logvar_mean_tracker = tf.keras.metrics.Mean(name="logvar_mean")
        
    def call(self, inputs, training=None):
        """Forward pass through complete VAE"""
        mu, log_var, z = self.encoder(inputs, training=training)
        reconstructed = self.decoder(z, training=training)
        return reconstructed
        
    def train_step(self, data):
        """Advanced training step with beta scheduling and comprehensive monitoring"""
        # Handle data input properly - extract just the input data
        if isinstance(data, tuple):
            # If data is (x, y) tuple, use only x for VAE
            input_data = data[0]
        else:
            # If data is just x, use as is
            input_data = data
            
        with tf.GradientTape() as tape:
            # Forward pass
            mu, log_var, z = self.encoder(input_data, training=True)
            reconstructed = self.decoder(z, training=True)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(input_data, reconstructed)
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            )
            
            # Get current beta from scheduler
            current_beta = self.beta_scheduler.get_beta() if self.beta_scheduler else self.beta
            
            # Total VAE loss
            total_loss = reconstruction_loss + current_beta * kl_loss
            
            # Add any additional losses from layers
            if self.losses:
                total_loss += tf.add_n(self.losses)
        
        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_weights)
        
        # Gradient clipping for stability
        gradients = [tf.clip_by_norm(grad, 1.0) for grad in gradients]
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.beta_tracker.update_state(current_beta)
        
        # Update distribution statistics
        self.mu_mean_tracker.update_state(tf.reduce_mean(mu))
        self.mu_std_tracker.update_state(tf.math.reduce_std(mu))
        self.logvar_mean_tracker.update_state(tf.reduce_mean(log_var))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "beta": current_beta,
            "mu_mean": tf.reduce_mean(mu),
            "mu_std": tf.math.reduce_std(mu),
            "logvar_mean": tf.reduce_mean(log_var)
        }
    
    def test_step(self, data):
        """Validation step with same metrics"""
        # Handle data input properly - extract just the input data
        if isinstance(data, tuple):
            # If data is (x, y) tuple, use only x for VAE
            input_data = data[0]
        else:
            # If data is just x, use as is
            input_data = data
            
        mu, log_var, z = self.encoder(input_data, training=False)
        reconstructed = self.decoder(z, training=False)
        
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(input_data, reconstructed)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        
        current_beta = self.beta_scheduler.get_beta() if self.beta_scheduler else self.beta
        total_loss = reconstruction_loss + current_beta * kl_loss
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "beta": current_beta,
            "mu_mean": tf.reduce_mean(mu),
            "mu_std": tf.math.reduce_std(mu),
            "logvar_mean": tf.reduce_mean(log_var)
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.beta_tracker,
            self.mu_mean_tracker,
            self.mu_std_tracker,
            self.logvar_mean_tracker
        ]
    
    def generate_samples(self, num_samples=16):
        """Generate new samples from the trained VAE"""
        random_latent = tf.random.normal(shape=(num_samples, self.latent_dim))
        generated_images = self.decoder(random_latent, training=False)
        return generated_images

# Create advanced VAE trainer with beta scheduling
print("Creating advanced VAE training system...")

# Configure beta scheduler with optimal beta value from Tutorial 8
# Start with small non-zero beta to prevent latent collapse
beta_scheduler = BetaScheduler(
    schedule_type='linear_warmup',
    initial_beta=0.01,  # Start with small non-zero value to prevent latent collapse
    final_beta=0.1,    # Use optimal value from Tutorial 8 - prevents KL domination
    warmup_epochs=4    # Faster warmup to reach optimal beta sooner
)

# Create advanced VAE trainer
advanced_vae = AdvancedVAETrainer(
    encoder=advanced_encoder,
    decoder=advanced_decoder,
    latent_dim=latent_dim,
    beta_scheduler=beta_scheduler
)

# Compile with advanced optimizer settings
advanced_vae.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
)

print("Advanced VAE training system created successfully!")
print(f"Beta scheduler: {beta_scheduler.schedule_type}")
print(f"Initial beta: {beta_scheduler.initial_beta}, Final beta: {beta_scheduler.final_beta} (optimal for reconstruction quality)")
print(f"Warmup epochs: {beta_scheduler.warmup_epochs} (aligned with training duration for optimal beta reach)")


# Step 4: Training the Complete VAE with Comprehensive Monitoring - Real-Time Analysis
# ********************************************************************
# Create comprehensive monitoring callback
# Create comprehensive monitoring callback
class VAEMonitoringCallback(tf.keras.callbacks.Callback):
    """
    Advanced monitoring callback for VAE training
    """
    def __init__(self, test_data, test_labels, sample_freq=5, save_path='vae_monitoring'):
        super().__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.sample_freq = sample_freq
        self.save_path = save_path
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize tracking lists
        self.epoch_metrics = []
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.sample_freq == 0:
            # Generate samples
            generated_images = self.model.generate_samples(16)
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(4, 6, figsize=(15, 10))
            
            # Plot generated samples
            for i in range(16):
                row, col = i // 4, i % 4
                axes[row, col].imshow(generated_images[i].numpy().squeeze(), cmap='gray')
                axes[row, col].set_title(f'Generated {i+1}')
                axes[row, col].axis('off')
            
            # Plot loss evolution
            if len(self.epoch_metrics) > 0:
                metrics_df = pd.DataFrame(self.epoch_metrics)
                
                # Reconstruction loss
                axes[0, 4].plot(metrics_df['reconstruction_loss'], label='Train')
                axes[0, 4].plot(metrics_df['val_reconstruction_loss'], label='Val')
                axes[0, 4].set_title('Reconstruction Loss')
                axes[0, 4].legend()
                axes[0, 4].grid(True)
                
                # KL loss
                axes[1, 4].plot(metrics_df['kl_loss'], label='Train')
                axes[1, 4].plot(metrics_df['val_kl_loss'], label='Val')
                axes[1, 4].set_title('KL Loss')
                axes[1, 4].legend()
                axes[1, 4].grid(True)
                
                # Beta evolution
                axes[2, 4].plot(metrics_df['beta'])
                axes[2, 4].set_title('Beta Schedule')
                axes[2, 4].grid(True)
                
                # Latent space statistics
                axes[3, 4].plot(metrics_df['mu_mean'], label='μ mean')
                axes[3, 4].plot(metrics_df['mu_std'], label='μ std')
                axes[3, 4].plot(metrics_df['logvar_mean'], label='log_var mean')
                axes[3, 4].set_title('Latent Statistics')
                axes[3, 4].legend()
                axes[3, 4].grid(True)
            
            # Test reconstruction quality
            test_sample = self.test_data[:4]
            mu, log_var, z = self.model.encoder(test_sample)
            reconstructed = self.model.decoder(z)
            
            for i in range(4):
                # Original
                axes[i, 5].imshow(test_sample[i].squeeze(), cmap='gray')
                axes[i, 5].set_title(f'Original {i+1}')
                axes[i, 5].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.save_path}/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Store metrics
            current_metrics = logs.copy()
            current_metrics['epoch'] = epoch
            self.epoch_metrics.append(current_metrics)
            
            # Print detailed status
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Total Loss: {logs['loss']:.4f} | Val Loss: {logs['val_loss']:.4f}")
            print(f"  Reconstruction: {logs['reconstruction_loss']:.4f} | Val Reconstruction: {logs['val_reconstruction_loss']:.4f}")
            print(f"  KL Loss: {logs['kl_loss']:.4f} | Val KL: {logs['val_kl_loss']:.4f}")
            print(f"  Beta: {logs['beta']:.4f}")
            print(f"  Latent μ: {logs['mu_mean']:.4f} ± {logs['mu_std']:.4f}")
            print(f"  Latent log_var: {logs['logvar_mean']:.4f}")

# Setup comprehensive callbacks
print("Setting up comprehensive training callbacks...")

# Create monitoring callback
monitoring_callback = VAEMonitoringCallback(
    test_data=test_images,
    test_labels=test_labels,
    sample_freq=5,
    save_path='vae_training_monitoring'
)

# Create model checkpoint directory
os.makedirs('complete_vae_checkpoints', exist_ok=True)

# Setup all callbacks
callbacks = [
    beta_scheduler,
    monitoring_callback,
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='complete_vae_checkpoints/best_model.weights.h5',
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.8,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        'complete_vae_training.log',
        separator=',',
        append=False
    )
]

# Build the model before training to enable weight saving
print("Building model architecture...")
# Build the model by calling it on sample data
sample_batch = train_images[:1]  # Use first training sample
_ = advanced_vae(sample_batch, training=False)  # Forward pass to build the model
print("✅ Model built successfully!")

# Start comprehensive training
print("\n" + "="*60)
print("STARTING COMPREHENSIVE VAE TRAINING")
print("="*60)
print(f"Training samples: {len(train_images)}")
print(f"Validation samples: {len(val_images)}")
print(f"Test samples: {len(test_images)}")
print(f"Latent dimensions: {latent_dim}")
print(f"Architecture: {architecture}")
print(f"Dropout rate: {dropout_rate}")
print("="*60)

# Train the model (reduced epochs for tutorial evaluation - increase to 50+ for production quality)
history = advanced_vae.fit(
    train_images,
    epochs=12,  # Reduced for tutorial speed - use 50+ epochs for higher quality results
    batch_size=64,
    validation_data=(val_images, val_images),
    callbacks=callbacks,
    verbose=1
)

# Post-training analysis
print("\n" + "="*60)
print("TRAINING COMPLETED - ANALYSIS")
print("="*60)

# Plot comprehensive training history
def plot_comprehensive_training_history(history):
    """Create comprehensive training history visualization"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Total loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Total Loss Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(history.history['reconstruction_loss'], label='Training Reconstruction')
    axes[0, 1].plot(history.history['val_reconstruction_loss'], label='Validation Reconstruction')
    axes[0, 1].set_title('Reconstruction Loss Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL loss
    axes[1, 0].plot(history.history['kl_loss'], label='Training KL')
    axes[1, 0].plot(history.history['val_kl_loss'], label='Validation KL')
    axes[1, 0].set_title('KL Divergence Loss Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Beta schedule
    axes[1, 1].plot(history.history['beta'])
    axes[1, 1].set_title('Beta Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta Value')
    axes[1, 1].grid(True)
    
    # Latent space statistics
    axes[2, 0].plot(history.history['mu_mean'], label='μ Mean')
    axes[2, 0].plot(history.history['mu_std'], label='μ Std')
    axes[2, 0].set_title('Latent Space μ Statistics')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Value')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # Log variance statistics
    axes[2, 1].plot(history.history['logvar_mean'], label='Log Var Mean')
    axes[2, 1].set_title('Latent Space Log Variance')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Log Variance')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('vae_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_comprehensive_training_history(history)

# Generate and display final samples
print("\nGenerating final samples...")
final_samples = advanced_vae.generate_samples(25)

plt.figure(figsize=(12, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(final_samples[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle('Final Generated Samples', fontsize=16)
plt.tight_layout()
plt.savefig('final_generated_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print("Comprehensive VAE training completed successfully!")
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final reconstruction loss: {history.history['reconstruction_loss'][-1]:.4f}")
print(f"Final KL loss: {history.history['kl_loss'][-1]:.4f}")
print(f"Final beta: {history.history['beta'][-1]:.4f}")


# Step 5: Comprehensive Evaluation and Benchmarking - Quantitative Assessment
# ********************************************************************
# Additional imports for comprehensive evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import entropy
import seaborn as sns

class VAEEvaluator:
    """
    Comprehensive VAE evaluation system with quantitative metrics
    """
    def __init__(self, vae_model, test_data, test_labels):
        self.vae_model = vae_model
        self.test_data = test_data
        self.test_labels = test_labels
        self.evaluation_results = {}
        
    def evaluate_reconstruction_quality(self):
        """
        Evaluate reconstruction quality using multiple metrics
        """
        print("Evaluating reconstruction quality...")
        
        # Get reconstructions
        mu, log_var, z = self.vae_model.encoder(self.test_data)
        reconstructed = self.vae_model.decoder(z)
        
        # Calculate metrics
        mse = tf.reduce_mean(tf.square(self.test_data - reconstructed)).numpy()
        mae = tf.reduce_mean(tf.abs(self.test_data - reconstructed)).numpy()
        
        # SSIM calculation (simplified version)
        def ssim_metric(original, reconstructed):
            # Simplified SSIM calculation
            original_mean = tf.reduce_mean(original)
            reconstructed_mean = tf.reduce_mean(reconstructed)
            
            original_var = tf.reduce_mean(tf.square(original - original_mean))
            reconstructed_var = tf.reduce_mean(tf.square(reconstructed - reconstructed_mean))
            
            covar = tf.reduce_mean((original - original_mean) * (reconstructed - reconstructed_mean))
            
            c1, c2 = 0.01, 0.03
            ssim = ((2 * original_mean * reconstructed_mean + c1) * (2 * covar + c2)) / \
                   ((original_mean**2 + reconstructed_mean**2 + c1) * (original_var + reconstructed_var + c2))
            
            return ssim.numpy()
        
        # Calculate SSIM for sample
        ssim_sample = ssim_metric(self.test_data[:100], reconstructed[:100])
        
        # Binary cross-entropy
        bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.test_data, reconstructed)).numpy()
        
        reconstruction_metrics = {
            'mse': mse,
            'mae': mae,
            'ssim': ssim_sample,
            'binary_crossentropy': bce
        }
        
        # Visualize reconstruction quality
        fig, axes = plt.subplots(3, 6, figsize=(18, 9))
        
        # Show original vs reconstructed
        for i in range(6):
            axes[0, i].imshow(self.test_data[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(reconstructed[i].numpy().squeeze(), cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
            
            # Difference
            diff = np.abs(self.test_data[i].squeeze() - reconstructed[i].numpy().squeeze())
            axes[2, i].imshow(diff, cmap='hot')
            axes[2, i].set_title(f'Difference {i+1}')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('reconstruction_quality_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.evaluation_results['reconstruction'] = reconstruction_metrics
        return reconstruction_metrics
    
    def evaluate_latent_space_quality(self):
        """
        Evaluate latent space organization and structure
        """
        print("Evaluating latent space quality...")
        
        # Encode all test data
        mu, log_var, z = self.vae_model.encoder(self.test_data)
        
        # Calculate KL divergence
        kl_divergence = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        ).numpy()
        
        # Latent space statistics
        latent_stats = {
            'kl_divergence': kl_divergence,
            'mu_mean': tf.reduce_mean(mu).numpy(),
            'mu_std': tf.math.reduce_std(mu).numpy(),
            'logvar_mean': tf.reduce_mean(log_var).numpy(),
            'logvar_std': tf.math.reduce_std(log_var).numpy()
        }
        
        # Evaluate digit separation in latent space
        latent_2d = PCA(n_components=2).fit_transform(z.numpy())
        
        # Calculate silhouette score for digit separation
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(latent_2d, self.test_labels)
        latent_stats['silhouette_score'] = silhouette
        
        # Visualize latent space
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA visualization
        scatter = axes[0, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                    c=self.test_labels, cmap='tab10', alpha=0.6)
        axes[0, 0].set_title('Latent Space (PCA)')
        axes[0, 0].set_xlabel('PCA Component 1')
        axes[0, 0].set_ylabel('PCA Component 2')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # t-SNE visualization
        if len(z) > 1000:
            tsne_subset = z[:1000]
            labels_subset = self.test_labels[:1000]
        else:
            tsne_subset = z
            labels_subset = self.test_labels
            
        tsne_2d = TSNE(n_components=2, random_state=42).fit_transform(tsne_subset.numpy())
        scatter = axes[0, 1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], 
                                    c=labels_subset, cmap='tab10', alpha=0.6)
        axes[0, 1].set_title('Latent Space (t-SNE)')
        axes[0, 1].set_xlabel('t-SNE Component 1')
        axes[0, 1].set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Latent dimension histograms
        axes[1, 0].hist(mu.numpy().flatten(), bins=50, alpha=0.7, label='μ')
        axes[1, 0].hist(np.exp(0.5 * log_var.numpy()).flatten(), bins=50, alpha=0.7, label='σ')
        axes[1, 0].set_title('Latent Parameter Distributions')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # KL divergence per dimension
        kl_per_dim = 0.5 * tf.reduce_mean(tf.square(mu) + tf.exp(log_var) - 1 - log_var, axis=0)
        axes[1, 1].bar(range(len(kl_per_dim)), kl_per_dim.numpy())
        axes[1, 1].set_title('KL Divergence per Latent Dimension')
        axes[1, 1].set_xlabel('Latent Dimension')
        axes[1, 1].set_ylabel('KL Divergence')
        
        plt.tight_layout()
        plt.savefig('latent_space_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.evaluation_results['latent_space'] = latent_stats
        return latent_stats
    
    def evaluate_generation_quality(self, num_samples=1000):
        """
        Evaluate generation quality and diversity
        """
        print("Evaluating generation quality...")
        
        # Generate samples
        random_latent = tf.random.normal(shape=(num_samples, self.vae_model.latent_dim))
        generated_images = self.vae_model.decoder(random_latent)
        
        # Calculate inception score approximation
        # (simplified version for MNIST)
        def calculate_inception_score(images, splits=10):
            # Use a simple CNN classifier as proxy for inception
            scores = []
            split_size = len(images) // splits
            
            for i in range(splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size
                split_images = images[start_idx:end_idx]
                
                # Calculate entropy
                pixel_mean = tf.reduce_mean(split_images, axis=0)
                kl_div = tf.reduce_sum(pixel_mean * tf.math.log(pixel_mean + 1e-10))
                scores.append(kl_div.numpy())
            
            return np.mean(scores), np.std(scores)
        
        is_mean, is_std = calculate_inception_score(generated_images)
        
        # Calculate FID approximation
        # Compare statistics between real and generated data
        real_mean = tf.reduce_mean(self.test_data, axis=0)
        real_var = tf.reduce_mean(tf.square(self.test_data - real_mean), axis=0)
        
        generated_mean = tf.reduce_mean(generated_images, axis=0)
        generated_var = tf.reduce_mean(tf.square(generated_images - generated_mean), axis=0)
        
        # Simplified FID calculation
        fid_score = tf.reduce_mean(tf.square(real_mean - generated_mean)) + \
                   tf.reduce_mean(tf.square(real_var - generated_var))
        
        generation_metrics = {
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'fid_score': fid_score.numpy(),
            'num_samples': num_samples
        }
        
        # Visualize generation quality
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        
        for i in range(25):
            row, col = i // 5, i % 5
            axes[row, col].imshow(generated_images[i].numpy().squeeze(), cmap='gray')
            axes[row, col].axis('off')
        
        plt.suptitle('Generated Samples Quality Assessment', fontsize=16)
        plt.tight_layout()
        plt.savefig('generation_quality_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        self.evaluation_results['generation'] = generation_metrics
        return generation_metrics
    
    def evaluate_interpolation_smoothness(self):
        """
        Evaluate interpolation quality in latent space
        """
        print("Evaluating interpolation smoothness...")
        
        # Select two random test samples
        idx1, idx2 = np.random.choice(len(self.test_data), 2, replace=False)
        img1, img2 = self.test_data[idx1], self.test_data[idx2]
        
        # Encode to latent space
        mu1, _, z1 = self.vae_model.encoder(img1[tf.newaxis, ...])
        mu2, _, z2 = self.vae_model.encoder(img2[tf.newaxis, ...])
        
        # Create interpolation
        alphas = np.linspace(0, 1, 10)
        interpolated_images = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            img_interp = self.vae_model.decoder(z_interp)
            interpolated_images.append(img_interp[0])
        
        # Calculate smoothness metric
        smoothness_scores = []
        for i in range(len(interpolated_images) - 1):
            diff = tf.reduce_mean(tf.square(interpolated_images[i] - interpolated_images[i+1]))
            smoothness_scores.append(diff.numpy())
        
        smoothness_metric = np.mean(smoothness_scores)
        
        # Visualize interpolation
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img.numpy().squeeze(), cmap='gray')
            axes[i].set_title(f'α={alphas[i]:.1f}')
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation', fontsize=16)
        plt.tight_layout()
        plt.savefig('interpolation_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        interpolation_metrics = {
            'smoothness_score': smoothness_metric,
            'num_interpolation_steps': len(alphas)
        }
        
        self.evaluation_results['interpolation'] = interpolation_metrics
        return interpolation_metrics
    
    def run_complete_evaluation(self):
        """
        Run comprehensive evaluation of all VAE aspects
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE VAE EVALUATION")
        print("="*60)
        
        # Run all evaluations
        reconstruction_metrics = self.evaluate_reconstruction_quality()
        latent_metrics = self.evaluate_latent_space_quality()
        generation_metrics = self.evaluate_generation_quality()
        interpolation_metrics = self.evaluate_interpolation_smoothness()
        
        # Create comprehensive report
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\nReconstruction Quality:")
        for key, value in reconstruction_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nLatent Space Quality:")
        for key, value in latent_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nGeneration Quality:")
        for key, value in generation_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nInterpolation Quality:")
        for key, value in interpolation_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save comprehensive report
        report_df = pd.DataFrame([
            {**reconstruction_metrics, **latent_metrics, 
             **generation_metrics, **interpolation_metrics}
        ])
        
        report_df.to_csv('vae_evaluation_report.csv', index=False)
        print("\nEvaluation report saved to 'vae_evaluation_report.csv'")
        
        return self.evaluation_results

# Run comprehensive evaluation
print("Starting comprehensive VAE evaluation...")
evaluator = VAEEvaluator(advanced_vae, test_images, test_labels)
evaluation_results = evaluator.run_complete_evaluation()

print("\nComprehensive evaluation completed!")
print("All evaluation results have been saved and visualized.")

# Step 6: Model Persistence and Deployment - Production-Ready Packaging
# ********************************************************************
import pickle
import json
from datetime import datetime

class VAEModelManager:
    """
    Comprehensive VAE model management for production deployment
    """
    def __init__(self, vae_model, encoder, decoder, config=None):
        self.vae_model = vae_model
        self.encoder = encoder
        self.decoder = decoder
        self.config = config or self._create_default_config()
        
    def _create_default_config(self):
        """Create default configuration dictionary"""
        # Get latent dimension from VAE trainer if available, otherwise use default
        latent_dim = getattr(self.vae_model, 'latent_dim', 20)  # Default to 20 if not found
        
        return {
            'latent_dim': latent_dim,
            'architecture': 'deep',
            'dropout_rate': 0.2,
            'beta_schedule': 'linear_warmup',
            'training_epochs': 50,
            'batch_size': 64,
            'learning_rate': 0.001,
            'created_timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'model_type': 'VAE'
        }
    
    def save_complete_model(self, save_path='complete_vae_model'):
        """
        Save complete VAE model with all components and metadata
        """
        print(f"Saving complete VAE model to '{save_path}'...")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        print("  Saving model weights...")
        self.vae_model.save_weights(os.path.join(save_path, 'vae_weights.weights.h5'))
        self.encoder.save_weights(os.path.join(save_path, 'encoder_weights.weights.h5'))
        self.decoder.save_weights(os.path.join(save_path, 'decoder_weights.weights.h5'))
        
        # Save model architectures
        print("  Saving model architectures...")
        with open(os.path.join(save_path, 'vae_architecture.json'), 'w') as f:
            f.write(self.vae_model.to_json())
        
        with open(os.path.join(save_path, 'encoder_architecture.json'), 'w') as f:
            f.write(self.encoder.to_json())
            
        with open(os.path.join(save_path, 'decoder_architecture.json'), 'w') as f:
            f.write(self.decoder.to_json())
        
        # Save configuration
        print("  Saving configuration...")
        # Filter out non-JSON serializable objects
        serializable_config = {}
        for key, value in self.config.items():
            try:
                # Test if the value is JSON serializable
                json.dumps(value)
                serializable_config[key] = value
            except (TypeError, ValueError):
                # Skip non-serializable objects
                print(f"    Skipping non-serializable config key: {key}")
                continue
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        # Save optimizer state
        print("  Saving optimizer state...")
        try:
            optimizer_weights = self.vae_model.optimizer.get_weights()
            with open(os.path.join(save_path, 'optimizer_weights.pkl'), 'wb') as f:
                pickle.dump(optimizer_weights, f)
        except Exception as e:
            print(f"    Warning: Could not save optimizer state: {e}")
        
        # Create model summary
        print("  Creating model summary...")
        summary_info = {
            'model_type': 'Variational Autoencoder',
            'total_parameters': self.vae_model.count_params(),
            'encoder_parameters': self.encoder.count_params(),
            'decoder_parameters': self.decoder.count_params(),
            'latent_dimensions': self.config.get('latent_dim', 'unknown'),
            'input_shape': list(self.encoder.input_shape[1:]),
            'output_shape': list(self.decoder.output_shape[1:]),
            'save_timestamp': datetime.now().isoformat(),
            'config': serializable_config  # Use the filtered config
        }
        
        with open(os.path.join(save_path, 'model_summary.json'), 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        # Save model performance metrics (if available)
        if hasattr(self, 'evaluation_results'):
            print("  Saving evaluation results...")
            # Filter evaluation results for JSON serialization
            serializable_results = {}
            for key, value in self.evaluation_results.items():
                try:
                    # Test if the value is JSON serializable
                    json.dumps(value)
                    serializable_results[key] = value
                except (TypeError, ValueError):
                    print(f"    Skipping non-serializable evaluation result: {key}")
                    continue
            
            with open(os.path.join(save_path, 'evaluation_results.json'), 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        print(f"Complete VAE model saved successfully to '{save_path}'!")
        return save_path
    
    @staticmethod
    def load_complete_model(load_path='complete_vae_model'):
        """
        Load complete VAE model from saved components
        """
        print(f"Loading complete VAE model from '{load_path}'...")
        
        # Load configuration
        print("  Loading configuration...")
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load model architectures
        print("  Loading model architectures...")
        with open(os.path.join(load_path, 'encoder_architecture.json'), 'r') as f:
            encoder = tf.keras.models.model_from_json(f.read(), 
                                                     custom_objects={'AdvancedReparameterizationLayer': AdvancedReparameterizationLayer})
        
        with open(os.path.join(load_path, 'decoder_architecture.json'), 'r') as f:
            decoder = tf.keras.models.model_from_json(f.read())
        
        # Recreate VAE trainer
        print("  Recreating VAE trainer...")
        beta_scheduler = BetaScheduler(
            schedule_type=config.get('beta_schedule', 'linear_warmup'),
            warmup_epochs=8  # Aligned with tutorial training duration
        )
        
        vae_model = AdvancedVAETrainer(
            encoder=encoder,
            decoder=decoder,
            latent_dim=config.get('latent_dim', 20),  # Default to 20 if not found
            beta_scheduler=beta_scheduler
        )
        
        # Compile model
        vae_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.get('learning_rate', 0.001)
            )
        )
        
        # Load weights
        print("  Loading weights...")
        encoder.load_weights(os.path.join(load_path, 'encoder_weights.weights.h5'))
        decoder.load_weights(os.path.join(load_path, 'decoder_weights.weights.h5'))
        
        # Try to load VAE weights (might fail if architecture changed)
        try:
            vae_model.load_weights(os.path.join(load_path, 'vae_weights.weights.h5'))
        except Exception as e:
            print(f"    Warning: Could not load VAE weights: {e}")
            print("    Individual component weights loaded successfully.")
        
        # Try to load optimizer state
        try:
            with open(os.path.join(load_path, 'optimizer_weights.pkl'), 'rb') as f:
                optimizer_weights = pickle.load(f)
            vae_model.optimizer.set_weights(optimizer_weights)
            print("    Optimizer state loaded successfully.")
        except Exception as e:
            print(f"    Warning: Could not load optimizer state: {e}")
        
        # Create model manager
        model_manager = VAEModelManager(vae_model, encoder, decoder, config)
        
        # Load evaluation results if available
        try:
            with open(os.path.join(load_path, 'evaluation_results.json'), 'r') as f:
                model_manager.evaluation_results = json.load(f)
            print("    Evaluation results loaded successfully.")
        except:
            pass
        
        print(f"Complete VAE model loaded successfully from '{load_path}'!")
        return model_manager
    
    def export_for_inference(self, export_path='vae_inference_model'):
        """
        Export VAE components for optimized inference deployment
        """
        print(f"Exporting VAE for inference to '{export_path}'...")
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Create inference-optimized encoder
        print("  Exporting encoder for inference...")
        try:
            # Try to get mu layer if it exists
            mu_output = self.encoder.get_layer('mu').output
            inference_encoder = tf.keras.Model(
                inputs=self.encoder.input,
                outputs=mu_output,  # Only output mean for inference
                name='inference_encoder'
            )
        except ValueError:
            # If mu layer doesn't exist, use first output (mu) from encoder
            inference_encoder = tf.keras.Model(
                inputs=self.encoder.input,
                outputs=self.encoder.output[0],  # First output is typically mu
                name='inference_encoder'
            )
        
        inference_encoder.save(os.path.join(export_path, 'encoder.keras'))
        
        # Export decoder
        print("  Exporting decoder for inference...")
        self.decoder.save(os.path.join(export_path, 'decoder.keras'))
        
        # Create inference configuration
        inference_config = {
            'latent_dim': self.config['latent_dim'],
            'input_shape': list(self.encoder.input_shape[1:]),
            'output_shape': list(self.decoder.output_shape[1:]),
            'model_type': 'VAE_Inference',
            'export_timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__
        }
        
        with open(os.path.join(export_path, 'inference_config.json'), 'w') as f:
            json.dump(inference_config, f, indent=2)
        
        # Create inference example script
        inference_script = f'''
import tensorflow as tf
import numpy as np
import json

# Load inference configuration
with open('inference_config.json', 'r') as f:
    config = json.load(f)

# Load models
encoder = tf.keras.models.load_model('encoder.keras')
decoder = tf.keras.models.load_model('decoder.keras')

# Example inference functions
def encode_image(image):
    """Encode image to latent space (mean only)"""
    return encoder(image)

def decode_latent(latent_code):
    """Decode latent code to image"""
    return decoder(latent_code)

def generate_sample():
    """Generate random sample"""
    latent_dim = config['latent_dim']
    random_latent = tf.random.normal(shape=(1, latent_dim))
    return decode_latent(random_latent)

def interpolate(image1, image2, steps=10):
    """Interpolate between two images"""
    z1 = encode_image(image1)
    z2 = encode_image(image2)
    
    interpolated = []
    for alpha in np.linspace(0, 1, steps):
        z_interp = (1 - alpha) * z1 + alpha * z2
        img_interp = decode_latent(z_interp)
        interpolated.append(img_interp)
    
    return interpolated

print("VAE inference models loaded successfully!")
print(f"Latent dimensions: {{config['latent_dim']}}")
print(f"Input shape: {{config['input_shape']}}")
print(f"Output shape: {{config['output_shape']}}")
'''
        
        with open(os.path.join(export_path, 'inference_example.py'), 'w') as f:
            f.write(inference_script)
        
        print(f"VAE exported for inference to '{export_path}'!")
        print("  Included:")
        print("    - encoder.keras (Keras model file)")
        print("    - decoder.keras (Keras model file)")
        print("    - inference_config.json")
        print("    - inference_example.py")
        
        return export_path
    
    def create_deployment_package(self, package_path='vae_deployment_package'):
        """
        Create complete deployment package with documentation
        """
        print(f"Creating deployment package at '{package_path}'...")
        
        # Save complete model
        model_path = os.path.join(package_path, 'model')
        self.save_complete_model(model_path)
        
        # Export for inference
        inference_path = os.path.join(package_path, 'inference')
        self.export_for_inference(inference_path)
        
        # Create README
        readme_content = f'''
# VAE Deployment Package

This package contains a complete Variational Autoencoder (VAE) deployment.

## Contents

### /model/
Complete model with all components and training state:
- Model weights and architectures
- Training configuration
- Optimizer state
- Evaluation results

### /inference/
Optimized models for production inference:
- encoder.keras (Keras model file)
- decoder.keras (Keras model file)
- Inference configuration
- Example usage script

## Quick Start

### Loading Complete Model
```python
from vae_model_manager import VAEModelManager
model_manager = VAEModelManager.load_complete_model('model/')
```

### Inference Only
```python
import tensorflow as tf
encoder = tf.keras.models.load_model('inference/encoder.keras')
decoder = tf.keras.models.load_model('inference/decoder.keras')
```

## Model Specifications
- Latent Dimensions: {self.config['latent_dim']}
- Architecture: {self.config['architecture']}
- Input Shape: {list(self.encoder.input_shape[1:])}
- Output Shape: {list(self.decoder.output_shape[1:])}
- Total Parameters: {self.vae_model.count_params():,}

## Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
## TensorFlow Version: {tf.__version__}
'''
        
        with open(os.path.join(package_path, 'README.md'), 'w') as f:
            f.write(readme_content)
        
        print(f"Deployment package created successfully at '{package_path}'!")
        return package_path

# Create comprehensive model management
print("Creating comprehensive model management system...")

# Add evaluation results to model manager if available
if 'evaluator' in locals() and hasattr(evaluator, 'evaluation_results'):
    config_with_eval = advanced_vae.beta_scheduler.__dict__.copy()
    config_with_eval.update({
        'latent_dim': latent_dim,
        'architecture': architecture,
        'dropout_rate': dropout_rate,
        'evaluation_results': evaluator.evaluation_results
    })
else:
    config_with_eval = {
        'latent_dim': latent_dim,
        'architecture': architecture,
        'dropout_rate': dropout_rate
    }

# Create model manager
model_manager = VAEModelManager(
    vae_model=advanced_vae,
    encoder=advanced_encoder,
    decoder=advanced_decoder,
    config=config_with_eval
)

# Save complete model
print("\n" + "="*60)
print("SAVING PRODUCTION VAE MODEL")
print("="*60)

model_save_path = model_manager.save_complete_model('production_vae_model')

# Export for inference
print("\n" + "="*60)
print("EXPORTING FOR INFERENCE DEPLOYMENT")
print("="*60)

inference_save_path = model_manager.export_for_inference('vae_inference_deployment')

# Create deployment package
print("\n" + "="*60)
print("CREATING DEPLOYMENT PACKAGE")
print("="*60)

deployment_package_path = model_manager.create_deployment_package('complete_vae_deployment')

# Test loading functionality
print("\n" + "="*60)
print("TESTING MODEL LOADING")
print("="*60)

try:
    print("Testing complete model loading...")
    loaded_model_manager = VAEModelManager.load_complete_model('production_vae_model')
    print("✅ Complete model loaded successfully!")
    
    # Test generation with loaded model
    test_samples = loaded_model_manager.vae_model.generate_samples(4)
    print(f"✅ Generated {len(test_samples)} test samples successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("\n" + "="*60)
print("MODEL PERSISTENCE AND DEPLOYMENT COMPLETED")
print("="*60)
print(f"✅ Complete model saved to: {model_save_path}")
print(f"✅ Inference model exported to: {inference_save_path}")
print(f"✅ Deployment package created at: {deployment_package_path}")
print("\nYour VAE is now ready for production deployment!")