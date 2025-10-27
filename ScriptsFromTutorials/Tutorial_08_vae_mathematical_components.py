# Tutorial 08: VAE Mathematical Components & Reparameterization
# Step 1: Setting Up Our VAE Mathematical Laboratory
# ********************************************************************
# Import essential libraries for VAE mathematical implementation
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Input, Lambda
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization
from tensorflow.keras import backend as K
import scipy.stats as stats
import seaborn as sns

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up VAE mathematical components laboratory...")
print("Today we'll learn the mathematical magic behind Variational Autoencoders!")

# Configure matplotlib for mathematical visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

# Step 2: Understanding Probability Distributions in VAEs
# ********************************************************************
def understand_probability_distributions():
    """
    Understand how VAEs use probability distributions instead of fixed points
    """
    print("\n" + "="*60)
    print("UNDERSTANDING PROBABILITY DISTRIBUTIONS IN VAES")
    print("="*60)
    
    print("Key Concept: Instead of encoding to a fixed point, VAEs encode to a distribution!")
    print("Regular AE: image → fixed latent vector")
    print("VAE: image → probability distribution over latent vectors")
    
    # Demonstrate the difference with visualizations
    plt.figure(figsize=(18, 6))
    
    # Regular autoencoder encoding
    plt.subplot(1, 4, 1)
    plt.scatter([2], [1], c='red', s=200, marker='o')
    plt.xlim(0, 4); plt.ylim(0, 3)
    plt.title('Regular AE:\nFixed Point Encoding')
    plt.xlabel('Latent Dim 1'); plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    
    # VAE encoding - distribution
    plt.subplot(1, 4, 2)
    x = np.linspace(0, 4, 100); y = np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x, y); pos = np.dstack((X, Y))
    rv = stats.multivariate_normal([2, 1.5], [[0.3, 0], [0, 0.2]])
    plt.contour(X, Y, rv.pdf(pos), colors='blue')
    plt.scatter([2], [1.5], c='blue', s=100, marker='x')
    plt.title('VAE:\nDistribution Encoding')
    plt.xlabel('Latent Dim 1'); plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    
    # Show multiple samples from VAE distribution
    plt.subplot(1, 4, 3)
    samples = rv.rvs(50)
    plt.scatter(samples[:, 0], samples[:, 1], c='blue', alpha=0.6, s=50)
    plt.scatter([2], [1.5], c='red', s=100, marker='x', label='Mean')
    plt.title('VAE: Multiple Samples\nfrom Distribution')
    plt.xlabel('Latent Dim 1'); plt.ylabel('Latent Dim 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show how this enables generation
    plt.subplot(1, 4, 4)
    x = np.linspace(-3, 3, 100); y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y); pos = np.dstack((X, Y))
    rv_standard = stats.multivariate_normal([0, 0], [[1, 0], [0, 1]])
    plt.contour(X, Y, rv_standard.pdf(pos), colors='green')
    gen_samples = rv_standard.rvs(30)
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], c='green', alpha=0.7, s=50)
    plt.title('Generation:\nSample from N(0,1)')
    plt.xlabel('Latent Dim 1'); plt.ylabel('Latent Dim 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Understand probability distributions
understand_probability_distributions()


# Step 3: The Reparameterization Trick
# ********************************************************************
def understand_reparameterization_trick():
    """
    Understand the reparameterization trick - the key insight that makes VAE training possible
    """
    print("\n" + "="*60)
    print("UNDERSTANDING THE REPARAMETERIZATION TRICK")
    print("="*60)
    
    print("The Problem: How do you backpropagate through random sampling?")
    print("The Solution: z = μ + σ * ε, where ε ~ N(0,1)")
    print("Now gradients can flow through μ and σ!")
    
    # Visualization of the reparameterization trick
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Step 1: Show the problem with direct sampling
    axes[0, 0].text(0.5, 0.5, 'Direct Sampling:\nz ~ N(μ, σ²)\n\n❌ No gradients\nthrough random\nsampling!', 
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightcoral'))
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('Problem: Direct Sampling')
    axes[0, 0].axis('off')
    
    # Step 2: Show the reparameterization solution
    axes[0, 1].text(0.5, 0.5, 'Reparameterization:\nε ~ N(0, 1)\nz = μ + σ * ε\n\n✅ Gradients flow\nthrough μ and σ!', 
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgreen'))
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('Solution: Reparameterization')
    axes[0, 1].axis('off')
    
    # Step 3: Visual demonstration with actual distributions
    mu, sigma = 2.0, 1.5
    x = np.linspace(-2, 6, 1000)
    
    # Original distribution
    original_dist = stats.norm(mu, sigma)
    axes[0, 2].plot(x, original_dist.pdf(x), 'b-', linewidth=2, label=f'N({mu}, {sigma}²)')
    axes[0, 2].axvline(mu, color='red', linestyle='--', alpha=0.7, label='μ')
    axes[0, 2].fill_between(x, 0, original_dist.pdf(x), alpha=0.3)
    axes[0, 2].set_title('Target Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Standard normal (epsilon)
    epsilon_dist = stats.norm(0, 1)
    x_std = np.linspace(-4, 4, 1000)
    axes[1, 0].plot(x_std, epsilon_dist.pdf(x_std), 'g-', linewidth=2, label='ε ~ N(0, 1)')
    axes[1, 0].fill_between(x_std, 0, epsilon_dist.pdf(x_std), alpha=0.3, color='green')
    axes[1, 0].set_title('Standard Normal (ε)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Transformation visualization
    axes[1, 1].arrow(0, 0.5, 0.8, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')
    axes[1, 1].text(0.4, 0.6, 'z = μ + σ * ε', ha='center', fontsize=14, weight='bold')
    axes[1, 1].text(0.4, 0.3, f'z = {mu} + {sigma} * ε', ha='center', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Transformation')
    axes[1, 1].axis('off')
    
    # Show samples from both methods give same result
    np.random.seed(42)
    direct_samples = np.random.normal(mu, sigma, 1000)
    epsilon_samples = np.random.normal(0, 1, 1000)
    reparam_samples = mu + sigma * epsilon_samples
    
    axes[1, 2].hist(direct_samples, bins=30, alpha=0.5, label='Direct sampling', density=True)
    axes[1, 2].hist(reparam_samples, bins=30, alpha=0.5, label='Reparameterized', density=True)
    axes[1, 2].plot(x, original_dist.pdf(x), 'r-', linewidth=2, label='True distribution')
    axes[1, 2].set_title('Sampling Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate the mathematical equivalence
    print("\nMathematical Demonstration:")
    print(f"μ = {mu}, σ = {sigma}")
    print(f"Direct sampling mean: {np.mean(direct_samples):.3f}")
    print(f"Reparameterized mean: {np.mean(reparam_samples):.3f}")
    print(f"Direct sampling std: {np.std(direct_samples):.3f}")
    print(f"Reparameterized std: {np.std(reparam_samples):.3f}")
    print("\n✅ Both methods produce identical distributions!")
    print("✅ But reparameterization allows gradient flow!")
    
    class ReparameterizationLayer(tf.keras.layers.Layer):
        """Custom layer that implements the reparameterization trick"""
        def call(self, inputs):
            mu, log_var = inputs
            batch_size = tf.shape(mu)[0]
            latent_dim = tf.shape(mu)[1]
            epsilon = tf.random.normal(shape=(batch_size, latent_dim))
            sigma = tf.exp(0.5 * log_var)
            return mu + sigma * epsilon

    return ReparameterizationLayer

# Understand and implement reparameterization
ReparameterizationLayer = understand_reparameterization_trick()

# Step 4: Understanding KL Divergence - The Latent Space Regularizer
# ********************************************************************
def understand_kl_divergence():
    """
    Understand KL divergence and its role in VAE training
    """
    print("\n" + "="*60)
    print("UNDERSTANDING KL DIVERGENCE IN VAES")
    print("="*60)
    
    print("KL Divergence measures how different two probability distributions are.")
    print("In VAEs, we use it to force each latent distribution to be close to N(0,1).")
    
    # Create visualization of KL divergence concept
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show what happens without KL regularization
    x = np.linspace(-4, 4, 1000)
    
    # Example distributions that are far from standard normal
    mu1, sigma1 = 2.0, 0.5
    mu2, sigma2 = -1.5, 2.0
    
    dist1 = stats.norm(mu1, sigma1)
    dist2 = stats.norm(mu2, sigma2)
    standard_normal = stats.norm(0, 1)
    
    axes[0, 0].plot(x, dist1.pdf(x), 'r-', linewidth=2, label=f'μ={mu1}, σ={sigma1}')
    axes[0, 0].plot(x, dist2.pdf(x), 'b-', linewidth=2, label=f'μ={mu2}, σ={sigma2}')
    axes[0, 0].plot(x, standard_normal.pdf(x), 'g--', linewidth=2, label='Standard Normal N(0,1)')
    axes[0, 0].set_title('Without KL Regularization: Chaotic Latent Space')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Show what happens with KL regularization
    mu3, sigma3 = 0.2, 1.1
    mu4, sigma4 = -0.1, 0.9
    
    dist3 = stats.norm(mu3, sigma3)
    dist4 = stats.norm(mu4, sigma4)
    
    axes[0, 1].plot(x, dist3.pdf(x), 'r-', linewidth=2, label=f'μ={mu3}, σ={sigma3}')
    axes[0, 1].plot(x, dist4.pdf(x), 'b-', linewidth=2, label=f'μ={mu4}, σ={sigma4}')
    axes[0, 1].plot(x, standard_normal.pdf(x), 'g--', linewidth=2, label='Standard Normal N(0,1)')
    axes[0, 1].set_title('With KL Regularization: Organized Latent Space')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calculate and show KL divergences
    def kl_divergence_analytical(mu, sigma):
        """Calculate KL divergence from N(mu, sigma) to N(0, 1)"""
        return 0.5 * (mu**2 + sigma**2 - 1 - 2*np.log(sigma))
    
    kl1 = kl_divergence_analytical(mu1, sigma1)
    kl2 = kl_divergence_analytical(mu2, sigma2)
    kl3 = kl_divergence_analytical(mu3, sigma3)
    kl4 = kl_divergence_analytical(mu4, sigma4)
    
    # Visualize KL divergence values
    distributions = ['Dist 1', 'Dist 2', 'Dist 3\n(regularized)', 'Dist 4\n(regularized)']
    kl_values = [kl1, kl2, kl3, kl4]
    colors = ['red', 'blue', 'orange', 'purple']
    
    bars = axes[1, 0].bar(distributions, kl_values, color=colors, alpha=0.7)
    axes[1, 0].set_title('KL Divergence Values')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, kl_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Show the mathematical formula
    axes[1, 1].text(0.1, 0.8, 'KL Divergence Formula:', fontsize=14, weight='bold')
    axes[1, 1].text(0.1, 0.6, 'KL(q(z|x) || p(z)) = ½(μ² + σ² - 1 - log(σ²))', fontsize=12)
    axes[1, 1].text(0.1, 0.4, 'Where:', fontsize=12, weight='bold')
    axes[1, 1].text(0.1, 0.3, '• q(z|x) = encoder distribution N(μ, σ)', fontsize=10)
    axes[1, 1].text(0.1, 0.2, '• p(z) = prior distribution N(0, 1)', fontsize=10)
    axes[1, 1].text(0.1, 0.1, '• Lower KL = closer to standard normal', fontsize=10)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate the effect numerically
    print(f"\nKL Divergence Examples:")
    print(f"Unregularized Dist 1 (μ={mu1}, σ={sigma1}): KL = {kl1:.3f}")
    print(f"Unregularized Dist 2 (μ={mu2}, σ={sigma2}): KL = {kl2:.3f}")
    print(f"Regularized Dist 3 (μ={mu3}, σ={sigma3}): KL = {kl3:.3f}")
    print(f"Regularized Dist 4 (μ={mu4}, σ={sigma4}): KL = {kl4:.3f}")
    print("\n✅ KL regularization forces distributions closer to N(0,1)!")
    print("✅ This creates a smooth, continuous latent space for generation!")

    def kl_divergence_loss(mu, log_var):
        """
        Calculate KL divergence loss for VAE training
        
        Args:
            mu: Mean of the encoded distribution
            log_var: Log variance of the encoded distribution
            
        Returns:
            KL divergence loss
        """
        kl_loss = 0.5 * (tf.square(mu) + tf.exp(log_var) - 1 - log_var)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return kl_loss
        
    return kl_divergence_loss

# Understand and implement KL divergence
kl_divergence_loss = understand_kl_divergence()


# Step 5: Building VAE Encoder and Decoder - Distribution-Based Architecture
# ********************************************************************
def build_vae_encoder(latent_dim=32):
    """
    Build VAE encoder that outputs distribution parameters (mu and log_var)
    """
    encoder_input = Input(shape=(28, 28, 1))
    
    # Feature extraction layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Flatten and create distribution parameters
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    
    # Critical: Output distribution parameters, not fixed codes
    mu = Dense(latent_dim, name='mu')(x)
    log_var = Dense(latent_dim, name='log_var')(x)
    
    # Use reparameterization trick
    z = ReparameterizationLayer()([mu, log_var])
    
    return Model(encoder_input, [mu, log_var, z], name='vae_encoder')

def build_vae_decoder(latent_dim=10):
    """
    Build VAE decoder that reconstructs from latent samples
    """
    decoder_input = Input(shape=(latent_dim,))
    
    # Expand latent code to image dimensions
    x = Dense(256, activation='relu')(decoder_input)
    x = Dense(4 * 4 * 128, activation='relu')(x)
    x = Reshape((4, 4, 128))(x)
    
    # Upsampling layers
    x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)  # 4x4 -> 8x8
    x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)   # 8x8 -> 16x16
    x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)   # 16x16 -> 32x32
    
    # Output layer
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Still 32x32
    
    # FIX: Crop to match MNIST size (32x32 -> 28x28)
    decoder_output = tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)
    
    return Model(decoder_input, decoder_output, name='vae_decoder')

# Build VAE components
latent_dim = 32
vae_encoder = build_vae_encoder(latent_dim)
vae_decoder = build_vae_decoder(latent_dim)

print("VAE Encoder Summary:")
vae_encoder.summary()
print("\nVAE Decoder Summary:")
vae_decoder.summary()

# Step 6: Implementing the Complete VAE Loss Function - Balancing Reconstruction and Regularization
# ********************************************************************
class VAETrainer(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        """
        Complete VAE trainer with dual loss objectives
        
        Args:
            encoder: VAE encoder model
            decoder: VAE decoder model
            beta: Weight for KL divergence term (beta-VAE)
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        
        # Track loss components
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")  # Changed name to "loss"
    
    def call(self, inputs, training=None):
        """Forward pass through complete VAE"""
        mu, log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed
    
    def train_step(self, data):
        """Custom training step implementing VAE loss"""
        with tf.GradientTape() as tape:
            # Forward pass
            mu, log_var, z = self.encoder(data)
            reconstructed = self.decoder(z)
            
            # Reconstruction loss (how well we recreate inputs)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstructed)
            )
            
            # KL divergence loss (how close to standard normal)
            kl_loss = kl_divergence_loss(mu, log_var)
            
            # Total VAE loss
            total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics - THIS IS THE KEY FIX
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        # Return the tracker results, not raw values
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """Validation step"""
        mu, log_var, z = self.encoder(data)
        reconstructed = self.decoder(z)
        
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(data, reconstructed)
        )
        kl_loss = kl_divergence_loss(mu, log_var)
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Update validation metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        # Return tracker results for validation
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

# Create and compile VAE trainer
vae_trainer = VAETrainer(vae_encoder, vae_decoder, beta=1.0)
vae_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

print("VAE Trainer created with dual loss objectives!")
print(f"Beta value: {vae_trainer.beta} (controls KL vs reconstruction balance)")


# Step 7: Training the VAE and Understanding the Learning Process - Observing the Mathematical Dance
# ********************************************************************
# Load and prepare data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = np.expand_dims(train_images, -1).astype("float32") / 255
test_images = np.expand_dims(test_images, -1).astype("float32") / 255

print(f"Training data shape: {train_images.shape}")
print(f"Test data shape: {test_images.shape}")

# Simple training function using existing components
def train_vae_simple(encoder, decoder, train_data, test_data, epochs=20, batch_size=32):
    """Simple VAE training using manual batching"""
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    # Training history
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        epoch_losses = []
        num_batches = len(train_data) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = train_data[start_idx:end_idx]
            
            with tf.GradientTape() as tape:
                # Forward pass
                mu, log_var, z = encoder(batch_data, training=True)
                reconstructed = decoder(z, training=True)
                
                # Reconstruction loss
                recon_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(batch_data, reconstructed)
                )
                
                # KL loss
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + log_var - tf.square(mu) - tf.exp(log_var)
                )
                
                # Total loss
             #   total_loss = recon_loss + kl_loss
                # Total loss with beta weighting for KL term
                beta = 0.1  # Start with much smaller KL weight
                total_loss = recon_loss + beta * kl_loss
            
            # Backward pass
            all_vars = encoder.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(total_loss, all_vars)
            optimizer.apply_gradients(zip(grads, all_vars))
            
            epoch_losses.append(total_loss.numpy())
            
            # Progress update
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {total_loss:.4f}")
        
        # Calculate epoch averages
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation
        val_mu, val_log_var, val_z = encoder(test_data[:1000], training=False)
        val_reconstructed = decoder(val_z, training=False)
        val_recon_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(test_data[:1000], val_reconstructed)
        )
        val_kl_loss = -0.5 * tf.reduce_mean(
            1 + val_log_var - tf.square(val_mu) - tf.exp(val_log_var)
        )
    #    val_loss = val_recon_loss + val_kl_loss
        val_loss = val_recon_loss + 0.1 * val_kl_loss  # Use same beta weighting
        val_losses.append(val_loss.numpy())
        
        print(f"  Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Generate samples every 2 epochs
        if epoch % 2 == 0:
            print("  Generating samples...")
            random_latent = tf.random.normal(shape=(16, latent_dim))
            generated_images = decoder(random_latent, training=False)
            
            # Quick visualization
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(generated_images[i].numpy().squeeze(), cmap='gray')
                ax.axis('off')
            plt.suptitle(f'Generated Images - Epoch {epoch + 1}')
            plt.tight_layout()
            plt.show()
    
    return {'loss': train_losses, 'val_loss': val_losses}

# Train the VAE using the simple approach
print("\nTraining VAE with simple approach...")
try:
    history = train_vae_simple(
        vae_encoder, 
        vae_decoder, 
        train_images, 
        test_images, 
        epochs=20,
        batch_size=64
    )
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training error: {e}")
    # Try with smaller batch size
    print("Trying with smaller batch size...")
    history = train_vae_simple(
        vae_encoder, 
        vae_decoder, 
        train_images[:5000],  # Smaller dataset
        test_images[:1000], 
        epochs=20,
        batch_size=32
    )

# Plot results
def plot_simple_results(history):
    """Plot the training results"""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('VAE Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], 'b-', label='Training')
    plt.plot(history['val_loss'], 'r-', label='Validation')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if 'history' in locals():
    plot_simple_results(history)

# Test final generation
print("\nTesting final generation capability...")
random_latent = tf.random.normal(shape=(8, latent_dim))
generated_images = vae_decoder(random_latent)

plt.figure(figsize=(12, 3))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(generated_images[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle('Final Generated Images from Random Latent Codes')
plt.tight_layout()
plt.show()

# Test reconstruction
print("\nTesting reconstruction capability...")
test_sample = test_images[:8]
mu, log_var, z = vae_encoder(test_sample)
reconstructed = vae_decoder(z)

plt.figure(figsize=(12, 6))
for i in range(8):
    # Original
    plt.subplot(2, 8, i+1)
    plt.imshow(test_sample[i].squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(2, 8, i+9)
    plt.imshow(reconstructed[i].numpy().squeeze(), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.suptitle('VAE Reconstruction Results')
plt.tight_layout()
plt.show()

print("VAE training and testing completed successfully!")
print("The model has learned to balance reconstruction and generation objectives.")