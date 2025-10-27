# Tutorial 07: Latent Space Exploration & Visualization
# Step 1: Setting Up Our Latent Space Laboratory
# ********************************************************************
# Import essential libraries for latent space analysis and visualization
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape, Input
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Set random seeds for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

print("TensorFlow version:", tf.__version__)
print("Setting up latent space exploration laboratory...")
print("Today we'll discover the hidden world inside autoencoders!")

# Configure matplotlib and seaborn for better visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
sns.set_palette("husl")

# Step 2: Building a Specialized Autoencoder for Latent Exploration
# ********************************************************************
def load_and_prepare_data():
    """
    Loads and normalizes MNIST data for autoencoder training and latent space analysis.
    Returns: train_images, test_images, train_labels, test_labels
    """
    print("\n" + "="*60)
    print("LOADING AND PREPARING DATA FOR LATENT EXPLORATION")
    print("="*60)
    
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and normalize images to [0,1] and add channel dimension
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255.0
    
    print(f"Training images: {train_images.shape}")
    print(f"Test images: {test_images.shape}")
    print("Data prepared for latent space exploration!")
    
    return train_images, test_images, train_labels, test_labels

def build_exploration_autoencoder(latent_dim=64):
    """
    Builds a convolutional autoencoder with a configurable latent space size.
    Returns: autoencoder model, encoder model, decoder model
    """
    print(f"\n--- Building Exploration Autoencoder with {latent_dim}D latent space ---")
    
    # Encoder: Compresses input images into a latent space
    encoder = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),                   # 28x28 -> 14x14
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),                   # 14x14 -> 7x7
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),                   # 7x7 -> 4x4
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(latent_dim, activation='linear', name='latent_layer')  # Linear for unrestricted latent space
    ], name='exploration_encoder')
    
    # Decoder: Reconstructs images from latent codes
    decoder = Sequential([
        Dense(256, activation='relu', input_shape=(latent_dim,)),
        BatchNormalization(),
        Dense(4 * 4 * 128, activation='relu'),
        Reshape((4, 4, 128)),
        
        Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),
        BatchNormalization(),                                                   # 4x4 -> 8x8
        
        Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
        BatchNormalization(),                                                   # 8x8 -> 16x16
        
        Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same'),
        BatchNormalization(),                                                   # 16x16 -> 32x32
        
        Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        tf.keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))  # 32x32 -> 28x28
    ], name='exploration_decoder')
    
    # Combine encoder and decoder into full autoencoder model
    autoencoder_input = Input(shape=(28, 28, 1))
    latent_code = encoder(autoencoder_input)
    reconstructed = decoder(latent_code)
    autoencoder = Model(autoencoder_input, reconstructed, name='exploration_autoencoder')
    
    return autoencoder, encoder, decoder

# Load data and build the autoencoder
train_images, test_images, train_labels, test_labels = load_and_prepare_data()
exploration_autoencoder, exploration_encoder, exploration_decoder = build_exploration_autoencoder(latent_dim=64)


# Step 3:  Training the Exploration Autoencoder
# ********************************************************************
def train_exploration_autoencoder(autoencoder, train_data, test_data, epochs=5):
    """
    Trains the autoencoder for image reconstruction.
    Uses early stopping and learning rate reduction for stability.
    """
    print(f"\n--- Training Exploration Autoencoder for {epochs} epochs ---")
    
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['mse']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=4,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    history = autoencoder.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        validation_data=(test_data, test_data),
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!")
    return history

# Train the autoencoder
print("\n" + "="*60)
print("TRAINING EXPLORATION AUTOENCODER")
print("="*60)
exploration_history = train_exploration_autoencoder(exploration_autoencoder, train_images, test_images)


# Step 4:  Understanding Latent Space Properties
# ********************************************************************
def analyze_and_visualize_latent_space_properties(encoder, test_data, test_labels, num_samples=1000):
    """
    Extracts latent codes and labels for a subset of test data.
    Provides three visualization options (t-SNE, PCA, KMeans).
    Only one visualization block should be active at a time.
    - If all are commented out: function returns codes/labels but produces no plot.
    - If one is uncommented: function displays a plot of the latent space.
    Returns: latent_codes, corresponding_labels
    """
    print(f"\n--- Analyzing and Visualizing Latent Space Properties ---")
    
    # Pass images through encoder to get latent codes (compressed features)
    latent_codes = encoder.predict(test_data[:num_samples], verbose=0)
    corresponding_labels = test_labels[:num_samples]
    
    # ----------- Option 1: t-SNE Visualization -----------
    # Use t-SNE to reduce latent space to 2D and plot, colored by true digit label.
    # Uncomment this block to use t-SNE; comment out the other two options.
    
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_codes)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=corresponding_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Digit Label')
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    
    
    # ----------- Option 2: PCA Visualization -----------
    # Use PCA to reduce latent space to 2D and plot, colored by true digit label.
    # Uncomment this block to use PCA; comment out the other two options.
    """
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_codes)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=corresponding_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='Digit Label')
    plt.title("PCA Visualization of Latent Space")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()
    """
    
    # ----------- Option 3: KMeans Clustering with t-SNE Visualization -----------
    # Use KMeans to cluster latent codes, then plot t-SNE colored by cluster assignment.
    # Uncomment this block to use KMeans; comment out the other two options.
    """
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_codes)
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_codes)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10), label='KMeans Cluster")
    plt.title("t-SNE of Latent Space with KMeans Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    """
    
    # If all visualization blocks are commented out, this function only returns codes/labels.
    # To see a plot, uncomment ONE visualization block above.
    return latent_codes, corresponding_labels

# Analyze and (optionally) visualize latent space properties
# By default, this will only assign latent_codes and labels.
# To see a plot, uncomment one visualization block in the function above.
latent_codes, labels = analyze_and_visualize_latent_space_properties(
    exploration_encoder, test_images, test_labels
)

# Step 5:  Latent Space Interpolation
# ********************************************************************

def perform_latent_interpolation(encoder, decoder, image1, image2, num_steps=10):
    """
    Perform smooth interpolation between two images in latent space.
    Shows a row of images morphing from image1 to image2.
    """
    # Encode both images to latent space
    z1 = encoder.predict(image1[np.newaxis, ...])
    z2 = encoder.predict(image2[np.newaxis, ...])

    # Generate interpolation coefficients between 0 and 1
    alphas = np.linspace(0, 1, num_steps)

    # Interpolate in latent space
    interpolated_z = np.array([(1 - a) * z1 + a * z2 for a in alphas]).squeeze()

    # Decode each interpolated latent vector to image space
    decoded_imgs = decoder.predict(interpolated_z)

    # Visualize the interpolation
    import matplotlib.pyplot as plt
    plt.figure(figsize=(2 * num_steps, 2))
    for i, img in enumerate(decoded_imgs):
        ax = plt.subplot(1, num_steps, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle("Latent Space Interpolation")
    plt.show()

print("\n" + "="*60)
print("LATENT SPACE INTERPOLATION EXPERIMENTS")
print("="*60)

perform_latent_interpolation(
    exploration_encoder,
    exploration_decoder,
    test_images[0],        # First image (can change index)
    test_images[100],      # Second image (can change index)
    num_steps=12           # Number of interpolation steps/images
)


# Step 6: Latent Space Arithmetic
# ********************************************************************
def perform_latent_arithmetic():
    """
    Perform arithmetic operations in latent space.
    Demonstrates: (A - B) + C ≈ D, where A, B, C, D are images.
    """
    # Select images for arithmetic: e.g., A=7, B=1, C=1, expect D~7
    idx_A = 0     # e.g., index of '7'
    idx_B = 1     # e.g., index of '1'
    idx_C = 2     # e.g., index of another '1'
    
    img_A = test_images[idx_A]
    img_B = test_images[idx_B]
    img_C = test_images[idx_C]
    
    # Encode images to latent space
    z_A = exploration_encoder.predict(img_A[np.newaxis, ...])
    z_B = exploration_encoder.predict(img_B[np.newaxis, ...])
    z_C = exploration_encoder.predict(img_C[np.newaxis, ...])
    
    # Latent arithmetic: (A - B) + C
    z_result = z_A - z_B + z_C
    
    # Decode all relevant latent codes
    decoded_A = exploration_decoder.predict(z_A)
    decoded_B = exploration_decoder.predict(z_B)
    decoded_C = exploration_decoder.predict(z_C)
    decoded_result = exploration_decoder.predict(z_result)
    
    # Visualize the arithmetic
    import matplotlib.pyplot as plt
    imgs = [decoded_A, decoded_B, decoded_C, decoded_result]
    titles = ["A (e.g. 7)", "B (e.g. 1)", "C (e.g. 1)", "(A - B) + C"]
    
    plt.figure(figsize=(12, 3))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.suptitle("Latent Space Arithmetic: (A - B) + C")
    plt.show()

perform_latent_arithmetic()


