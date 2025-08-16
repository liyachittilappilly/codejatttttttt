import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import joblib
import seaborn as sns

def train_model(data_path, model_path):
    """Train the model and save it"""
    # Define dataset path
    DATASET_PATH = data_path
    
    # Check folder structure
    print("Contents of dataset directory:")
    print(os.listdir(DATASET_PATH))
    print(f"\nNumber of '0' (no abs) images: {len(os.listdir(os.path.join(DATASET_PATH, '0')))}")
    print(f"Number of '3' (abs) images: {len(os.listdir(os.path.join(DATASET_PATH, '3')))}")
    
    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        classes=['0', '3'],  # Explicitly define class order
        subset='training')
    
    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        classes=['0', '3'],  # Explicitly define class order
        subset='validation')
    
    # Display class indices
    print("\nClass indices:", train_generator.class_indices)  # Should show {'0': 0, '3': 1}
    
    # Load pre-trained VGG16 model for feature extraction
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    
    # Function to extract features - FIXED VERSION
    def extract_features(generator):
        features = []
        labels = []
        for inputs_batch, labels_batch in generator:
            features_batch = feature_extractor.predict(inputs_batch)
            features_batch = features_batch.reshape(len(inputs_batch), -1)
            features.append(features_batch)
            labels.append(labels_batch)
            # Stop when we've processed all samples
            if len(features) * generator.batch_size >= generator.samples:
                break
        return np.concatenate(features), np.concatenate(labels)
    
    # Extract features
    print("\nExtracting features...")
    train_features, train_labels = extract_features(train_generator)
    validation_features, validation_labels = extract_features(validation_generator)
    
    print(f"\nTrain features shape: {train_features.shape}")
    print(f"Validation features shape: {validation_features.shape}")
    
    # Perform K-Means clustering
    print("\nPerforming K-Means clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(train_features)
    
    # Visualize clusters using PCA
    print("Creating cluster visualization...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    train_features_pca = pca.fit_transform(train_features)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=train_labels, cmap='coolwarm', alpha=0.6)
    plt.title("Actual Classes (0: No Abs, 1: Abs)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.subplot(1, 2, 2)
    plt.scatter(train_features_pca[:, 0], train_features_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title("K-Means Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'clustering_visualization.png'))
    plt.close()
    print("Cluster visualization saved to:", os.path.join(model_path, 'clustering_visualization.png'))
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(train_features, train_labels)
    
    # Evaluate on validation set
    print("Evaluating model...")
    val_predictions = classifier.predict(validation_features)
    val_accuracy = accuracy_score(validation_labels, val_predictions)
    print(f"\nValidation Accuracy: {val_accuracy:.2%}")
    
    # Display confusion matrix
    print("Creating confusion matrix...")
    cm = confusion_matrix(validation_labels, val_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Abs (0)', 'Abs (3)'], 
                yticklabels=['No Abs (0)', 'Abs (3)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(model_path, 'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved to:", os.path.join(model_path, 'confusion_matrix.png'))
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(validation_labels, val_predictions, 
                              target_names=['No Abs (0)', 'Abs (3)']))
    
    # Save models for future use
    print("\nSaving models...")
    feature_extractor.save(os.path.join(model_path, 'feature_extractor.h5'))
    joblib.dump(classifier, os.path.join(model_path, 'classifier.pkl'))
    print("Models saved successfully!")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Abs Detection Model Training')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='./models',
                        help='Path to save models')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    train_model(args.data_path, args.model_path)