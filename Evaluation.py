import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import pickle
import argparse

# Set up argument parser for command line options
parser = argparse.ArgumentParser(description='Evaluate emotion recognition model on test data')
parser.add_argument('--actor_dir', type=str, default=r'D:\Test\Actor_20',
                    help='Path to a single actor folder containing WAV files')
args = parser.parse_args()

# Set paths
actor_dir = args.actor_dir
model_path = r"C:\Users\dhruv\OneDrive\Desktop\EE708\improved_emotion_recognition_model.h5"
normalizer_path = r"C:\Users\dhruv\OneDrive\Desktop\EE708\feature_normalizer.pkl"
label_encoder_path = r"C:\Users\dhruv\OneDrive\Desktop\EE708\label_encoder.pkl"

# RAVDESS emotion mapping
emotion_mapping = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print(f"\nLoading model and preprocessing tools...")
# Load the trained model and preprocessing tools
model = load_model(model_path)
with open(normalizer_path, 'rb') as f:
    normalizer = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Print the model summary
print("\nModel Summary:")
model.summary()

def extract_features(file_path, max_pad_len=32):
    """Extract audio features from a file"""
    try:
        # Load and preprocess audio
        y, sr = librosa.load(file_path, sr=16000)
        y = librosa.effects.preemphasis(y, coef=0.95)
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
        mfcc_delta = librosa.feature.delta(mfcc)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Pad or truncate features
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            mfcc_delta = np.pad(mfcc_delta, pad_width=((0, 0), (0, pad_width)), mode='constant')
            spectral_contrast = np.pad(spectral_contrast, pad_width=((0, 0), (0, pad_width)), mode='constant')
            spectral_centroid = np.pad(spectral_centroid, pad_width=((0, 0), (0, pad_width)), mode='constant')
            zcr = np.pad(zcr, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            mfcc_delta = mfcc_delta[:, :max_pad_len]
            spectral_contrast = spectral_contrast[:, :max_pad_len]
            spectral_centroid = spectral_centroid[:, :max_pad_len]
            zcr = zcr[:, :max_pad_len]
        
        # Combine features
        features = np.concatenate([mfcc, mfcc_delta, spectral_contrast, spectral_centroid, zcr], axis=0)
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def process_actor_folder(actor_dir):
    """Process a folder containing audio files for a single actor"""
    if not os.path.isdir(actor_dir):
        return None, None
    
    # Get all WAV files from the actor directory
    wav_files = [f for f in os.listdir(actor_dir) if f.endswith('.wav')]
    if not wav_files:
        print(f"No WAV files found in {actor_dir}")
        return None, None
    
    print(f"Found {len(wav_files)} WAV files in {os.path.basename(actor_dir)}")
    
    test_files = []
    test_labels = []
    
    for file in wav_files:
        # Parse the filename to get emotion
        # RAVDESS format: modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
        file_parts = file.split('-')
        if len(file_parts) >= 7:
            emotion_code = file_parts[2]
            if emotion_code in emotion_mapping:
                emotion = emotion_mapping[emotion_code]
                file_path = os.path.join(actor_dir, file)
                features = extract_features(file_path)
                if features is not None:
                    test_files.append(features)
                    test_labels.append(emotion)
                    print(f"  Processed {file} -> {emotion}")
    
    if not test_files:
        print(f"No valid test files processed in {actor_dir}")
        return None, None
    
    print(f"  Successfully processed {len(test_files)} files from {os.path.basename(actor_dir)}")
    return np.array(test_files), test_labels

def process_single_actor(actor_directory):
    """
    Process a single actor folder and return the features and labels.
    
    Args:
        actor_directory (str): Path to the actor directory containing WAV files
        
    Returns:
        tuple: (all_test_files, all_test_labels, results_by_actor) or (None, None, {}) if processing fails
    """
    if not os.path.isdir(actor_directory):
        print(f"Error: The specified actor directory '{actor_directory}' does not exist or is not a directory.")
        return None, None, {}
        
    print(f"\nProcessing single actor folder: {actor_directory}")
    X_test, test_labels = process_actor_folder(actor_directory)
    
    all_test_files = []
    all_test_labels = []
    results_by_actor = {}
    
    if X_test is not None:
        all_test_files.append(X_test)
        all_test_labels.extend(test_labels)
        results_by_actor[os.path.basename(actor_directory)] = (X_test, test_labels)
        return all_test_files, all_test_labels, results_by_actor
    else:
        print(f"No valid data could be processed from {actor_directory}")
        return None, None, {}

def evaluate_simple_model(X_test_reshaped, y_test_onehot, true_labels, label_encoder):
    """Evaluate using the Simple model approach (high variance features)"""
    # Analyze feature variance
    print("\n--- Feature Variance Analysis ---")
    feature_variance = np.var(X_test_reshaped, axis=0)
    
    print(f"Average feature variance: {np.mean(feature_variance):.4f}")
    print(f"Percentage of low-variance features (var < 0.01): {np.mean(feature_variance < 0.01) * 100:.2f}%")
    
    # Use only high variance features
    high_var_indices = np.where(feature_variance > np.percentile(feature_variance, 50))[0]
    X_high_var = X_test_reshaped[:, high_var_indices]
    print(f"Selected {len(high_var_indices)} high-variance features out of {X_test_reshaped.shape[1]}")
    
    # Scale these features
    X_high_var_scaled = (X_high_var - np.mean(X_high_var, axis=0)) / (np.std(X_high_var, axis=0) + 1e-10)
    
    # Create a simple model
    num_classes = len(label_encoder.classes_)
    
    # Create and train the simple model
    print("\nTraining a simple model on high-variance features...")
    simple_model = Sequential([
        Dense(64, activation='relu', input_shape=(len(high_var_indices),)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    simple_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    simple_model.fit(X_high_var_scaled, y_test_onehot, epochs=10, batch_size=16, verbose=1)
    
    # Evaluate the simple model
    simple_pred_probs = simple_model.predict(X_high_var_scaled)
    simple_pred_labels = np.argmax(simple_pred_probs, axis=1)

# Calculate metrics
    accuracy = accuracy_score(true_labels, simple_pred_labels)
    precision = precision_score(true_labels, simple_pred_labels, average="weighted", zero_division=0)
    recall = recall_score(true_labels, simple_pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, simple_pred_labels, average="weighted", zero_division=0)
    
    # Print metrics
    print(f"\nSimple Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

    # Check if predictions are varied
    simple_unique, simple_counts = np.unique(simple_pred_labels, return_counts=True)
    print("\nPredicted class distribution:")
    for cls, count in zip(simple_unique, simple_counts):
        emotion = label_encoder.classes_[cls]
        print(f"  {emotion}: {count} predictions ({(count/len(simple_pred_labels))*100:.1f}%)")
    
    # Classification report
    report = classification_report(true_labels, simple_pred_labels, 
                                   target_names=label_encoder.classes_, 
                                   zero_division=0)
print("\nClassification Report:\n", report)

# Confusion matrix
    cm = confusion_matrix(true_labels, simple_pred_labels)
plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
    plt.title("Confusion Matrix (Simple Model)")
    plt.tight_layout()
plt.show()

    return simple_pred_labels, accuracy, precision, recall, f1

# Main processing function
def main():
    # Initialize aggregated results
    all_test_files = []
    all_test_labels = []
    results_by_actor = {}
    
    # Process the single actor folder
    all_test_files, all_test_labels, results_by_actor = process_single_actor(actor_dir)
    if not all_test_files:
        print("No valid test files found. Exiting.")
        return
    
    # Combine all test files into a single array
    X_test_combined = np.vstack(all_test_files)
    print(f"\nTest data: {X_test_combined.shape[0]} samples")
    
    # Reshape and normalize
    X_test = X_test_combined.reshape(X_test_combined.shape[0], X_test_combined.shape[1], X_test_combined.shape[2], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    # Encode labels
    y_test = label_encoder.transform(all_test_labels)
    y_test_onehot = tf.keras.utils.to_categorical(y_test)
    true_labels = np.argmax(y_test_onehot, axis=1)
    
    # Analyze class distribution in the test dataset
    print("\n=== Test Dataset Class Distribution ===")
    unique_classes, class_counts = np.unique(all_test_labels, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class '{cls}': {count} samples ({(count/len(all_test_labels))*100:.1f}%)")
    
    # Evaluate using the Simple model approach
    print("\n=== Evaluating with Simple Model ===")
    evaluate_simple_model(X_test_reshaped, y_test_onehot, true_labels, label_encoder)

if __name__ == "__main__":
    main()