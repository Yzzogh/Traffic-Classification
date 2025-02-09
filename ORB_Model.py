import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
import psutil
import threading
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global stop signal for monitoring
stop_monitoring = threading.Event()

# Shared lists to store CPU and memory usage
cpu_usages = []
memory_usages = []

# Function to monitor CPU and memory usage
def monitor_resources():
    current_pid = os.getpid()  # Get the current process ID
    cpu_count = psutil.cpu_count(logical=True)  # Number of logical CPUs (calculated once)
    
    while not stop_monitoring.is_set():
        process = psutil.Process(current_pid)
        if process.status() != psutil.STATUS_ZOMBIE:  # Check if the process is still running
            cpu_usage = process.cpu_percent(interval=1) / cpu_count
            memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory in MB
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)
        else:
            break

# Function to extract ORB features from a list of image paths
def extract_orb_features(image_paths, labels, max_keypoints=500):
    features = []
    valid_labels = []
    orb = cv2.ORB_create(nfeatures=max_keypoints)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, img_path, label, orb, max_keypoints) for img_path, label in zip(image_paths, labels)]
        for future in as_completed(futures):
            desc, label = future.result()
            if desc is not None:
                features.append(desc.flatten())
                valid_labels.append(label)
    return features, valid_labels

def process_image(img_path, label, orb, max_keypoints):
    img = cv2.imread(img_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptor = orb.detectAndCompute(gray, None)
        if descriptor is not None:
            descriptor = descriptor[:max_keypoints] if len(descriptor) > max_keypoints else np.pad(descriptor, ((0, max_keypoints - len(descriptor)), (0, 0)), mode='constant')
            return descriptor, label
    return None, label

# Paths to dataset folders
folders_paths = [
    'C:/Users/DELL/Desktop/IEEE/Images/Syn',
    'C:/Users/DELL/Desktop/IEEE/Images/UDP',
    'C:/Users/DELL/Desktop/IEEE/Images/MSSQL',
    'C:/Users/DELL/Desktop/IEEE/Images/LDAP',
    'C:/Users/DELL/Desktop/IEEE/Images/BENIGN',
]

# Create lists of all image file paths and corresponding labels
image_paths = []
labels = []

for label, folder_path in enumerate(folders_paths):
    file_list = os.listdir(folder_path)
    for filename in file_list:
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            image_paths.append(img_path)
            labels.append(label)

# Convert to NumPy arrays
image_paths = np.array(image_paths)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels, shuffle=True
)

# Start monitoring resources
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# Measure training time
train_start_time = time.time()

# Extract ORB features for training set
X_train, y_train = extract_orb_features(X_train_paths, y_train)

# Measure the time taken for feature extraction from the training set
feature_extraction_train_time = time.time() - train_start_time

# Convert lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Scale the training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create a neural network model with dropout layers
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),  # Increased dropout rate
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')
])

# Print a summary of the model
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
model_start_time = time.time()
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# Measure total training time
total_training_time = time.time() - model_start_time


# Measure testing time
testing_start_time = time.time()


# Extract ORB features for testing set
X_test, y_test = extract_orb_features(X_test_paths, y_test)

# Scale the testing features
X_test_scaled = scaler.transform(X_test)

# Predict labels for validation data
y_pred = np.argmax(model.predict(X_test_scaled), axis=-1)

# Measure total testing time
total_testing_time = time.time() - testing_start_time

# Stop monitoring after prediction
stop_monitoring.set()
monitor_thread.join()

# Calculate average CPU and memory usage during testing
avg_cpu = np.mean(cpu_usages)
avg_memory = np.mean(memory_usages)


# Evaluate the performance of the neural network model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Replace class labels for better readability in the confusion matrix
target_names = ['Syn', 'UDP', 'MSSQL', 'LDAP', 'BENIGN']
conf_matrix_df = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)

# Print Average CPU/Memory Usage
print(f"Average CPU Usage During Training: {avg_cpu:.2f}%")
print(f"Average Memory Usage During Training: {avg_memory:.2f} MB")


# Print accuracy, confusion matrix, and classification report
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix_df}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Total training time includes feature extraction + model training
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Total testing time includes feature extraction + testing
print(f"Total Testing Time: {total_testing_time:.2f} seconds")
