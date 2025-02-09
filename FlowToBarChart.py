import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# Define paths
dataset_path = r"C:\Users\DELL\Desktop\CIC2019\CIC-DDoS2019-TrafficClassification.csv"
images_path = r"C:\Users\DELL\Desktop\CIC2019\Images"

# Ensure the images directory exists
os.makedirs(images_path, exist_ok=True)

# Load the dataset
df = pd.read_csv(dataset_path)

# Separate features and labels
data = df.iloc[:, :-1]  # All columns except the last
labels = df.iloc[:, -1]  # Last column (class labels)

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Get unique class labels
unique_labels = labels.unique()

# Create a directory for each unique class label
for label in unique_labels:
    class_dir = os.path.join(images_path, str(label))
    os.makedirs(class_dir, exist_ok=True)

# Generate bar chart images and save to the corresponding directories
for index, row in enumerate(normalized_data):
    label = labels.iloc[index]  # Get the label for the current row
    class_dir = os.path.join(images_path, str(label))  # Get the corresponding directory
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(row)), row, color='black')  # Use black bars
    plt.title(f"Row {index} - Class {label}")
    plt.xlabel("Features")
    plt.ylabel("Normalized Values")
    plt.tight_layout()
    
    # Save the image in the appropriate class directory
    image_file = os.path.join(class_dir, f"row_{index}.png")
    plt.savefig(image_file)
    plt.close()

print(f"Normalized bar chart images saved in directories under {images_path}")
