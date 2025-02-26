import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import librosa
import numpy as np
from feature_extraction import get_mfcc_features


# Load the dataset from the CSV file
df = pd.read_csv('spectrograms_with_labels.csv')

# Separate the features and the target label
X = df.drop('label', axis=1)
y = df['label']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,)

# Feature scaling for better performance of KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the KNN classifier
# n_neighbors is set to 5 as a default value; you might want to optimize this
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)
# print(y_pred)
# print(X_test)

df = pd.read_csv('test.csv')

# Separate the features and the target label
tes = df.drop('label', axis=1)
tes =scaler.transform(tes)
y_pred = knn.predict(tes)
print(y_pred)




# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))


