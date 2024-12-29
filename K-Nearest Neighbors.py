import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = "https://raw.githubusercontent.com/Ypknvp/Ypknvp/main/diabetes.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, header=None, skiprows=1, names=column_names)

print("First few rows of the dataset:\n")
print(df.head())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
acc_score = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(f"Accuracy Score: {acc_score:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from matplotlib.colors import ListedColormap

X_train_reduced = X_train[:, :2]
X_test_reduced = X_test[:, :2]
knn_reduced = KNeighborsClassifier(n_neighbors=8)
knn_reduced.fit(X_train_reduced, y_train)

h = 0.02
x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn_reduced.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'blue')))
plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=y_train, s=20, edgecolor='k')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.title('KNN Decision Boundary (using first two features)')
plt.show()

unknown_point = np.array([[5, 120, 70, 35, 0, 32.0, 0.5, 30]])
unknown_point_scaled = scaler.transform(unknown_point)
unknown_class = knn.predict(unknown_point_scaled)
print(f"The unknown point was classified as: {'Diabetes' if unknown_class[0] == 1 else 'No Diabetes'}")
unknown_points = np.array([
    [2, 140, 85, 40, 0, 33.6, 0.627, 32],
    [7, 115, 74, 0, 0, 25.9, 0.587, 51],
    [1, 89, 66, 23, 94, 28.1, 0.167, 21]
])

unknown_points_scaled = scaler.transform(unknown_points)
unknown_classes = knn.predict(unknown_points_scaled)

for i, point in enumerate(unknown_classes):
    print(f"Unknown point {i + 1} was classified as: {'Diabetes' if point == 1 else 'No Diabetes'}")
In this section, we're predicting multiple unknown points and classifying them as either 'Diabetes' or 'No Diabetes'. After scaling the data, we use the trained KNN model to make predictions and print the results for each unknown point.
