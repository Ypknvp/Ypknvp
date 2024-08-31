
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/Ypknvp/Ypknvp/main/data.csv"
df = pd.read_csv(url)

print("Dataset Preview:")
print(df.head())

print("\nColumns in Dataset:")
print(df.columns)

df_numeric = df.select_dtypes(include=[np.number])

X = df[['temperature', 'day_of_week', 'is_weekend', 'is_holiday', 'is_start_of_semester', 'is_during_semester', 'month', 'hour']]  # Selecting features
y = df['number_people']  # Selecting the label
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio for each Principal Component:")
print(explained_variance)

print("\nPrincipal Component Values (First 5 Rows):")
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
print(pca_df.head())  # Display first 5 rows of PCA components

pca = PCA(n_components=5)  # Reducing to 5 components
X_pca_reduced = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
model = RandomForestRegressor()
estimators = np.arange(10, 200, 10)
scores = []

for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print("\nRandom Forest Regressor Scores on Original Data:")
for n, score in zip(estimators, scores):
    print(f"n_estimators={n}: {score}")

# Random Forest Regressor on PCA-transformed dataset
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca_reduced, y, test_size=0.2, random_state=1)
model_pca = RandomForestRegressor()
scores_pca = []

for n in estimators:
    model_pca.set_params(n_estimators=n)
    model_pca.fit(X_train_pca, y_train_pca)
    scores_pca.append(model_pca.score(X_test_pca, y_test_pca))

print("\nRandom Forest Regressor Scores on PCA Transformed Data:")
for n, score in zip(estimators, scores_pca):
    print(f"n_estimators={n}: {score}")
