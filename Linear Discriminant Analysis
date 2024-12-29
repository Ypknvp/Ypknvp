import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
url = 'https://github.com/Ypknvp/Ypknvp/raw/main/pokemon.csv'
df = pd.read_csv(url)
df_encoded = pd.get_dummies(df, drop_first=True)
imputer = SimpleImputer(strategy='mean')
df_encoded_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)
target_column = 'is_legendary'  # Target column
features = [col for col in df_encoded_imputed.columns if col != target_column]
X = df_encoded_imputed[features]
y = df_encoded_imputed[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
n_classes = len(y_train.unique())
n_features = X_train.shape[1]
n_components = min(n_features, n_classes - 1)
lda_model = LinearDiscriminantAnalysis(n_components=n_components)
X_transformed = lda_model.fit_transform(X_train, y_train)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.subplot(1, 2, 2)
if X_transformed.shape[1] > 1:
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.title('LDA Transformed Data')
else:
    plt.scatter(X_transformed[:, 0], [0] * len(X_transformed), c=y_train, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('LD 1')
    plt.title('LDA Transformed Data (1D)')

plt.tight_layout()
plt.show()
