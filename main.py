import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.evaluate import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

# 1. Data loading
df = pd.read_csv('/Users/lisizhuta/Desktop/updated_housing_prices_final.csv')

# 2. Data cleaning
# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print("Duplicate Rows:", duplicate_rows)

# Ensure data types are appropriate
df['Location'] = df['Location'].astype('category')
df['Property Type'] = df['Property Type'].astype('category')
df['Condition'] = df['Condition'].astype('category')
data_types = df.dtypes
print("Data Types:\n", data_types)

# Summary statistics to identify any outliers or anomalies
summary_statistics = df.describe()
print("Summary Statistics:\n", summary_statistics)

# Check for unique values in categorical columns
unique_locations = df['Location'].unique()
unique_property_types = df['Property Type'].unique()
unique_conditions = df['Condition'].unique()
print("Unique Locations:\n", unique_locations)
print("Unique Property Types:\n", unique_property_types)
print("Unique Conditions:\n", unique_conditions)

# 3. Data warehousing
# Create an SQLite database and store the dataset
conn = sqlite3.connect('housing_data.db')
df.to_sql('houses', conn, if_exists='replace', index=False)

# Retrieve and print data from the database
query = "SELECT * FROM houses"
df_retrieved = pd.read_sql(query, conn)
print("Retrieved Data:\n", df_retrieved)

# 4. Visualization
# Histogram of Listing Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Listing Price'], kde=True)
plt.title('Distribution of Listing Prices')
plt.xlabel('Listing Price')
plt.ylabel('Frequency')
plt.show()

# Box plot of House Sizes by Property Type
plt.figure(figsize=(12, 6))
sns.boxplot(x='Property Type', y='Size (m^2)', data=df)
plt.title('House Sizes by Property Type')
plt.xlabel('Property Type')
plt.ylabel('Size (m^2)')
plt.show()

# Heatmap of Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Scatter plot of Listing Prices vs Property Type
plt.figure(figsize=(12, 6))
sns.stripplot(x='Property Type', y='Listing Price', data=df, jitter=True)
plt.title('Listing Prices vs Property Type')
plt.xlabel('Property Type')
plt.ylabel('Listing Price')
plt.show()

# 5. Clusters
# Select features for clustering
features = df[['Size (m^2)', 'Number of Bedrooms', 'Number of Bathrooms', 'Listing Price', 'Year Built', 'Lot Size (m^2)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size (m^2)', y='Listing Price', hue='Cluster', data=df, palette='viridis')
plt.title('Clusters of Houses')
plt.xlabel('Size (m^2)')
plt.ylabel('Listing Price')
plt.show()

# Compute the silhouette score
silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
print(f'Silhouette Score for 5 clusters: {silhouette_avg:.2f}')

# 6. Similarity
# Select features
numerical_features = ['Size (m^2)', 'Number of Bedrooms', 'Number of Bathrooms', 'Listing Price', 'Year Built', 'Lot Size (m^2)']
categorical_features = ['Location', 'Property Type', 'Condition']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing
X = preprocessor.fit_transform(df)

# Calculate Euclidean distances for the numerical features
similarity_matrix = cdist(X, X, metric='euclidean')
similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# Find most similar and dissimilar houses
most_similar_indices = np.unravel_index(np.argsort(similarity_matrix, axis=None), similarity_matrix.shape)
most_dissimilar_indices = np.unravel_index(np.argsort(similarity_matrix, axis=None)[::-1], similarity_matrix.shape)

# Top 10 most similar houses
print("Houses with the highest similarity:")
similar_pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        similar_pairs.append((i, j, similarity_matrix[i][j]))

# Sort by similarity score (ascending)
similar_pairs = sorted(similar_pairs, key=lambda x: x[2])

for index1, index2, sim in similar_pairs[:10]:
    print(f"House {index1 + 1} and House {index2 + 1}, Similarity: {sim:.2f}")

# Top 10 most dissimilar houses
print("\nHouses with the highest dissimilarity:")
dissimilar_pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        dissimilar_pairs.append((i, j, similarity_matrix[i][j]))

# Sort by dissimilarity score (descending)
dissimilar_pairs = sorted(dissimilar_pairs, key=lambda x: x[2], reverse=True)

for index1, index2, dissim in dissimilar_pairs[:10]:
    print(f"House {index1 + 1} and House {index2 + 1}, Dissimilarity: {dissim:.2f}")

# 7. Price prediction
# Features and target variable
X = df[['Size (m^2)', 'Number of Bedrooms', 'Number of Bathrooms', 'Year Built', 'Lot Size (m^2)']]
y = df['Listing Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)

# Train Gradient Boosting Regressor for price prediction
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions
y_pred_gbr = gbr_model.predict(X_test)

# Evaluate the model
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

print("Gradient Boosting Regressor Mean Absolute Error:", mae_gbr)
print("Gradient Boosting Regressor Mean Squared Error:", mse_gbr)
print("Gradient Boosting Regressor Root Mean Squared Error:", rmse_gbr)
print("Gradient Boosting Regressor R^2 Score:", r2_gbr)

# 8. Advanced Classification (Random Forest)
from sklearn.ensemble import RandomForestClassifier

X = df[['Size (m^2)', 'Number of Bedrooms', 'Number of Bathrooms', 'Year Built', 'Lot Size (m^2)']]
y = df['Property Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))

# 9. Anomaly Detection
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(scaled_features)

# Visualize anomalies
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Size (m^2)', y='Listing Price', hue='Anomaly', data=df, palette='coolwarm')
plt.title('Anomaly Detection in Housing Data')
plt.xlabel('Size (m^2)')
plt.ylabel('Listing Price')
plt.show()

# 10. Recommendation System
def recommend_houses(house_index, num_recommendations=5):
    similar_houses = similarity_df.iloc[house_index].sort_values().head(num_recommendations + 1).index
    return df.loc[similar_houses[1:]]

# Recommend houses similar to the house at index 0
recommended_houses = recommend_houses(0)
print("Recommended Houses:\n", recommended_houses)

# Example usage of recommendation system
def predict_price():
    # Gather user input
    size = float(input("Enter Size (m^2): "))
    bedrooms = int(input("Enter Number of Bedrooms: "))
    bathrooms = int(input("Enter Number of Bathrooms: "))
    year_built = int(input("Enter Year Built: "))
    lot_size = float(input("Enter Lot Size (m^2): "))

    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[size, bedrooms, bathrooms, year_built, lot_size]],
                              columns=['Size (m^2)', 'Number of Bedrooms', 'Number of Bathrooms', 'Year Built',
                                       'Lot Size (m^2)'])

    # Predict the price
    predicted_price = gbr_model.predict(input_data)
    print(f"Predicted Listing Price: ${predicted_price[0]:,.2f}")

# Example usage
predict_price()
