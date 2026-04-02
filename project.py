import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# ---------------------------
# LOAD DATA
# ---------------------------
with open('genre_mapping.json', 'r') as file:
    genre = json.load(file)

df = pd.read_csv('tmdb_movies_cleaned.csv')

# ---------------------------
# CLEAN DATA
# ---------------------------
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['month'] = df['release_date'].dt.month

# Keep important columns
df = df[['genre_ids', 'vote_average', 'vote_count', 'popularity', 'month']].dropna()

# Remove low-quality movies (VERY IMPORTANT)
df = df[df['vote_count'] > 50]

#Outcome variable distribution
plt.figure()
plt.hist(df['vote_average'], bins=30)
plt.title("Distribution of Movie Ratings (vote_average)")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()
#Input variable distributions
plt.figure()
plt.hist(df['popularity'], bins=30)
plt.title("Distribution of Popularity")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# ENCODE GENRES
# ---------------------------
df['genre_ids'] = df['genre_ids'].astype(str).str.split(',')

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_ids'])

# ---------------------------
# FEATURES & TARGET
# ---------------------------
X = np.hstack((
    genre_encoded,
    df[['popularity', 'vote_count', 'month']].values
))

y = df['vote_average'].values

# ---------------------------
# TRAIN / TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=420
)

# ---------------------------
# MODEL
# ---------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=250,
    random_state=420
)
model.fit(X_train, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
#linear_regressor = linear_model.LinearRegression()
#linear_regressor.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(f"R² Score: {sm.r2_score(y_test, y_test_pred):.3f}")

# ---------------------------
# USER INPUT
# ---------------------------
genre_array = list(genre.values())
print("\nAvailable genres:")
print(genre_array)

user_genre = input("\nEnter a genre: ")

# Find genre ID
genre_id = None
for key, value in genre.items():
    if value.lower() == user_genre.lower():
        genre_id = key
        break

if genre_id is None:
    print("Genre not found")
    exit()

# ---------------------------
# PREDICT BEST MONTH
# ---------------------------
months = np.arange(1, 13)

# Create genre vector
genre_vector = mlb.transform([[genre_id]])

best_month = None
best_rating = -1

for m in months:
    # Use average values for other features
    popularity_mean = df['popularity'].mean()
    vote_count_mean = df['vote_count'].mean()

    X_input = np.hstack((
        genre_vector,
        [[popularity_mean, vote_count_mean, m]]
    ))

    prediction = model.predict(X_input)[0]

    if prediction > best_rating:
        best_rating = prediction
        best_month = m

# Convert month to name
months_names = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

print(f"\nBest month for {user_genre}: {months_names[best_month - 1]}")
print(f"Predicted rating: {round(best_rating, 2)}")
