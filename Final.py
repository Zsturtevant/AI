import json
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor

# Azure SDK Imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# 1. AZURE & DATA CONFIG
# ---------------------------
ENDPOINT = "use your own"
API_KEY = "use your own"

with open('genre_mapping.json', 'r') as file:
    genre_map = json.load(file)

df = pd.read_csv('tmdb_movies_cleaned.csv')
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['month'] = df['release_date'].dt.month
df = df.dropna(subset=['genre_ids', 'vote_average', 'vote_count', 'popularity', 'month'])
df = df[df['vote_count'] > 50]
df['genre_list'] = df['genre_ids'].astype(str).str.split(',')

# Train Model
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_list'])
X = np.hstack((genre_encoded, df[['popularity', 'vote_count', 'month']].values))
y = df['vote_average'].values
model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=420)
model.fit(X, y)

# ---------------------------
# 2. AZURE TEXT ANALYTICS CLIENT
# ---------------------------
text_client = TextAnalyticsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(API_KEY))

def identify_genre_with_azure(user_input):
    """Uses Azure to extract key phrases and match them to your mapping."""
    try:
        # Pass the input as a list to the client
        response = text_client.extract_key_phrases(documents=[user_input])
        # Get phrases from the document
        phrases = [p.lower() for p in response[0].key_phrases]
        
        # Match Azure phrases against  JSON mapping
        for gid, gname in genre_map.items():
            if any(gname.lower() in p for p in phrases) or gname.lower() in user_input.lower():
                return gid, gname
        return None, None
    except Exception as e:
        print(f"Azure Error: {e}")
        return None, None

# ---------------------------
# 3. USER INPUT & PREDICTION
# ---------------------------
print("\n--- Azure Movie Analytics ---")

# PRINT VALID GENRES
print("Valid Genres:", ", ".join(sorted(genre_map.values())))
print("-" * 30)

user_genre = input("Enter a genre or movie description: ").strip()

gid, gname = identify_genre_with_azure(user_genre)

if not gid:
    print(f"Could not identify a valid genre from: '{user_genre}'")
else:
    print(f"\nAzure identified this as: {gname} (ID: {gid})")
    
    # Use baseline averages for prediction
    avg_pop = df['popularity'].mean()
    avg_votes = df['vote_count'].mean()
    genre_vec = mlb.transform([[str(gid)]])
    
    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    monthly_preds = []
    for m in range(1, 13):
        X_in = np.hstack((genre_vec, [[avg_pop, avg_votes, m]]))
        monthly_preds.append(model.predict(X_in))

    best_month_idx = np.argmax(monthly_preds)
    
    print(f"Best Month for {gname}: {month_names[best_month_idx]}")
    print(f"Predicted Rating: {round(float(monthly_preds[best_month_idx][0]), 2)}")

