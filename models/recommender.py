import datetime
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Define global interest/dish list for consistent encoding
ALL_INTERESTS = ["movies", "cooking", "music", "reading", "travel", "sports"]

def preprocess_user(user):
    # Favorite and recent categories
    categories = [p["catagory"] for p in user["purchase_history"]]
    fav_cat = Counter(categories).most_common(1)[0][0]
    recent_cat = sorted(user["purchase_history"], key=lambda x: x["date"], reverse=True)[0]["catagory"]

    # Binary encoding for interests
    interests = {f"interest_{i}": int(i in user["interests"]) for i in ALL_INTERESTS}

    # Days since last login
    last_login = datetime.date.fromisoformat(user["last_login"])
    days_since_login = (datetime.date.today() - last_login).days

    return {
        "age": user["age"],
        "gender": user["gender"],
        "location": user["location"],
        "last_login_days_ago": days_since_login,
        "fav_catagory": fav_cat,
        "recent_catagory": recent_cat,
        **interests
    }

def preprocess_hotels(hotel_list):
    hotels_df = pd.DataFrame(hotel_list)

    # One-hot encode cuisine
    cuisine_ohe = pd.get_dummies(hotels_df['cuisine'], prefix="cuisine")

    # For simplicity, drop 'location' for now
    features = pd.concat([hotels_df[['rating']], cuisine_ohe], axis=1)

    return hotels_df, features

def recommend_hotels(user_raw, hotels_raw, top_n=3):
    user_processed = preprocess_user(user_raw)
    hotels_df, hotel_features = preprocess_hotels(hotels_raw)

    # Create a user vector matching hotel feature columns (simplified for now)
    user_vec = [user_processed["age"] / 100, 1 if user_processed["fav_catagory"] == "chinese" else 0]
    user_vec = np.array(user_vec).reshape(1, -1)
    
    hotel_vecs = hotel_features.values[:, :2]  # match dimensions for demo

    # Compute similarity
    sims = cosine_similarity(user_vec, hotel_vecs).flatten()
    top_indices = sims.argsort()[::-1][:top_n]

    recommended = hotels_df.iloc[top_indices]
    return recommended.to_dict(orient="records")