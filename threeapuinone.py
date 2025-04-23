from flask import Flask, jsonify, json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from collections import defaultdict
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd


app = Flask(__name__)


@app.route("/")
def home():
    return "Hello, Flask on Windows! "


# Load dummy data
with open("dummy10k_data.json", "r") as f:
    data = json.load(f)["site_users"]


def load_data_for_surprise():
    # Convert dummy data to a DataFrame (user, service, count)
    rows = []
    for user, user_data in data.items():
        for service, count in user_data["category_usage"].items():
            rows.append({"user": user, "service": service, "count": count})
    df = pd.DataFrame(rows)
    
    # Define rating scale (e.g., 0-10 usage counts)
    reader = Reader(rating_scale=(0, 10))
    return Dataset.load_from_df(df[["user", "service", "count"]], reader)


# Load and train model
def get_trained_model():
    dataset = load_data_for_surprise()
    trainset, testset = train_test_split(dataset, test_size=0.2)
    model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model

model_svd = get_trained_model()


def analyze_food_preferences():
    # Initialize data structures
    user_food_counts = defaultdict(lambda: defaultdict(int))
    user_food_recency = defaultdict(dict)
    all_dishes = set()
    
    # Process all users
    for user_id, user_data in data.items():
        for order in user_data.get("food_orders", []):
            dish = order["dish"]
            timestamp = datetime.strptime(order["timestamp"], "%Y-%m-%d %H:%M:%S")
            
            # Count frequency
            user_food_counts[user_id][dish] += 1
            
            # Track most recent order
            if dish not in user_food_recency[user_id] or \
               timestamp > user_food_recency[user_id][dish]:
                user_food_recency[user_id][dish] = timestamp
            
            all_dishes.add(dish)
    
    return user_food_counts, user_food_recency, list(all_dishes)

# Initialize models
food_counts, food_recency, ALL_DISHES = analyze_food_preferences()


def find_similar_food_users(target_user, n=3):
    # Simple similarity based on food order overlap
    similarities = []
    target_dishes = set(food_counts[target_user].keys())
    
    for user in food_counts:
        if user == target_user:
            continue
            
        common_dishes = target_dishes & set(food_counts[user].keys())
        similarity = len(common_dishes) / len(target_dishes) if target_dishes else 0
        similarities.append((user, similarity))
    
    # Return top N similar users
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]


# first api to get current_favorites
def predict_favorite_food(username):
    
    # Fast exit if user not found
    user_foods = food_counts.get(username)
    if user_foods is None:
        return jsonify({"error": "User not found"}), 404
    
    # if username not in food_counts:
    #     return jsonify({"error": "User not found"}), 404
    
    # Calculate weighted scores (frequency + recency)
    scores = []
    current_time = datetime.now()
    
    for dish in food_counts[username]:
        frequency = food_counts[username][dish]
        last_order_age = (current_time - food_recency[username][dish]).days
        recency_weight = max(0, 30 - last_order_age) / 30  # 0-1 scale (30-day window)
        
        # Combined score (70% frequency, 30% recency)
        score = 0.7 * frequency + 0.3 * recency_weight * 10
        scores.append((dish, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return jsonify({
        # "user": username,
        # "favorite_foods": [{"dish": dish, "score": round(score, 2)} for dish, score in scores[:5]],
        "top_prediction": scores[0][0] if scores else None
    })

from collections import Counter
#second api to get best time to order food
def best_time_and_unused_service(username):
    user_data = data.get(username)
    if not user_data:
        return jsonify({"error": "User not found"}), 404

    try:
        order_hours = [
            datetime.strptime(order["timestamp"], "%Y-%m-%d %H:%M:%S").hour
            for order in user_data.get("food_orders", [])
        ]
            
    except Exception as e:
        return jsonify({"error": f"Error parsing timestamps: {str(e)}"}), 400

    # Efficient count using Counter
    category_usage = user_data.get("category_usage", {})
    unused = [service for service, count in category_usage.items() if count == 0]
    
    if order_hours:
        best_hour = Counter(order_hours).most_common(1)[0][0]
    else:
        best_hour = 12  # Default fallback

    return jsonify({
        "unused_services": unused,
        "best_notification_hour": best_hour
    })


# third api to get unused service
# def unused_services(username):
#     user_data = data.get(username)
#     if not user_data:
#         return jsonify({"error": "User not found"}), 404

#     category_usage = user_data.get("category_usage", {})
#     unused = [service for service, count in category_usage.items() if count == 0]

#     return jsonify(unused)


@app.route("/all_recommendations/<username>")
def all_recommendations(username):
    
    favorites = predict_favorite_food(username).get_json()
    
    best_notification_time = best_time_and_unused_service(username).get_json()
    
    # Get similar users' dishes
    similar_users = find_similar_food_users(username)
    recommendations = set()
    
    for user, _ in similar_users:
        for dish in food_counts[user]:
            if dish not in food_counts[username]:  # Only new dishes
                recommendations.add(dish)
    
    return jsonify({
        "user": username,
        "current_favorites": favorites,
        "recommended_new_dishes": list(recommendations)[:5],
        "similar_users": [user for user, _ in similar_users],
        "best_notification_hour": best_notification_time,
    })
    

@app.route("/for_all_users")
def for_all_users():
    all_recommendations_list = []
    
    for username in data.keys():
        print(f"Processing recommendations for {username}...")
        recommendations = all_recommendations(username).get_json()
        all_recommendations_list.append(recommendations)
    
    return jsonify(all_recommendations_list)

if __name__ == "__main__":
    app.run(debug=True)
    

# @app.route("/recommendations/<username>")
# def get_recommendations(username):
#     if username not in data:
#         return jsonify({"error": "User not found"}), 404
    
#     # Step 1: Get unused services
#     unused = [
#         service for service, count in data[username]["category_usage"].items() 
#         if count == 0
#     ]
    
#     # Step 2: Predict engagement for unused services
#     recommendations = []
#     for service in unused:
#         predicted_usage = model_svd.predict(username, service).est
#         if service == "emergency":
#             predicted_usage *= 1.2
#         recommendations.append({
#             "service": service,
#             "predicted_usage": predicted_usage
#         })
    
#     # Step 3: Sort by highest predicted usage
#     recommendations.sort(key=lambda x: x["predicted_usage"], reverse=True)
    
#     return jsonify({
#         # "user": username,
#         "unused_services": unused,
#         "recommendations": recommendations
#     })