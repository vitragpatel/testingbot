from flask import Flask, render_template, jsonify, request, Response
import os
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from utils.al_prompt import *
from utils.helpers import *
from utils.get_data import *
from utils.automation import *
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd

# this model takes 15-20 seconds to load
# llm = Llama(model_path="./models/mistral-7b-instruct-v0.1-q4_k_m.gguf", n_ctx=2048)

# this model takes 6-8 seconds to load
# llm = Llama(model_path="./models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf", n_ctx=1024)


load_dotenv()
llm = llm_init()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)























# --- NLP Suggestion Route by user name ---
@app.route("/nlp_suggest_by_name/<user_name>")
def nlp_suggest_local_llm_by_name(user_name):

    all_users = get_site_user_data()

    user_data = None
    for uid, user in all_users.items():
        if user.get("name", "").lower() == user_name.lower():
            user_data = user
            break

    if not user_data:
        return jsonify({"error": "User not found"}), 404

    user_lat = user_data.get("latitude")
    user_lon = user_data.get("longitude")

    if user_lat is None or user_lon is None:
        return jsonify({"error": "User coordinates missing"}), 400

    hotels = get_matching_hotels(user_lat, user_lon)
    prompt = f"[INST] {build_prompt(user_name, user_data, hotels)} [/INST]"

    try:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that recommends food and hotels directly to the user."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            stop=["\n"]
        )
        suggestion = output["choices"][0]["message"]["content"].strip()
        return jsonify({"user_name": user_name, "suggestion": suggestion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# LLM API
@app.route("/nlp_suggest_all")
def nlp_suggest_all():
    try:
        users_ref = db.reference("users/users")
        users_data = users_ref.get()

        if not users_data:
            return jsonify({"error": "No users found"}), 404

        all_suggestions = {}
        for user_id, user_data in users_data.items():
            if user_id == "user_006":
                break

            hotels = get_matching_hotels(user_data.get("location", ""))
            prompt_body = build_prompt(user_id, user_data, hotels)
            prompt = f"[INST] {prompt_body} [/INST]"

            try:
                print(f"Processing user:============== {user_id}")
                ## this is for tinyllama-1.1b-chat-v1.0-q4_k_m.gguf model
                # llm = llm_init()
                
                output = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that recommends food and hotels direct to the user."},
                        {"role": "user", "content": prompt}
                        ],
                    max_tokens=40,
                    stop=["\n"]
                    )
                suggestion = output["choices"][0]["message"]["content"].strip()                
                
                if suggestion:
                    all_suggestions[user_id] = suggestion
                else:
                    all_suggestions[user_id] = "No suggestion generated."
            except Exception as inner_error:
                all_suggestions[user_id] = f"Error generating suggestion: {str(inner_error)}"

        return jsonify(all_suggestions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# GET /data – fetch data from Firebase
@app.route("/data", methods=["GET"])
def get_data():
    hotel_data = get_hotel_data()
    return jsonify(hotel_data), 200


# POST /data – post data to Firebase
@app.route("/data", methods=["POST"])
def post_data():
    content = request.json
    if not content:
        return jsonify({"error": "No data provided"}), 400

    # # Push data (creates unique key)
    # new_ref = ref.push(content)
    
    ref = db.reference("hotels")
    ref.set(content)

    return jsonify({"message": "Data saved"}), 201


@app.route("/ask-question", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    length = data.get("length", "short")  # short, medium, long

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Set token count based on length
    token_map = {
        "short": 100,
        "medium": 300,
        "long": 1000
    }
    max_tokens = token_map.get(length, 150)

    prompt = f"Answer the following question clearly:\n\nQ: {question}\nA:"

    def generate():
        llm = llm_init()
        completion = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in completion:
            if "choices" in chunk:
                yield chunk["choices"][0]["text"]

    return Response(generate(), content_type='text/plain')


# This API endpoint saves user data to Firebase
# and rotates names for each entry.
@app.route("/site-user-data", methods=["POST"])
def site_user_data():
    content = request.json
    if not content:
        return jsonify({"error": "No data provided"}), 400

    ref = db.reference("site-user")
    
    # Predefined list of 5 names to cycle through
    ROTATING_NAMES = [
        "akshay",
        "atul", 
        "chirah",
        "jaydip",
        "vitrag"
    ]
    
    # Get current count of entries to determine rotation position
    snapshot = ref.get()
    current_count = len(snapshot) if snapshot else 0
    
    def process_item(item, index):
        """Process each item with rotating name"""
        # Determine which name to use (cycle through 5 names)
        name_index = index % 5
        assigned_name = ROTATING_NAMES[name_index]
        
        # Create new data with rotated name
        return {
            "name": assigned_name,
            "category": item.get("category"),
            "contactNo": item.get("contactNo"),
            "emergencyNo": item.get("emergencyNo"), 
            "latitude": item.get("latitude"),
            "longitude": item.get("longitude"),
            "location": item.get("location"),
            "types": item.get("types", [])
        }
    
    if isinstance(content, list):
        # Process multiple items
        for i, item in enumerate(content, current_count):
            new_data = process_item(item, i)
            ref.push(new_data)
    else:
        # Process single item
        new_data = process_item(content, current_count)
        ref.push(new_data)
    
    return jsonify({"message": "Data saved with rotating names"}), 201

@app.route("/get-all-data", methods=["GET"])
def get_all_data():
    
    data = get_site_user_data()
    site_user_list = []

    # Loop through each Firebase key-value pair
    for firebase_key, location_data in data.items():
        site_user_list.append({
            "id": firebase_key,  # Keep Firebase's unique ID
            "name": location_data.get("name", ""),
            "category": location_data.get("category", ""),
            "contactNo": location_data.get("contactNo", ""),
            "emergencyNo": location_data.get("emergencyNo", ""),
            "latitude": location_data.get("latitude", 0),
            "longitude": location_data.get("longitude", 0),
            "location": location_data.get("location", ""),
            "types": location_data.get("types", [])
        })

    # ref = db.reference("users")
    # data = ref.get()
    # users_data = data.get("users", [])
    
    ref = db.reference("hotels")
    data = ref.get()
    hotels_data = data.get("hotels", [])
    
    return jsonify({
        "site_user_data": site_user_list,
        # "users_data": users_data,
        "hotels_data": hotels_data
    }), 200


@app.route("/")
def home():
    data_processing()
    return "Hello, Flask on Windows! "


@app.route("/time")
def time():
    return render_template("time.html")


@app.route("/home")
def home1():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("ask.html")


@app.route("/get_user_data_by_name/<user_name>", methods=["GET"])
def get_user_data_by_name(user_name):
    all_users = get_site_user_data()

    user_data = []
    for uid, user in all_users.items():
        if user.get("name", "").lower() == user_name.lower():
            user_data.append(user)
    if not user_data:
        return jsonify({"error": "User not found"}), 404

    return jsonify(user_data), 200


@app.route("/automation", methods=["GET"])
def automation():
    # automation_function_step1()
    print("step 1 done-----------------------------------")
    
    data = automation_function_step2_userwise()
    print("step 2 done-----------------------------------")
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    # response = model.generate_content(prompt)
    result = {}

    for user, services in data.items():
        prompt = build_user_service_data_prompt(user, services)
        
        print(f"Processing user:============== {user}")
        print(f"Prompt: {prompt}")
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Just the top line = the most interested category
        result[user] = content.split('\n')[0]

    # return {"data": result}
    
    return jsonify({"data": result}), 200

# Load dummy data
with open("dummy_data.json", "r") as f:
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

@app.route("/recommendations/<username>")
def get_recommendations(username):
    if username not in data:
        return jsonify({"error": "User not found"}), 404
    
    # Step 1: Get unused services
    unused = [
        service for service, count in data[username]["category_usage"].items() 
        if count == 0
    ]
    
    # Step 2: Predict engagement for unused services
    recommendations = []
    for service in unused:
        predicted_usage = model_svd.predict(username, service).est
        if service == "emergency":
            predicted_usage *= 1.2
        recommendations.append({
            "service": service,
            "predicted_usage": predicted_usage
        })
    
    # Step 3: Sort by highest predicted usage
    recommendations.sort(key=lambda x: x["predicted_usage"], reverse=True)
    
    return jsonify({
        "user": username,
        "unused_services": unused,
        "recommendations": recommendations
    })
    
@app.route("/predict_favorite_food/<username>")
def predict_favorite_food(username):
    if username not in food_counts:
        return jsonify({"error": "User not found"}), 404
    
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
        "user": username,
        "favorite_foods": [{"dish": dish, "score": round(score, 2)} for dish, score in scores[:5]],
        "top_prediction": scores[0][0] if scores else None
    })


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

@app.route("/food_recommendations/<username>")
def food_recommendations(username):
    # Get user's own favorites
    favorites = predict_favorite_food(username).get_json()
    
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
        "similar_users": [user for user, _ in similar_users]
    })
    

@app.route("/best_time/<username>")
def best_time(username):
    if username not in data:
        return jsonify({"error": "User not found"}), 404
    
    # Mock: Predict best time (e.g., most frequent order hour)
    order_hours = [
        datetime.strptime(order["timestamp"], "%Y-%m-%d %H:%M:%S").hour 
        for order in data[username]["food_orders"]
    ]
    best_hour = max(set(order_hours), key=order_hours.count) if order_hours else 12
    
    return jsonify({
        "user": username,
        "best_notification_hour": best_hour
    })
    
    
@app.route("/unused_services/<username>")
def unused_services(username):
    if username not in data:
        return jsonify({"error": "User not found"}), 404
    
    unused = [
        service for service, count in data[username]["category_usage"].items() 
        if count == 0
    ]
    
    return jsonify({
        "user": username,
        "unused_services": unused
    })




if __name__ == "__main__":
    app.run(debug=True)
    
    
    
# # Helper: Get user features (simplified for demo)
# def get_user_features():
#     return {
#         user: [
#             user_data["category_usage"]["food:delivery"],
#             len(user_data["food_orders"]),
#             user_data["location"]["lat"],
#             user_data["location"]["lng"]
#         ]
#         for user, user_data in data.items()
#     }

# # Train a simple k-NN model for recommendations
# user_features = get_user_features()
# users = list(user_features.keys())
# X = np.array(list(user_features.values()))
# model_knn = NearestNeighbors(n_neighbors=3).fit(X)


# @app.route("/recommend/<username>")
# def recommend(username):
#     if username not in users:
#         return jsonify({"error": "User not found"}), 404
    
#     # Find similar users
#     distances, indices = model_knn.kneighbors([user_features[username]])
#     similar_users = [users[i] for i in indices[0]]
    
#     # Recommend dishes from similar users
#     recommended_dishes = set()
#     for user in similar_users:
#         for order in data[user]["food_orders"]:
#             recommended_dishes.add(order["dish"])
    
#     return jsonify({
#         "user": username,
#         "recommendations": list(recommended_dishes),
#         "similar_users": similar_users
#     })
    