from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from collections import Counter
from llama_cpp import Llama

# Load model once globally

# this model takes 15-20 seconds to load
# llm = Llama(model_path="./models/mistral-7b-instruct-v0.1-q4_k_m.gguf", n_ctx=2048)

# this model takes 6-8 seconds to load
llm = Llama(model_path="./models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf", n_ctx=1024)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://flaskapp-6b34b-default-rtdb.firebaseio.com/"}
)

ref = db.reference("/flask_db")
# Example usage
# ref = db.reference('/data')
# ref.set({'hello': 'world user1'})


# openAI
# def build_prompt(user_id, user_data):
#     interests = ", ".join(user_data.get("interests", []))
#     location = user_data.get("location", "Unknown")
#     age = user_data.get("age", "N/A")
#     gender = user_data.get("gender", "N/A")
#     history = user_data.get("purchase_history", [])
#     past_items = ", ".join([f'{h["item"]} ({h["catagory"]})' for h in history])
#     prompt = (
#         f"User Profile:\n"
#         f"- Age: {age}\n"
#         f"- Gender: {gender}\n"
#         f"- Location: {location}\n"
#         f"- Interests: {interests}\n"
#         f"- Past food items: {past_items}\n\n"
#         # f"Based on this, suggest a food item the user might enjoy next, with
#         # a friendly and casual tone."
#         # f"give me answer in one line what is a favorite food"
#         f"Based on the user's profile and food history, what is their favorite food? Respond in one clear sentence only and direct for user./n"
#     )
#     print("promt-----", prompt)

#     return prompt
def get_matching_hotels(location):
    hotels_ref = db.reference("hotels/hotels/")
    print("hotels_ref----------------------", hotels_ref)
    all_hotels = hotels_ref.get()
    return [hotel for hotel in all_hotels.values() if hotel["location"].lower() == location.lower()]


# # this prompt is for mistral-7b-instruct model   
# def build_prompt(user_id, user_data, hotel_matches):
#     interests = ", ".join(user_data.get("interests", []))
#     location = user_data.get("location", "Unknown")
#     age = user_data.get("age", "N/A")
#     gender = user_data.get("gender", "N/A")
#     history = user_data.get("purchase_history", [])

#     past_items = ", ".join([f'{h["item"]} ({h["catagory"]})' for h in history])
#     hotel_lines = "\n".join(
#         [f'- {hotel["name"]} ({hotel["cuisine"]}, {hotel["rating"]}★): {", ".join(hotel["popular_dishes"])}'
#          for hotel in hotel_matches]
#     )   
#     prompt = (
#         f"User Profile:\n"
#         f"- Age: {age}\n"
#         f"- Gender: {gender}\n"
#         f"- Location: {location}\n"
#         f"- Interests: {interests}\n"
#         f"- Past food items: {past_items}\n\n"
#         f"Nearby Hotels & Dishes:\n{hotel_lines}\n\n"
#         f"Based on the user's profile and nearby hotel options, suggest a favorite food and hotel in one line."
#     )
#     return prompt

# this prompt is for tinyllama-1.1b-chat-v1.0-q4_k_m.gguf model
def build_prompt(user_id, user_data, hotel_matches):
    interests = ", ".join(user_data.get("interests", []))
    location = user_data.get("location", "Unknown")
    age = user_data.get("age", "N/A")
    gender = user_data.get("gender", "N/A")
    history = user_data.get("purchase_history", [])

    past_items = ", ".join([f'{h["item"]} ({h["catagory"]})' for h in history])
    hotel_lines = "\n".join(
        [f'- {hotel["name"]} ({hotel["cuisine"]}, {hotel["rating"]}★): {", ".join(hotel["popular_dishes"])}'
         for hotel in hotel_matches]
    )
    
    prompt = (
        # f"You are a food assistant helping users find a dish and hotel they will love.\n\n"
        f"You have to suggest one best food and hotel who have that food.\n\n"
        f"User Info:\n"
        f"- Age: {age}\n"
        f"- Gender: {gender}\n"
        f"- Location: {location}\n"
        f"- Interests: {interests}\n"
        f"- Previously Enjoyed Foods: {past_items}\n\n"
        f"Nearby Hotels:\n{hotel_lines}\n\n"
        f"Suggest ONE best dish and hotel in ONE short sentence.\n"
        f"Format: \"Enjoy a delicious plate of <dish> at <hotel>  and have a very good meal.\""
    )
    return prompt

    # return (
    #     f"User Profile:\n"
    #     f"- Age: {age}\n"
    #     f"- Gender: {gender}\n"
    #     f"- Location: {location}\n"
    #     f"- Interests: {interests}\n"
    #     f"- Past food items: {past_items}\n\n"
    #     f"Nearby Hotels & Dishes:\n{hotel_lines}\n\n"
    #     f"Suggest ONE best dish and hotel in ONE short sentence. Format: '<dish> at <hotel name>.' Nothing else."
    # )


# LLP function
@app.route("/nlp_suggest/<user_id>")
def nlp_suggest_local_llm(user_id):
    user_ref = db.reference(f"users/users/{user_id}")
    user_data = user_ref.get()

    if not user_data:
        return jsonify({"error": "User not found"}), 404

    hotels = get_matching_hotels(user_data.get("location", ""))
    prompt = f"[INST] {build_prompt(user_id, user_data, hotels)} [/INST]"

    try:
        print("start llm==========================")
        # this output is for mistral-7b-instruct model
        # output = llm(prompt, max_tokens=150, stop=["[/INST]"])
        # suggestion = output["choices"][0]["text"].strip()
        
        # this output is for tinyllama-1.1b-chat-v1.0-q4_k_m.gguf model
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that recommends food and hotels direct to the user."},
                {"role": "user", "content": prompt}
                ],
            max_tokens=50,
            stop=["\n"]
            )
        suggestion = output["choices"][0]["message"]["content"].strip()
        print("output----------------------", output)
        return jsonify({"user_id": user_id, "suggestion": suggestion})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# @app.route("/nlp_suggest/<user_id>")
# def nlp_suggest_gemini(user_id):
#     user_ref = db.reference(f"users/users/{user_id}")
#     user_data = user_ref.get()

#     if not user_data:
#         return jsonify({"error": "User not found"}), 404
    
#     # new added for hotel
#     hotels = get_matching_hotels(user_data.get("location", ""))
#     prompt = build_prompt(user_id, user_data, hotels)

#     try:
#         model = genai.GenerativeModel("gemini-2.0-flash")
#         response = model.generate_content(prompt)
#         return jsonify({"user_id": user_id, "suggestion": response.text})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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
                output = llm.create_chat_completion(
                    messages=[
                        
                        {"role": "system", "content": "You are a helpful assistant that recommends food and hotels direct to the user."},
                        {"role": "user", "content": prompt}
                        ],
                    max_tokens=40,
                    stop=["\n"]
                    )
                suggestion = output["choices"][0]["message"]["content"].strip()                
                
                ## this is for mistral-7b-instruct model
                
                # output = llm(prompt, max_tokens=150)
                # suggestion = output["choices"][0]["text"].strip()

                if suggestion:
                    all_suggestions[user_id] = suggestion
                else:
                    all_suggestions[user_id] = "No suggestion generated."
            except Exception as inner_error:
                all_suggestions[user_id] = f"Error generating suggestion: {str(inner_error)}"

        return jsonify(all_suggestions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# @app.route("/nlp_suggest_all")
# def nlp_suggest_all():
#     try:
#         users_ref = db.reference("users/users")
#         users_data = users_ref.get()

#         if not users_data:
#             return jsonify({"error": "No users found"}), 404

#         model = genai.GenerativeModel("gemini-2.0-flash")
#         all_suggestions = {}
#         for user_id, user_data in users_data.items():
#             hotels = get_matching_hotels(user_data.get("location", ""))
#             prompt = build_prompt(user_id, user_data, hotels)
#             if user_id == "user_006":
#                 break
#             try:
#                 print("user", user_id)
#                 response = model.generate_content(prompt)
#                 all_suggestions[user_id] = response.text
#             except Exception as inner_error:
#                 all_suggestions[user_id] = (
#                     f"Error generating suggestion: {str(inner_error)}"
#                 )
#         return jsonify(all_suggestions)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# GET /data – fetch data from Firebase
@app.route("/data", methods=["GET"])
def get_data():
    data = ref.get()
    return jsonify(data), 200


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


# This could come from a database or static file
food_suggestions = {
    "italian": ["pizza", "pasta", "lasagna", "burger", "risotto"],
    "japanese": ["sushi", "ramen", "tempura", "fries", "teriyaki"],
    "indian": ["biryani", "butter chicken", "naan", "samosa", "paneer"],
    "mexican": ["taco", "burrito", "quesadilla", "nachos", "enchilada"],
    "chinese": [
        "noodles",
        "dumplings",
        "sweet and sour pork",
        "spring rolls",
        "chow mein",
    ],
}


def get_favorite_category(purchase_history):
    categories = [entry["catagory"] for entry in purchase_history]
    most_common = Counter(categories).most_common(1)
    return most_common[0][0] if most_common else None


def suggest_items(favorite_cat, purchase_history):
    tried_items = {entry["item"] for entry in purchase_history}
    possible_items = food_suggestions.get(favorite_cat, [])
    return [item for item in possible_items if item not in tried_items]


@app.route("/suggest/<user_id>")
def suggest_for_user(user_id):
    ref = db.reference(f"users/{user_id}")
    user_data = ref.get()

    if not user_data:
        return jsonify({"error": "User not found"}), 404

    purchase_history = user_data.get("purchase_history", [])
    favorite_cat = get_favorite_category(purchase_history)
    suggestions = suggest_items(favorite_cat, purchase_history)

    return jsonify(
        {
            "user_id": user_id,
            "favorite_category": favorite_cat,
            "suggestions": suggestions,
        }
    )


@app.route("/suggest_all")
def suggest_all():
    users_ref = db.reference("users")
    all_users = users_ref.get()

    if not all_users:
        return jsonify({"error": "No users found"}), 404

    results = {}

    for user_id, user_data in all_users.items():
        purchase_history = user_data.get("purchase_history", [])
        favorite_cat = get_favorite_category(purchase_history)
        suggestions = suggest_items(favorite_cat, purchase_history)

        results[user_id] = {
            "favorite_category": favorite_cat,
            "suggestions": suggestions,
        }

    return jsonify(results)

from flask import request, jsonify

@app.route("/ask-question", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        print("Processing question:", question)

        # Format prompt (add system prompt or wrap if needed)
        # prompt = f"[INST] {question} [/INST]"
        # prompt = f"Answer the following question clearly:\n\nQ: {question}\nA:"
        prompt = f"Answer the following question clearly:\n\nQ: {question}\nA:"

        print("Prompt========:", prompt)

        response = llm(prompt, max_tokens=150, stop=["</s>"])
        
        print("Response========:", response)

        # Extract response text
        answer = response["choices"][0]["text"].strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Replace values with your actual MySQL credentials
# app.config['SQLALCHEMY_DATABASE_URI'] =
# 'mysql+pymysql://root:admin@localhost/flask_app_db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

#   to create a model in database use this in terminal
# python
# with app.app_context():
#    db.create_all()

# exit()


# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100))

@app.route("/site-user-data", methods=["POST"])
def site_user_data():
    content = request.json
    if not content:
        return jsonify({"error": "No data provided"}), 400

    ref = db.reference("site-user")

    # If it's a list of items, loop through and push each
    if isinstance(content, list):
        for item in content:
            ref.push(item)
    else:
        # If it's just one item, push it directly
        ref.push(content)

    return jsonify({"message": "Data saved"}), 201


@app.route("/")
def home():
    return "Hello, Flask on Windows! "


@app.route("/home")
def home1():
    return render_template("index.html")


# # GET all users
# @app.route("/api/users", methods=["GET"])
# def get_users():
#     users = User.query.all()
#     return jsonify([{"id": u.id, "name": u.name} for u in users])

# # POST a new user
# @app.route("/api/users", methods=["POST"])
# def add_user():
#     data = request.get_json()
#     name = data.get("name")

#     if not name:
#         return jsonify({"error": "Name is required"}), 400

#     user = User(name=name)
#     db.session.add(user)
#     db.session.commit()

#     return jsonify({"id": user.id, "name": user.name}), 201


if __name__ == "__main__":
    app.run(debug=True)
