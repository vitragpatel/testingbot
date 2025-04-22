import math
from firebase_admin import credentials, db
import json
import os

# --- Haversine formula for distance calculation ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# --- Find nearby hotels and append distance ---
def get_matching_hotels(user_lat, user_lon, max_distance_km=5):
    hotels_ref = db.reference("hotels/hotels")
    all_hotels = hotels_ref.get()

    if not all_hotels:
        return []

    matching_hotels = []
    for hotel in all_hotels.values():
        h_lat = hotel.get("latitude")
        h_lon = hotel.get("longitude")
        if h_lat is not None and h_lon is not None:
            distance = haversine(user_lat, user_lon, h_lat, h_lon)
            if distance <= max_distance_km:
                hotel["distance_km"] = round(distance, 2)
                matching_hotels.append(hotel)

    # Sort by distance ascending
    return sorted(matching_hotels, key=lambda h: h["distance_km"])


def data_processing():
    DATA_PATH = "data.json"
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        exit()
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            hotels_data = data.get("hotels_data", {})
            # users_data = data.get("users_data", {})
            site_user_data = data.get("site_user_data", {})
            print(f"Loaded data for {len(hotels_data)} hotels and site_user_data {len(site_user_data)}.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {DATA_PATH}")
        exit()



# #This is a old code for the llm model
# def get_matching_hotels(location):
#     hotels_ref = db.reference("hotels/hotels/")
#     all_hotels = hotels_ref.get()
#     return [hotel for hotel in all_hotels.values() if hotel["location"].lower() == location.lower()]


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
#         # f"You are a food assistant helping users find a dish and hotel they will love.\n\n"
#         f"You have to suggest one best food and hotel who have that food.\n\n"
#         f"User Info:\n"
#         f"- Age: {age}\n"
#         f"- Gender: {gender}\n"
#         f"- Location: {location}\n"
#         f"- Interests: {interests}\n"
#         f"- Previously Enjoyed Foods: {past_items}\n\n"
#         f"Nearby Hotels:\n{hotel_lines}\n\n"
#         f"Suggest ONE best dish and hotel in ONE short sentence.\n"
#         f"Format: \"Enjoy a delicious plate of <dish> at <hotel>  and have a very good meal.\""
#     )
#     return prompt


# # LLP function
# @app.route("/nlp_suggest/<user_id>")
# def nlp_suggest_local_llm(user_id):
#     user_ref = db.reference(f"users/users/{user_id}")
#     user_data = user_ref.get()

#     if not user_data:
#         return jsonify({"error": "User not found"}), 404

#     hotels = get_matching_hotels(user_data.get("location", ""))
#     prompt = f"[INST] {build_prompt(user_id, user_data, hotels)} [/INST]"

#     try:
#         llm = llm_init()
#         output = llm.create_chat_completion(
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that recommends food and hotels direct to the user."},
#                 {"role": "user", "content": prompt}
#                 ],
#             max_tokens=50,
#             stop=["\n"]
#             )
#         suggestion = output["choices"][0]["message"]["content"].strip()
#         return jsonify({"user_id": user_id, "suggestion": suggestion})
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500