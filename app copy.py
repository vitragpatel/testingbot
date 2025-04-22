from flask import Flask, render_template, jsonify, request
from llama_cpp import Llama
import json
from flask import Flask, request, jsonify
from llama_cpp import Llama

# --- Configuration ---
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"
DATA_PATH = "data.json"
N_CTX = 1024  # Match your LLM context size
MAX_TOKENS = 250  # Max tokens for LLM response

# --- Load Data ---
try:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    hotels_data = data.get("hotels_data", {})
    users_data = data.get("users_data", {})
    print(f"Loaded data for {len(hotels_data)} hotels and {len(users_data)} users.")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {DATA_PATH}")
    exit()

# --- Initialize LLM ---
try:
    # Adjust params as needed (n_gpu_layers, etc.)
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, verbose=False)
    print("LLM loaded successfully.")
except Exception as e:
    print(f"Error loading LLM: {e}")
    exit()

# --- Initialize Flask App ---
app = Flask(__name__)


# --- Helper Function for LLM Interaction ---
def ask_llm(prompt):
    """Sends a prompt to the LLM and returns the response."""
    try:
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            stop=["\n", "User:", "Assistant:"],  # Adjust stop tokens if needed
            echo=False,
        )
        # Extract the text part of the response
        response_text = output["choices"][0]["text"].strip()
        return response_text
    except Exception as e:
        print(f"Error during LLM inference: {e}")
        return "Sorry, I encountered an error trying to process your request."


# --- API Endpoints ---

@app.route("/")
def home():
    return "Welcome to the Hotel & User Data API!"


@app.route("/user_summary/<user_id>", methods=["GET"])
def get_user_summary(user_id):
    """Generates a natural language summary of user preferences."""
    print(f"Generating summary for user: {user_id}")
    user = users_data.get(user_id)
    print(f"User data: {user}")
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Prepare context for LLM
    interests = ", ".join(user.get("interests", []))
    purchase_summary = []
    for purchase in user.get("purchase_history", []):
        purchase_summary.append(f"{purchase['item']} ({purchase['catagory']})")

    # Limit history length to avoid exceeding context window
    history_text = ", ".join(purchase_summary[:10])  # Take last 10 items

    # Construct the prompt
    prompt = f"""
User Profile:
Interests: {interests}
Recent Purchases: {history_text}
Location: {user.get('location', 'Unknown')}
Age: {user.get('age', 'Unknown')}
Gender: {user.get('gender', 'Unknown')}

Task: Based ONLY on the User Profile provided above, write a brief, friendly summary (2-3 sentences) describing this user's likely food preferences or dining habits. Do not invent information not present in the profile.

Summary:
"""
    # Get summary from LLM
    summary = ask_llm(prompt)
    return jsonify({"user_id": user_id, "summary": summary})


@app.route("/recommendations/<user_id>", methods=["GET"])
def get_recommendations(user_id):
    """Provides hotel recommendations based on user profile."""
    user = users_data.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    num_recs = request.args.get(
        "count", 3, type=int
    )  # Allow specifying number of recommendations

    # --- Simple Recommendation Logic (can be enhanced) ---
    # 1. Find user's most frequent cuisine categories
    cuisine_counts = {}
    for purchase in user.get("purchase_history", []):
        cat = purchase.get("catagory", "").lower()
        cuisine_counts[cat] = cuisine_counts.get(cat, 0) + 1

    # Sort cuisines by frequency
    sorted_cuisines = sorted(
        cuisine_counts.items(), key=lambda item: item[1], reverse=True
    )
    top_cuisines = [c[0] for c in sorted_cuisines[:2]]  # Top 2 cuisines

    # 2. Filter hotels based on top cuisines (limit hotel data for prompt)
    potential_hotels = []
    for hotel_id, hotel_info in hotels_data.items():
        if hotel_info.get("cuisine", "").lower() in top_cuisines:
            # Limit details sent to LLM to conserve context
            potential_hotels.append(
                {
                    "id": hotel_id,
                    "name": hotel_info.get("name"),
                    "cuisine": hotel_info.get("cuisine"),
                    "location": hotel_info.get("location"),
                    "rating": hotel_info.get("rating"),
                    "popular_dishes": hotel_info.get("popular_dishes", [])[
                        :2
                    ],  # Just first 2 popular dishes
                }
            )

    # Limit number of hotels passed to LLM
    hotels_for_prompt = potential_hotels[:10]

    # 3. Construct the prompt for the LLM
    interests = ", ".join(user.get("interests", []))
    history_summary = ", ".join(
        [f"{p['item']} ({p['catagory']})" for p in user.get("purchase_history", [])[:5]]
    )

    prompt = f"""
User Profile:
User ID: {user_id}
Interests: {interests}
Preferred Cuisines (from history): {', '.join(top_cuisines) if top_cuisines else 'None specified'}
Recent Purchases Sample: {history_summary}

Available Hotels (subset):
{json.dumps(hotels_for_prompt, indent=2)}

Task: Based ONLY on the User Profile and the Available Hotels list provided, recommend {num_recs} hotels for this user. For each recommendation, provide the hotel name and a very brief (1 sentence) justification based on the user's profile (e.g., matching cuisine preference, interests). Do not recommend hotels not in the list. Format the output clearly.

Recommendations:
"""
    # Get recommendations from LLM
    recommendations_text = ask_llm(prompt)

    # Basic parsing (you might need more robust parsing)
    # This is simple, LLM might format differently
    recs = []
    lines = recommendations_text.split("\n")
    current_rec = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Very simple parsing - assumes "Hotel Name:" and "Justification:" prefixes or similar
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip().lower()
            value = parts[1].strip()
            if "hotel" in key or "name" in key:
                if current_rec:  # Save previous before starting new
                    recs.append(current_rec)
                current_rec = {"name": value}
            elif "justification" in key or "reason" in key:
                if current_rec:
                    current_rec["justification"] = value

    if current_rec and "name" in current_rec:  # Add the last one
        recs.append(current_rec)

    # Fallback if parsing fails
    if not recs and recommendations_text:
        recs = [{"raw_recommendation": recommendations_text}]

    return jsonify({"user_id": user_id, "recommendations": recs[:num_recs]})


@app.route("/search_hotels", methods=["POST"])
def search_hotels_nl():
    """Searches hotels based on a natural language query."""
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    # --- Approach 1: LLM extracts criteria --- (More reliable for smaller LLMs)
    prompt_extract = f"""
        User Query: "{query}"

        Available Hotel Attributes: cuisine, location, name, popular_dishes (list), rating (number).

        Task: Analyze the User Query and extract key filtering criteria based ONLY on the query text and the Available Hotel Attributes. Output the criteria as a JSON object. If a criterion is not mentioned, omit it. Be strict about matching attributes. For example:
        - If query mentions "Indian food in Mumbai", output: {{"cuisine": "Indian", "location": "Mumbai"}}
        - If query mentions "highly rated sushi place", output: {{"cuisine": "Japanese", "popular_dishes": ["Sushi"], "rating_min": 4.5}} (Assume 'highly rated' means >= 4.5)
        - If query mentions "Pasta Paradise", output: {{"name": "Pasta Paradise"}}

        Extracted Criteria (JSON):
        """
    criteria_str = ask_llm(prompt_extract)

    try:
        # Clean up potential markdown code blocks
        criteria_str = criteria_str.replace("```json", "").replace("```", "").strip()
        criteria = json.loads(criteria_str)
        if not isinstance(criteria, dict):  # Ensure it's a dictionary
            raise ValueError("LLM did not return a valid JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing criteria from LLM: {e}\nLLM Output: {criteria_str}")
        # Fallback: Maybe ask LLM to perform the search directly (less reliable)
        # Or return an error indicating parsing failure
        return (
            jsonify(
                {
                    "error": "Could not understand search criteria from query.",
                    "llm_output": criteria_str,
                }
            ),
            500,
        )

    # --- Filter hotels using extracted criteria (Python code) ---
    results = []
    for hotel_id, hotel_info in hotels_data.items():
        match = True
        for key, value in criteria.items():
            hotel_value = hotel_info.get(key)
            if hotel_value is None:
                match = False
                break

            # Handle different types/comparisons
            if (
                key == "cuisine"
                and isinstance(value, str)
                and isinstance(hotel_value, str)
            ):
                if value.lower() not in hotel_value.lower():
                    match = False
                    break
            elif (
                key == "location"
                and isinstance(value, str)
                and isinstance(hotel_value, str)
            ):
                if value.lower() not in hotel_value.lower():
                    match = False
                    break
            elif (
                key == "name"
                and isinstance(value, str)
                and isinstance(hotel_value, str)
            ):
                if value.lower() not in hotel_value.lower():
                    match = False
                    break
            elif key == "rating_min" and isinstance(value, (int, float)):
                if not (isinstance(hotel_value, (int, float)) and hotel_value >= value):
                    match = False
                    break
            elif (
                key == "popular_dishes"
                and isinstance(value, list)
                and isinstance(hotel_value, list)
            ):
                # Check if *all* requested dishes are in the popular list
                hotel_dishes_lower = [d.lower() for d in hotel_value]
                if not all(
                    req_dish.lower() in hotel_dishes_lower for req_dish in value
                ):
                    match = False
                    break
            # Add more specific checks if needed
            elif isinstance(value, str) and isinstance(
                hotel_value, str
            ):  # Generic string check
                if value.lower() not in hotel_value.lower():
                    match = False
                    break
            elif hotel_value != value:  # Generic equality check
                match = False
                break

        if match:
            # Add hotel_id to the result dict
            result_hotel = hotel_info.copy()
            result_hotel["hotel_id"] = hotel_id
            results.append(result_hotel)

    return jsonify({"query": query, "extracted_criteria": criteria, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
