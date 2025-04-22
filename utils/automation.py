from utils.get_data import get_site_user_data, get_hotel_data
from utils.al_prompt import *
import json
from collections import defaultdict
import json

def automation_function_step1():
    # Step 1: Fetch data
    site_users = get_site_user_data()
    hotels = get_hotel_data()

    # Step 2: Combine into one dict
    data = {
        "site_users": site_users,
        "hotels": hotels
    }

    # Step 3: Write to JSON file
    with open("firebase_data_dump.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    return {
        "message": "Automation step 1 completed. Data saved to firebase_data_dump.json"
    }
    
    
# def automation_function_step2():
#     # Step 1: Load data
#     with open("firebase_data_dump.json", "r") as json_file:
#         data = json.load(json_file)

#     site_users = data.get("site_users", {})

#     # Step 2: Analysis containers
#     type_counts = defaultdict(int)
#     category_counts = defaultdict(int)
#     subcategory_details = defaultdict(list)

#     for user_id, user_data in site_users.items():
#         category = user_data.get("category", "unknown")  # e.g., "handiman:carpenter"
#         base_type = category.split(":")[0]  # "handiman"
#         full_category = category  # full subcategory like "handiman:carpenter"

#         # Count by base type
#         type_counts[base_type] += 1

#         # Count by full subcategory
#         category_counts[full_category] += 1

#         # Add details for reference
#         subcategory_details[full_category].append({
#             "name": user_data.get("name", "unknown"),
#             "contactNo": user_data.get("contactNo", ""),
#             "location": user_data.get("location", ""),
#             "latitude": user_data.get("latitude"),
#             "longitude": user_data.get("longitude")
#         })

#     # Step 3: Create structured summary
#     summary = {
#         "total_users": len(site_users),
#         "user_types_distribution": dict(type_counts),
#         "user_subcategories_distribution": dict(category_counts),
#         "subcategory_details": subcategory_details
#     }


#     return summary

    # Simulate mock visits for each user (replace with actual logic later)


def automation_function_step2_userwise():
    try:
        with open("firebase_data_dump.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return {"error": "Data file not found"}

    site_users = data.get("site_users", {})
    user_service_counts = defaultdict(lambda: defaultdict(int))
    all_services = set()  # Track all possible services

    # First pass: Collect all services and user counts
    for user_id, user_data in site_users.items():
        name = user_data.get("name", "Unknown")
        category = user_data.get("category")
        if category:
            user_service_counts[name][category] += 1
            all_services.add(category)

    # Second pass: Ensure every user has all services, even if 0
    user_service_list = {}
    for user, categories in user_service_counts.items():
        # Initialize with all services set to 0
        user_services = {service: 0 for service in all_services}
        # Update with actual counts
        user_services.update(categories)
        user_service_list[user] = user_services

    # Save to file
    with open("user_service_counts.json", "w") as json_file:
        json.dump(user_service_list, json_file, indent=4)

    return user_service_list
    # return "user_service_count_list.json created successfully."

    


def automation_function_step3(data):
    llm = llm_init()
    result = {}

    for user, services in data.items():
        prompt = build_user_service_data_prompt(user, services)
        
        print("-----------prompt------------",prompt)

        messages = [
            {"role": "system", "content": "You are an assistant that identifies the service category a user is most interested in."},
            {"role": "user", "content": prompt}
        ]

        response = llm.create_chat_completion(messages)
        content = response["choices"][0]["message"]["content"].strip()

        # Keep only the first non-empty line
        result[user] = content.split('\n')[0]

    return result
