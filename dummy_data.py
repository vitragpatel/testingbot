import json
import random
from datetime import datetime, timedelta

# run this command to generate dummy data
# python generate_dummy_data.py
# python dummy_data.py

# Mock services
services = [
    "food:delivery", "food:dinein", "hotel", "transport", 
    "emergency", "club", "govt", "handiman:electrician", "casino", "handiman:carpenter",
    "handiman:ac services", "handiman:refrigerator services", "handiman:plumber",
]

def random_timestamp():
    return (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S")

def generate_users(num_users=1000):
    users = {}
    for user_id in range(1, num_users + 1):
        name = f"user_{user_id}"
        category_usage = {service: random.randint(0, 10) for service in services}

        user_data = {
            "name": name,
            "category_usage": category_usage,
            "food_orders": [
                {
                    "dish": random.choice(["pizza", "burger", "sushi", "tacos", "pasta", "ramen", "biryani", "shawarma", "noodles", "dumplings"]),
                    "timestamp": random_timestamp()
                }
                for _ in range(category_usage["food:delivery"])
            ],
            "hotel_bookings": [
                {
                    "hotel_name": random.choice(["Hotel Royal Palace", "Sunrise Inn", "Ocean View", "City Stay", "Comfort Suites"]),
                    "check_in": (datetime.now() - timedelta(days=i*2)).strftime("%Y-%m-%d"),
                    "check_out": (datetime.now() - timedelta(days=i*2 - 1)).strftime("%Y-%m-%d")
                }
                for i in range(category_usage["hotel"])
            ],
            "transport_usage": [
                {
                    "mode": random.choice(["auto-rickshaw", "cab", "bus", "bike", "metro"]),
                    "timestamp": random_timestamp()
                }
                for _ in range(category_usage["transport"])
            ],
            "emergency_visits": [
                {
                    "type": random.choice(["hospital", "police", "fire station"]),
                    "timestamp": random_timestamp()
                }
                for _ in range(category_usage["emergency"])
            ],
            "govt_services_used": [
                {
                    "service": random.choice(["passport", "ration card", "Aadhar update", "license renewal"]),
                    "timestamp": random_timestamp()
                }
                for _ in range(category_usage["govt"])
            ],
            "casino_visits": [
                {
                    "casino_name": random.choice(["The Golden Chip", "Surat High Stakes", "Luck Lounge"]),
                    "timestamp": random_timestamp()
                }
                for _ in range(category_usage["casino"])
            ],
            "handiman_services": {
                "electrician": [
                    {
                        "issue": random.choice(["fan not working", "short circuit", "bulb replacement"]),
                        "timestamp": random_timestamp()
                    }
                    for _ in range(category_usage["handiman:electrician"])
                ],
                "carpenter": [
                    {
                        "issue": random.choice(["broken chair", "table fix", "wardrobe door jammed"]),
                        "timestamp": random_timestamp()
                    }
                    for _ in range(category_usage["handiman:carpenter"])
                ],
                "ac services": [
                    {
                        "issue": random.choice(["AC not cooling", "AC gas refill", "filter cleaning"]),
                        "timestamp": random_timestamp()
                    }
                    for _ in range(category_usage["handiman:ac services"])
                ],
                "refrigerator services": [
                    {
                        "issue": random.choice(["fridge not cooling", "ice formation", "compressor noise"]),
                        "timestamp": random_timestamp()
                    }
                    for _ in range(category_usage["handiman:refrigerator services"])
                ],
                "plumber": [
                    {
                        "issue": random.choice(["leaking tap", "clogged drain", "pipe burst"]),
                        "timestamp": random_timestamp()
                    }
                    for _ in range(category_usage["handiman:plumber"])
                ]
            },
            "location": {
                "lat": random.uniform(12.0, 13.0),
                "lng": random.uniform(77.0, 78.0)
            }
        }

        users[name] = user_data

    return users


## this function is used to create dummy data for 10k users

# def generate_users(num_users=2000):
#     users = {}
#     for user_id in range(1, num_users + 1):
#         name = f"user_{user_id}"
#         users[name] = {
#             "name": name,
#             "category_usage": {service: random.randint(0, 10) for service in services},
#             "food_orders": [
#                 {
#                     "dish": random.choice(["pizza", "burger", "sushi", "tacos", "pasta", "ramen", "biryani", "shawarma", "noodles", "dumplings"]), 
#                     "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d %H:%M:%S")
#                 }
#                 for _ in range(random.randint(1, 20))
#             ],
#             "location": {
#                 "lat": random.uniform(12.0, 13.0),
#                 "lng": random.uniform(77.0, 78.0)
#             }
#         }
#     return users

# Save to JSON
with open("dummy_data_all_details.json", "w") as f:
    json.dump({"site_users": generate_users()}, f, indent=4)