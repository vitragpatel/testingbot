from llama_cpp import Llama


def llm_init():
    return Llama(model_path="./models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf",
                 n_gpu_layers=20,
                 seed=42,
                 mirostat=2,
                 mirostat_tau=5.0,
                 mirostat_eta=0.1,
                 repeat_penalty=1.1,
                 n_ctx=1024,
                 n_batch=512,
                 n_threads=8,
                #  n_predict=128,
                 n_predict=512,
                 top_k=40,
                 top_p=0.95,
                 temp=0.8,
                 repeat_last_n=64,
                 stop=["</s>"],
                 streaming=True,
                 use_mlock=True,
                 use_mmap=True,
                 use_kv_memory=True,
                 verbose=True)


# --- Build LLM prompt with distance included ---
def build_prompt(user_name, user_data, hotel_matches):
    location = user_data.get("location", "Unknown")

    # hotel_lines = "\n".join(
    #     [f'- {hotel["name"]} ({hotel["cuisine"]}, {hotel["rating"]}★, {hotel["distance_km"]} km away): '
    #      f'{", ".join(hotel["popular_dishes"])}'
    #      for hotel in hotel_matches]
    # )
    hotel_lines = "\n".join(
        [f'- {hotel["name"]} ({hotel["cuisine"]}, {hotel["rating"]}★, {hotel["distance_km"]} km away): {", ".join(hotel["popular_dishes"])}'
         for hotel in hotel_matches[:5]]  # Only include top 5 hotels
        )

    prompt = (
        f"You have to suggest one best food and hotel who have that food.\n\n"
        f"User Info:\n"
        f"- Name: {user_name}\n"
        f"- Location: {location}\n\n"
        f"Nearby Hotels:\n{hotel_lines}\n\n"
        f"Suggest ONE best dish and hotel in ONE short sentence.\n"
        f'Format: "Enjoy a delicious plate of <dish> at <hotel> and have a very good meal."'
    )
    return prompt


def build_user_service_data_prompt(user, services):
    prompt = f"Here are the services used by {user}:\n\n"
    for category, count in services.items():
        prompt += f"- {category}: {count}\n"
    
    prompt += f"\nWhich category is {user} never used? Return only the category name."
    # prompt += f"\nWhich category is {user} most interested in? Return only the category name."
    return prompt