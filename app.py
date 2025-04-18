from flask import Flask, render_template, jsonify, request
from llama_cpp import Llama


# this model takes 6-8 seconds to load
llm = Llama(model_path="./models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf", n_ctx=1024)

app = Flask(__name__)

from flask import request, jsonify

@app.route("/ask")
def ask():
    return render_template("ask_question.html")

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


@app.route("/")
def home():
    return "Hello, Flask on Windows! "


@app.route("/home")
def home1():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
