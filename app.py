from flask import Flask, request, jsonify
import numpy as np
import joblib
import requests
from pathlib import Path
import requests


# ============================================================
# 1. PROJECT ROOT
# ============================================================

project_root = Path(__file__).resolve().parent.parent


# ============================================================
# 2. FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder="../frontend",
    static_url_path=""
)


# ============================================================
# 3. MODEL PATHS
# ============================================================

model_path = project_root / "model" / "crop_model.pkl"
minmax_path = project_root / "model" / "minmax_scaler.pkl"
standard_path = project_root / "model" / "standard_scaler.pkl"


# ============================================================
# 4. LOAD MACHINE LEARNING MODEL
# ============================================================

try:

    model = joblib.load(model_path)

    minmax_scaler = joblib.load(minmax_path)

    standard_scaler = joblib.load(standard_path)

    print("ML model and scalers loaded successfully!")

except Exception as e:

    print("ERROR loading ML model:")
    print(e)

    model = None
    minmax_scaler = None
    standard_scaler = None


# ============================================================
# 5. CROP DICTIONARY
# ============================================================

crop_dict = {
    1: "Rice",
    2: "Maize",
    3: "Jute",
    4: "Cotton",
    5: "Coconut",
    6: "Papaya",
    7: "Orange",
    8: "Apple",
    9: "Muskmelon",
    10: "Watermelon",
    11: "Grapes",
    12: "Mango",
    13: "Banana",
    14: "Pomegranate",
    15: "Lentil",
    16: "Blackgram",
    17: "Mungbean",
    18: "Mothbeans",
    19: "Pigeonpeas",
    20: "Kidneybeans",
    21: "Chickpea",
    22: "Coffee"
}


# ============================================================
# 6. HOME PAGE
# ============================================================

@app.route("/")
def home():

    return app.send_static_file("index.html")


# ============================================================
# 7. CROP PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": "ML model is not loaded."
        }), 500

    try:
        data = request.get_json()

        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])

        features = np.array([
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]).reshape(1, -1)

        # Apply the same preprocessing used during training
        features = minmax_scaler.transform(features)
        features = standard_scaler.transform(features)

        # Predict crop
        prediction = model.predict(features)

        # Model already returns crop name
        crop = str(prediction[0])

        return jsonify({
            "success": True,
            "crop": crop,
            "message": f"{crop} is the recommended crop."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Invalid numerical value: {str(e)}"
        }), 400

        # Check fields
        for field in required_fields:

            if field not in data:

                return jsonify({
                    "success": False,
                    "error": f"Missing field: {field}"
                }), 400


            if data[field] == "":

                return jsonify({
                    "success": False,
                    "error": f"{field} is empty"
                }), 400


        # Convert values
        N = float(data["N"])

        P = float(data["P"])

        K = float(data["K"])

        temperature = float(
            data["temperature"]
        )

        humidity = float(
            data["humidity"]
        )

        ph = float(
            data["ph"]
        )

        rainfall = float(
            data["rainfall"]
        )


        # Create input array
        features = np.array([
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]).reshape(1, -1)


        print("Input features:")
        print(features)


        # Check model
        if model is None:

            return jsonify({
                "success": False,
                "error": "ML model is not loaded."
            }), 500


        # ====================================================
        # PREPROCESSING
        # ====================================================

        minmax_features = minmax_scaler.transform(
            features
        )

        scaled_features = standard_scaler.transform(
            minmax_features
        )


        # ====================================================
        # RANDOM FOREST PREDICTION
        # ====================================================

        prediction = model.predict(
            scaled_features
        )


        crop_number = int(prediction[0])


        crop = crop_dict.get(
            crop_number,
            "Unknown"
        )


        print("Predicted crop:", crop)


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "success": True,

            "crop": crop,

            "message":
                f"{crop} is the recommended crop."

        })


    except ValueError as e:

        print("VALUE ERROR:", e)

        return jsonify({

            "success": False,

            "error":
                f"Invalid numerical value: {str(e)}"

        }), 400


    except Exception as e:

        print("PREDICTION ERROR:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# 8. OLLAMA CHATBOT API
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        print("\n================================")
        print("OLLAMA CHAT REQUEST")
        print("================================")


        # Get JSON
        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No message received."

            }), 400


        # Get user message
        user_message = data.get(
            "message",
            ""
        ).strip()


        if not user_message:

            return jsonify({

                "success": False,

                "error":
                    "Please enter a message."

            }), 400


        print("User message:")
        print(user_message)


        # ====================================================
        # OLLAMA API
        # ====================================================

        ollama_url = (
            "http://localhost:11434/api/chat"
        )


        payload = {

            "model": "llama3.2",

            "messages": [

                {
                    "role": "system",

                    "content": """
You are an AI Agriculture Assistant
for an AI Crop Recommendation System.

Help farmers and users with:

- Crop recommendation
- Soil conditions
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall
- Crop cultivation
- Agriculture concepts
- Machine learning concepts related to this project

Give simple, clear and useful answers.

The crop recommendation system uses
a Random Forest classification model
for crop prediction.

Do not claim that you personally tested
the user's soil.

Do not provide dangerous or highly
specific chemical pesticide instructions.
"""
                },

                {
                    "role": "user",

                    "content": user_message
                }

            ],

            "stream": False
        }


        # Send request to Ollama
        response = requests.post(

            ollama_url,

            json=payload,

            timeout=120

        )


        print("Ollama status:",
              response.status_code)


        # ====================================================
        # CHECK OLLAMA RESPONSE
        # ====================================================

        if response.status_code != 200:

            return jsonify({

                "success": False,

                "error":
                    f"Ollama error: {response.text}"

            }), 500


        result = response.json()


        # Get AI response
        answer = result["message"]["content"]


        print("AI response received.")


        # ====================================================
        # RETURN AI RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "answer": answer

        })


    except requests.exceptions.ConnectionError:

        return jsonify({

            "success": False,

            "error":
                "Cannot connect to Ollama. "
                "Please make sure Ollama is installed "
                "and running."

        }), 500


    except requests.exceptions.Timeout:

        return jsonify({

            "success": False,

            "error":
                "Ollama took too long to respond."

        }), 500


    except KeyError:

        return jsonify({

            "success": False,

            "error":
                "Unexpected response from Ollama."

        }), 500


    except Exception as e:

        print("CHAT ERROR:", e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# 9. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("     AI CROP RECOMMENDATION SYSTEM")
    print("==========================================")
    print("ML Model : Random Forest")
    print("AI Model : Ollama + Llama 3.2")
    print("Server   : http://127.0.0.1:5000")
    print("==========================================")
    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )