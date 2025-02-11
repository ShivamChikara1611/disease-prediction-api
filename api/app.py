from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Compute the absolute path for the model file relative to this file's location
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(base_dir, "models", "disease_prediction_model.pkl")

# Load the trained model
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    logging.info("Model loaded successfully from %s", model_path)
except Exception as e:
    logging.error("Error loading model: %s", str(e))
    model = None

# List of required symptom columns (with duplicates removed)
REQUIRED_COLUMNS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", 
    "spotting_ urination", "fatigue", "weight_gain", "anxiety", 
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", 
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", 
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", 
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea", 
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", 
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", 
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", 
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise", 
    "blurred_and_distorted_vision", "phlegm", "throat_irritation", 
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", 
    "chest_pain", "weakness_in_limbs", "fast_heart_rate", 
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", 
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", 
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties", 
    "excessive_hunger", "extra_marital_contacts", 
    "drying_and_tingling_lips", "slurred_speech", "knee_pain", "hip_joint_pain", 
    "muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", 
    "spinning_movements", "loss_of_balance", "unsteadiness", 
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", 
    "foul_smell_of urine", "continuous_feel_of_urine", "passage_of_gases", 
    "internal_itching", "toxic_look_(typhos)", "depression", "irritability", 
    "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain", 
    "abnormal_menstruation", "dischromic _patches", "watering_from_eyes", 
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", 
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", 
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", 
    "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", 
    "blood_in_sputum", "prominent_veins_on_calf", 
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", 
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", 
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Disease Prediction API. Use POST /predict with a valid JSON payload to get a prediction."
    })

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        # Expect a JSON payload with symptom features
        input_data = request.get_json(force=True)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Input data must be a JSON object."}), 400

        # Validate that all required symptom columns are provided
        missing = [col for col in REQUIRED_COLUMNS if col not in input_data]
        if missing:
            return jsonify({"error": "Missing required symptom(s): " + ", ".join(missing)}), 400

        # Validate that each provided value is either 0 or 1.
        invalid_columns = []
        for col in REQUIRED_COLUMNS:
            try:
                # Attempt to cast the value to an integer
                val = int(input_data[col])
                if val not in (0, 1):
                    invalid_columns.append(col)
            except (ValueError, TypeError):
                invalid_columns.append(col)
        if invalid_columns:
            return jsonify({
                "error": "Invalid value for symptom(s): " + ", ".join(invalid_columns) +
                        ". Only 0 or 1 are allowed."
            }), 400

        # Create a DataFrame with the validated values in the correct order
        df = pd.DataFrame([{col: int(input_data[col]) for col in REQUIRED_COLUMNS}])
        logging.info("Input DataFrame shape: %s, columns: %s", df.shape, df.columns.tolist())

        # Get the prediction from the model
        prediction = model.predict(df)
        return jsonify({"prognosis": prediction[0]})
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
