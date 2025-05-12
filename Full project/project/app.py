import os
# Set TensorFlow environment variables before importing TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from collections import Counter
from scipy.optimize import linprog
from pymongo import MongoClient
from functools import wraps
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import io
import base64

# Initialize Flask app
app = Flask(__name__)

# Set secret key for session-based flash messaging
app.secret_key = 'your_secret_key_here'

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
mongo_db = client['feedbackDB']
feedback_collection = mongo_db['feedbacks']
users_collection = mongo_db['users']
disease_history_collection = mongo_db['disease_history']

# Load dataset (optional, not used in prediction directly)
# crop_data = pd.read_csv('C:/Users/jaina/OneDrive/Desktop/1222/project/Data-processed/crop_recommendation.csv')

# States and cities
states_and_cities = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Nellore"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzzafarpur", "Purnia"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangalore", "Belagavi"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Trichy", "Salem"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur", "Kota", "Ajmer"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Punjab": ["Chandigarh", "Amritsar", "Ludhiana", "Jalandhar", "Patiala"],
    "Haryana": ["Chandigarh", "Faridabad", "Gurugram", "Ambala", "Hisar"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Kottayam", "Thrissur"],
    "Delhi": ["New Delhi", "Dwarka", "Vasant Kunj", "Connaught Place", "Saket"],
    "Uttarakhand": ["Dehradun", "Nainital", "Haridwar", "Rishikesh", "Roorkee"],
    "Himachal Pradesh": ["Shimla", "Manali", "Kullu", "Dharamshala", "Kangra"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Korba", "Raigarh"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"],
    "Assam": ["Guwahati", "Dibrugarh", "Jorhat", "Silchar", "Tezpur"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Hazaribagh", "Deoghar"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda"],
    "Telangana": ["Hyderabad", "Warangal", "Khammam", "Nizamabad", "Karimnagar"],
    "Andaman and Nicobar Islands": ["Port Blair", "Car Nicobar", "Mayabunder", "Diglipur", "Hut Bay"],
    "Lakshadweep": ["Kavaratti", "Agatti", "Amini", "Kadmat", "Kalapeni"],
    "Sikkim": ["Gangtok", "Mangan", "Namchi", "Jorethang", "Rangpo"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Ziro", "Pasighat", "Bomdila"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Mon", "Tuensang"]
}
crops = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}
reversed_crops = {v: k for k, v in crops.items()}

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Majority voting for prediction
def predict_crop(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity, state, city):
    input_data = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
    predictions = model.predict(input_data)
    majority_vote = Counter(predictions).most_common(1)[0][0]
    return majority_vote

# Check if user is logged in
def is_logged_in():
    return 'user_id' in session

# Route protection decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            flash("❌ Please log in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    # Check if user is logged in and pass this info to the template
    logged_in = is_logged_in()
    user_name = session.get('name', 'User') if logged_in else None
    return render_template('main.html', logged_in=logged_in, user_name=user_name)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Handle registration logic here
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("❌ Passwords do not match!")
            return redirect(url_for('register'))
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            flash("❌ Email already registered!")
            return redirect(url_for('register'))
        
        # Save the user data to MongoDB
        new_user = {
            'name': name,
            'email': email,
            'password': password  # In production, use password hashing!
        }
        users_collection.insert_one(new_user)
        
        flash("✅ Registration successful! Please log in.")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if the user exists and the password matches
        user = users_collection.find_one({'email': email})
        
        if user and user['password'] == password:  # In production, use password hashing!
            # Store user info in session
            session['user_id'] = str(user['_id'])
            session['email'] = user['email']
            session['name'] = user.get('name', 'User')
            
            flash("✅ Login successful!")
            return redirect(url_for('home'))
        else:
            result = "❌ No feasible fertilizer combination found."
            flash("❌ Invalid email or password. Please try again.")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/predict_crop', methods=['GET', 'POST'])
@login_required
def predict_crop_route():
    states = list(states_and_cities.keys())
    if request.method == 'POST':
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        state = request.form['stt']
        city = request.form['city']

        crop_result = predict_crop(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity, state, city)

        return render_template('crop_recommendation.html',
                               crop_result=crops.get(crop_result),
                               states=states,
                               cities=states_and_cities[state],
                               states_and_cities=states_and_cities)

    return render_template('crop_recommendation.html', states=states, states_and_cities=states_and_cities)

@app.route('/fertility', methods=['GET', 'POST'])
@login_required
def fertility():
    states = list(states_and_cities.keys())
    if request.method == 'POST':
        try:
            nitrogen = float(request.form['nitrogen'])
            phosphorous = float(request.form['phosphorous'])
            potassium = float(request.form['pottasium'])
            crop = request.form['crop'].strip().lower()
            
            # Validate that all nutrient values are greater than zero
            if nitrogen <= 0 or phosphorous <= 0 or potassium <= 0:
                flash("❌ Nutrient values must be greater than zero.")
                return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                   fertility_result="❌ All nutrient values (Nitrogen, Phosphorous, Potassium) must be greater than zero.")

            ideal_df = pd.read_csv('FertilizerData1.csv')
            fertilizer_df = pd.read_csv('fertilizer_composition.csv')

            if crop not in ideal_df['Crop'].str.lower().values:
                return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                   fertility_result=f"❌ Crop '{crop}' not found in ideal data.")

            ideal_values = ideal_df[ideal_df['Crop'].str.lower() == crop].iloc[0]
            ideal_N = ideal_values['N']
            ideal_P = ideal_values['P']
            ideal_K = ideal_values['K']

            deficiency_N = max(ideal_N - nitrogen, 0)
            deficiency_P = max(ideal_P - phosphorous, 0)
            deficiency_K = max(ideal_K - potassium, 0)

            N_content = fertilizer_df['N_content'].values
            P_content = fertilizer_df['P_content'].values
            K_content = fertilizer_df['K_content'].values

            c = np.ones(len(fertilizer_df))

            A = [
                -N_content,
                -P_content,
                -K_content
            ]
            b = [
                -deficiency_N,
                -deficiency_P,
                -deficiency_K
            ]

            bounds = [(0, None) for _ in range(len(fertilizer_df))]

            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

            if res.success:
                fertilizer_quantities = res.x
                recommendations = []
                for i, qty in enumerate(fertilizer_quantities):
                    if qty > 0:
                        fertilizer_name = fertilizer_df['Fertilizer'].iloc[i]
                        recommendations.append(f"Apply {qty:.2f} kg/ha of {fertilizer_name}")
                result = " | ".join(recommendations)            
            else:
                    result = "❌ No feasible fertilizer combination found."
                
            return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                               fertility_result=result)
                               
        except ValueError as e:
            # Handle input conversion errors
            error_message = f"❌ Invalid input values: {str(e)}"
            flash(error_message)
            return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                  fertility_result=error_message)
        except Exception as e:
            # Handle other unexpected errors
            error_message = f"❌ An error occurred: {str(e)}"
            flash(error_message)
            return render_template('fertility.html', states=states, states_and_cities=states_and_cities,
                                  fertility_result=error_message)

    return render_template('fertility.html', states=states, states_and_cities=states_and_cities)

@app.route('/diseases')
@login_required
def diseases():
    return render_template('diseases.html')

# Handle feedback form submission
@app.route('/submit_feedback', methods=['POST'])
@login_required
def submit_feedback():
    name = request.form.get('name')
    email = request.form.get('email')
    feedback = request.form.get('feedback')

    if name and email and feedback:
        feedback_doc = {
            "name": name,
            "email": email,
            "feedback": feedback
        }
        try:
            feedback_collection.insert_one(feedback_doc)
            flash("✅ Feedback submitted successfully!")
        except Exception as e:
            flash(f"❌ Failed to submit feedback: {str(e)}")
    else:
        flash("❌ Please fill out all fields.")

    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash("✅ You have been successfully logged out.")
    return redirect(url_for('home'))

# List of plant diseases for our classifier
plant_diseases = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Disease detection model - we'll use MobileNetV2 which is pre-trained and adapt it for our task
disease_model = None

def load_disease_model():
    global disease_model
    try:
        # Check if model is already loaded
        if disease_model is None:
            # Create a MobileNetV2 model pretrained on ImageNet
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Add our own classifier on top
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            predictions = tf.keras.layers.Dense(len(plant_diseases), activation='softmax')(x)
            
            # Create the full model
            disease_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
            
            # Load weights if available, otherwise use base model
            try:
                disease_model.load_weights('plant_disease_model.h5')
                print("Loaded plant disease model weights")
            except:
                print("Using base MobileNetV2 model - fine-tuning needed for better results")
        
        return True
    except Exception as e:
        print(f"Error loading disease model: {str(e)}")
        return False

# Preprocess image for disease detection
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    return img

@app.route('/detect_disease', methods=['POST'])
@login_required
def detect_disease():
    if request.method == 'POST':
        # Check if model is loaded
        if not load_disease_model():
            flash("❌ Disease detection model failed to load.")
            return redirect(url_for('diseases'))
        
        # Get uploaded file
        if 'image' not in request.files:
            flash("❌ No image uploaded.")
            return redirect(url_for('diseases'))
        
        file = request.files['image']
        
        if file.filename == '':
            flash("❌ No image selected.")
            return redirect(url_for('diseases'))
        
        try:
            # Open and process the image
            img = Image.open(file.stream)
            
            # Convert to RGB if the image is not in RGB format (e.g. RGBA PNG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Preprocess the image
            processed_img = preprocess_image(img)
            processed_img = np.expand_dims(processed_img, axis=0)
            
            # Get prediction
            predictions = disease_model.predict(processed_img)
            top_prediction_idx = np.argmax(predictions[0])
            confidence = predictions[0][top_prediction_idx] * 100
            
            # Get the predicted disease label
            predicted_disease = plant_diseases[top_prediction_idx]
              # Get treatment recommendation
            treatment = get_treatment_recommendation(predicted_disease)
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Save detection result to history
            if is_logged_in():
                detection_history = {
                    "user_id": session['user_id'],
                    "user_name": session.get('name', 'User'),
                    "disease": predicted_disease,
                    "confidence": float(confidence),
                    "treatment": treatment,
                    "image": img_str,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                disease_history_collection.insert_one(detection_history)
            
            return render_template('diseases.html', 
                                   prediction_result=True,
                                   disease=predicted_disease,
                                   confidence=confidence,
                                   treatment=treatment,
                                   image=img_str)
        
        except Exception as e:
            flash(f"❌ Error processing image: {str(e)}")
            return redirect(url_for('diseases'))
    
    return redirect(url_for('diseases'))

def get_treatment_recommendation(disease):
    # Dictionary of treatments for various plant diseases
    treatments = {
        'Apple___Apple_scab': 'Apply fungicides like captan or myclobutanil. Prune infected branches during dormant season. Remove fallen leaves to reduce fungal spores.',
        'Apple___Black_rot': 'Remove mummified fruits, cankers, and infected plant parts. Apply fungicides like thiophanate-methyl or captan at 2-week intervals.',
        'Apple___Cedar_apple_rust': 'Keep apple trees away from cedar trees (alternate host). Apply fungicides containing mancozeb or myclobutanil during spring.',
        'Apple___healthy': 'Your apple plant is healthy! Continue regular care with proper watering, fertilization, and pruning.',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Rotate crops with non-host plants. Apply fungicides containing pyraclostrobin, azoxystrobin, or propiconazole.',
        'Corn_(maize)___Common_rust': 'Apply fungicides with active ingredients like azoxystrobin or pyraclostrobin. Plant resistant corn varieties.',
        'Corn_(maize)___Northern_Leaf_Blight': 'Use foliar fungicides like propiconazole or azoxystrobin. Rotate crops and till soil to reduce inoculum.',
        'Corn_(maize)___healthy': 'Your corn plant is healthy! Maintain good agricultural practices like adequate spacing and proper fertilization.',
        'Grape___Black_rot': 'Remove mummified berries and infected leaves. Apply fungicides like myclobutanil or mancozeb during growing season.',
        'Grape___Esca_(Black_Measles)': 'Prune infected wood and protect pruning wounds. Currently no effective chemical treatment; focus on prevention.',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides containing mancozeb or copper compounds. Ensure good air circulation through proper pruning.',
        'Grape___healthy': 'Your grape vine is healthy! Continue regular vine training, pruning, and appropriate watering.',
        'Potato___Early_blight': 'Apply fungicides containing chlorothalonil or copper. Rotate crops and maintain proper plant spacing.',
        'Potato___Late_blight': 'Apply fungicides with mancozeb or chlorothalonil. Remove infected plants immediately. Avoid overhead irrigation.',
        'Potato___healthy': 'Your potato plant is healthy! Maintain appropriate hilling, watering, and fertilization practices.',
        'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Rotate crops and avoid overhead watering. Remove infected plants.',
        'Tomato___Early_blight': 'Apply fungicides with chlorothalonil or copper. Remove lower infected leaves. Mulch soil to prevent spore splash.',
        'Tomato___Late_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Remove infected plants. Avoid wetting leaves during watering.',
        'Tomato___Leaf_Mold': 'Improve air circulation. Reduce humidity in greenhouses. Apply fungicides containing chlorothalonil or mancozeb.',
        'Tomato___Septoria_leaf_spot': 'Apply fungicides with chlorothalonil or copper. Remove infected leaves. Rotate crops annually.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply insecticidal soap or neem oil. Introduce predatory mites. Increase humidity around plants.',
        'Tomato___Target_Spot': 'Apply fungicides containing chlorothalonil or copper. Improve air circulation and reduce leaf wetness.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly vectors with appropriate insecticides or insecticidal soap. Use virus-resistant varieties.',
        'Tomato___Tomato_mosaic_virus': 'No direct cure for viral infection. Remove and destroy infected plants. Control aphids and practice strict hygiene.',
        'Tomato___healthy': 'Your tomato plant is healthy! Continue proper staking, pruning, and regular fertilization.'
    }
    
    # For healthy plants or unknown diseases
    if disease not in treatments:
        if 'healthy' in disease:
            return "Your plant appears to be healthy! Continue with regular care and monitoring."
        else:
            return "No specific treatment information available for this disease. Consult a local agricultural extension service."
    
    return treatments[disease]

@app.route('/disease_history')
@login_required
def disease_history():
    if not is_logged_in():
        flash("❌ Please log in to view your disease detection history.")
        return redirect(url_for('login'))
    
    # Get filter parameters
    plant_filter = request.args.get('plant', 'all')
    health_filter = request.args.get('health', 'all')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Build the query filter
    query = {"user_id": session['user_id']}
    
    # Add plant type filter
    if plant_filter != 'all':
        query["disease"] = {"$regex": f"^{plant_filter}___"}
    
    # Add health status filter
    if health_filter == 'healthy':
        if "disease" in query:
            query["disease"] = {"$and": [query["disease"], {"$regex": "healthy$"}]}
        else:
            query["disease"] = {"$regex": "healthy$"}
    elif health_filter == 'diseased':
        if "disease" in query:
            query["disease"] = {"$and": [query["disease"], {"$regex": "^((?!healthy).)*$"}]}
        else:
            query["disease"] = {"$regex": "^((?!healthy).)*$"}
            
    # Add date range filter
    if date_from:
        if date_to:
            query["timestamp"] = {"$gte": date_from, "$lte": date_to + " 23:59:59"}
        else:
            query["timestamp"] = {"$gte": date_from}
    elif date_to:
        query["timestamp"] = {"$lte": date_to + " 23:59:59"}
      
    # Get the user's detection history
    user_history = list(disease_history_collection.find(query).sort("timestamp", -1))
    
    # Get unique plant types from the user's history for the filter dropdown
    all_detections = list(disease_history_collection.find({"user_id": session['user_id']}))
    plant_types = set()
    for detection in all_detections:
        disease = detection.get('disease', '')
        if '___' in disease:
            plant_type = disease.split('___')[0]
            plant_types.add(plant_type)
    
    # Calculate some statistics
    total_detections = len(all_detections)
    healthy_count = sum(1 for d in all_detections if 'healthy' in d.get('disease', ''))
    disease_count = total_detections - healthy_count
    health_percentage = int((healthy_count / total_detections) * 100) if total_detections > 0 else 0
      # Get the most recent detection date
    most_recent_date = all_detections[0].get('timestamp') if all_detections else None
    
    return render_template('disease_history.html', 
                          history=user_history, 
                          user_name=session.get('name', 'User'),
                          plant_types=sorted(list(plant_types)),
                          current_plant=plant_filter,
                          current_health=health_filter,
                          date_from=date_from,
                          date_to=date_to,
                           stats={
                               'total': total_detections,
                               'healthy': healthy_count,
                               'diseased': disease_count,
                               'health_percentage': health_percentage,
                               'most_recent': most_recent_date
                           })

@app.route('/delete_detection/<detection_id>')
@login_required
def delete_detection(detection_id):
    if not is_logged_in():
        flash("❌ Please log in to manage your disease detection history.")
        return redirect(url_for('login'))
    
    try:
        # Convert string ID to ObjectId
        from bson.objectid import ObjectId
        obj_id = ObjectId(detection_id)
        
        # Check if the detection belongs to the logged-in user
        detection = disease_history_collection.find_one({"_id": obj_id})
        
        if detection and detection.get("user_id") == session['user_id']:
            # Delete the detection
            disease_history_collection.delete_one({"_id": obj_id})
            flash("✅ Detection record deleted successfully.")
        else:
            flash("❌ You don't have permission to delete this record.")
    except Exception as e:
        flash(f"❌ Error deleting detection: {str(e)}")
    
    return redirect(url_for('disease_history'))

@app.route('/batch_delete_detections', methods=['POST'])
@login_required
def batch_delete_detections():
    if not is_logged_in():
        flash("❌ Please log in to manage your disease detection history.")
        return redirect(url_for('login'))
    
    try:
        # Get the list of IDs to delete
        detection_ids = request.form.get('detection_ids', '').split(',')
        if not detection_ids or detection_ids[0] == '':
            flash("❌ No records selected for deletion.")
            return redirect(url_for('disease_history'))
        
        # Convert string IDs to ObjectId and verify user ownership
        obj_ids = []
        for id_str in detection_ids:
            obj_id = ObjectId(id_str.strip())
            detection = disease_history_collection.find_one({"_id": obj_id})
            
            if detection and detection.get("user_id") == session['user_id']:
                obj_ids.append(obj_id)
        
        # Delete the detections
        if obj_ids:
            result = disease_history_collection.delete_many({"_id": {"$in": obj_ids}})
            if result.deleted_count > 0:
                flash(f"✅ Successfully deleted {result.deleted_count} detection records.")
            else:
                flash("❌ No records were deleted.")
        else:
            flash("❌ You don't have permission to delete these records.")
            
    except Exception as e:
        flash(f"❌ Error deleting detections: {str(e)}")
    
    return redirect(url_for('disease_history'))

if __name__ == "__main__":
    app.run(debug=True)
