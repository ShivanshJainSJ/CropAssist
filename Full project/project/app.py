import os
# Set TensorFlow environment variables before importing TensorFlow
import os
import sqlite3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR (suppress more warnings)

import pickle
import numpy as np
import pandas as pd
import json  # For loading class labels
import traceback  # For better error handling
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from collections import Counter
from scipy.optimize import linprog
from pymongo import MongoClient
from bson.objectid import ObjectId  # Add this import for MongoDB ObjectId
from functools import wraps
import warnings
import logging

# Email notification imports
from email_config import send_welcome_email, send_login_notification

# Model improvements imports removed - not currently used

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')

import tensorflow as tf
# Set TensorFlow logging level
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.preprocessing import image 
from PIL import Image
import io
import base64
import random  # Added for random sampling in test-time augmentation

# TensorFlow and Keras imports for plant disease model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
try:
    import cv2  # Import OpenCV for image quality assessment
except ImportError:
    pass  # CV2 is optional, fallback mechanisms are in place
from soil_fertility_api import predict_best_fertilizer, get_fertilizer_recommendation

# Import optimized fertility predictor
try:
    from optimized_fertility_predictor import predict_fertilizer_optimized
    OPTIMIZED_MODEL_AVAILABLE = True
    print("✅ Optimized fertility model loaded successfully")
except ImportError as e:
    OPTIMIZED_MODEL_AVAILABLE = False
    print(f"⚠️ Optimized fertility model not available: {e}")

# Initialize Flask app
app = Flask(__name__)

# Set secret key for session-based flash messaging
app.secret_key = 'your_secret_key_here'

# Configure Flask for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

# MongoDB setup
try:
    # Try connecting with a shorter timeout
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    # Validate the connection
    client.server_info()
    mongo_db = client['feedbackDB']
    feedback_collection = mongo_db['feedbacks']
    users_collection = mongo_db['users']
    disease_history_collection = mongo_db['disease_history']
    mongo_available = True
    print("Successfully connected to MongoDB")
except Exception as e:
    # MongoDB is not available, set up fallback using SQLite
    import sqlite3
    print(f"MongoDB connection failed: {str(e)}")
    print("Using SQLite as fallback database")
    mongo_available = False
    
    # Create or connect to SQLite database
    sqlite_conn = sqlite3.connect('app_database.db', check_same_thread=False)
    sqlite_cursor = sqlite_conn.cursor()
    
    # Create tables if they don't exist
    sqlite_cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')
    
    sqlite_cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            feedback TEXT,
            timestamp TEXT
        )
    ''')
    
    sqlite_cursor.execute('''
        CREATE TABLE IF NOT EXISTS disease_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            user_name TEXT,
            disease TEXT,
            confidence REAL,
            confidence_level TEXT,
            treatment TEXT,
            image TEXT,
            timestamp TEXT,
            used_augmentation INTEGER,
            original_dimensions TEXT,
            was_upscaled INTEGER,
            plant_likelihood REAL
        )
    ''')
    
    sqlite_conn.commit()

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

# Load the crop recommendation model with proper path
project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, 'model.pkl')

# Try to load crop model, use fallback if not available
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("✅ Crop recommendation model loaded successfully")
except FileNotFoundError:
    print("⚠️ Crop recommendation model not found, using fallback predictions")
    model = None

# Crop prediction with fallback
def predict_crop(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity, state, city):
    if model is not None:
        try:
            input_data = np.array([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
            predictions = model.predict(input_data)
            majority_vote = Counter(predictions).most_common(1)[0][0]
            return majority_vote
        except Exception as e:
            print(f"Model prediction error: {e}")
            # Fall back to rule-based prediction if model fails
            return predict_crop_fallback(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity)
    else:
        return predict_crop_fallback(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity)

def predict_crop_fallback(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity):
    """Rule-based crop prediction fallback that returns crop indices matching the crops dictionary"""
    # Return indices that match the crops dictionary (0-21)
    
    # Rice (index 20) - high nitrogen and rainfall
    if nitrogen > 80 and rainfall > 1000:
        return 20  # rice
    # Wheat (not in crops dict, use closest - maybe lentil index 10)
    elif nitrogen > 60 and temperature < 25:
        return 10  # lentil (closest to wheat)
    # Cotton (index 6) - high potassium and temperature
    elif potassium > 40 and temperature > 25:
        return 6   # cotton
    # Maize (index 11) - moderate conditions
    elif temperature > 20 and rainfall < 800:
        return 11  # maize
    # Mango (index 12) - high phosphorous
    elif phosphorous > 40:
        return 12  # mango
    # Banana (index 1) - high humidity
    elif humidity > 80:
        return 1   # banana
    # Apple (index 0) - cooler temperatures
    elif temperature < 20:
        return 0   # apple
    # Default to rice
    else:
        return 20  # rice

# Check if user is logged in
def is_logged_in():
    return 'user_id' in session

# Route protection decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            flash("🔒 Please log in to access this page.")
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
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Basic validation
        if not all([name, email, password, confirm_password]):
            flash("❌ Please fill in all fields.")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("❌ Password must be at least 6 characters long.")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("❌ Passwords do not match!")
            return redirect(url_for('register'))
        
        try:
            if mongo_available:
                # MongoDB implementation                # Check if user already exists
                existing_user = users_collection.find_one({'email': email})
                if existing_user:
                    flash("❌ Email already registered! Please use a different email.")
                    return redirect(url_for('register'))
                
                # Save the user data to MongoDB
                new_user = {
                    'name': name,
                    'email': email,
                    'password': password  # In production, use password hashing!
                }
                result = users_collection.insert_one(new_user)
                
                # Send welcome email
                try:
                    if send_welcome_email(name, email):
                        print(f"Welcome email sent successfully to {email}")
                    else:
                        print(f"Failed to send welcome email to {email}")
                except Exception as email_error:
                    print(f"Error sending welcome email: {str(email_error)}")
            else:
                # SQLite implementation
                # Check if user already exists                sqlite_cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
                existing_user = sqlite_cursor.fetchone()
                if existing_user:
                    flash("❌ Email already registered! Please use a different email.")
                    return redirect(url_for('register'))
                
                # Save the user data to SQLite
                sqlite_cursor.execute(
                    "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                    (name, email, password)  # In production, use password hashing!
                )
                sqlite_conn.commit()
                
                # Send welcome email
                try:
                    if send_welcome_email(name, email):
                        print(f"Welcome email sent successfully to {email}")
                    else:
                        print(f"Failed to send welcome email to {email}")
                except Exception as email_error:
                    print(f"Error sending welcome email: {str(email_error)}")
            
            flash("✅ Registration successful! Welcome to CropAssist. Please log in.")
            return redirect(url_for('login'))
            
        except Exception as e:
            print(f"Registration error: {str(e)}")
            flash("❌ An error occurred during registration. Please try again.")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        # Basic validation
        if not email or not password:
            flash("❌ Please enter both email and password.")
            return redirect(url_for('login'))

        try:            # Check if the user exists and the password matches
            if mongo_available:
                # MongoDB implementation
                user = users_collection.find_one({'email': email})
                
                if user and user['password'] == password:  # In production, use password hashing!
                    # Store user info in session
                    session['user_id'] = str(user['_id'])                    session['email'] = user['email']
                    session['name'] = user.get('name', 'User')
                    
                    # Send login notification email
                    try:
                        if send_login_notification(session['name'], session['email']):
                            print(f"Login notification sent successfully to {session['email']}")
                        else:
                            print(f"Failed to send login notification to {session['email']}")
                    except Exception as email_error:
                        print(f"Error sending login notification: {str(email_error)}")
                    
                    flash("✅ Welcome back! Login successful.")
                    return redirect(url_for('home'))
                else:
                    flash("❌ Invalid email or password. Please check your credentials.")
                    return redirect(url_for('login'))
            else:
                # SQLite implementation
                sqlite_cursor.execute("SELECT id, email, name, password FROM users WHERE email = ?", (email,))
                user = sqlite_cursor.fetchone()
                
                if user and user[3] == password:  # In production, use password hashing!
                    # Store user info in session
                    session['user_id'] = str(user[0])
                    session['email'] = user[1]
                    session['name'] = user[2] if user[2] else 'User'
                    
                    # Send login notification email
                    try:
                        if send_login_notification(session['name'], session['email']):
                            print(f"Login notification sent successfully to {session['email']}")
                        else:
                            print(f"Failed to send login notification to {session['email']}")
                    except Exception as email_error:
                        print(f"Error sending login notification: {str(email_error)}")
                    
                    flash("✅ Welcome back! Login successful.")
                    return redirect(url_for('home'))
                else:
                    flash("❌ Invalid email or password. Please check your credentials.")
                    return redirect(url_for('login'))
        except Exception as e:
            print(f"Login error: {str(e)}")
            flash("❌ An error occurred during login. Please try again.")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/predict_crop', methods=['GET', 'POST'])
@login_required
def predict_crop_route():
    states = list(states_and_cities.keys())
    if request.method == 'POST':
        try:
            # Get form data with proper field names
            nitrogen = float(request.form['nitrogen'])
            phosphorous = float(request.form['phosphorous'])
            potassium = float(request.form['potassium'])  # Fixed typo: was 'pottasium'
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            state = request.form['stt']
            city = request.form['city']

            # Validate inputs
            if any(val < 0 for val in [nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity]):
                flash("❌ All values must be non-negative.")
                return render_template('crop_recommendation.html', 
                                     states=states, 
                                     states_and_cities=states_and_cities,
                                     error="All values must be non-negative.")

            # Validate pH range
            if not (0 <= ph <= 14):
                flash("❌ pH must be between 0 and 14.")
                return render_template('crop_recommendation.html', 
                                     states=states, 
                                     states_and_cities=states_and_cities,
                                     error="pH must be between 0 and 14.")

            # Validate state
            if state not in states_and_cities:
                flash("❌ Please select a valid state.")
                return render_template('crop_recommendation.html', 
                                     states=states, 
                                     states_and_cities=states_and_cities,
                                     error="Please select a valid state.")

            # Get crop prediction
            crop_result = predict_crop(nitrogen, phosphorous, potassium, ph, rainfall, temperature, humidity, state, city)
            
            # Get cities for the selected state
            cities = states_and_cities.get(state, [])
            
            return render_template('crop_recommendation.html',
                                   crop_result=crops.get(crop_result, "Unknown Crop"),
                                   states=states,
                                   cities=cities,
                                   states_and_cities=states_and_cities,
                                   success=True,
                                   # Pass back the form data for display
                                   nitrogen=nitrogen,
                                   phosphorous=phosphorous,
                                   potassium=potassium,
                                   ph=ph,
                                   rainfall=rainfall,
                                   temperature=temperature,
                                   humidity=humidity,
                                   state=state,
                                   city=city)

        except ValueError as e:
            flash("❌ Please enter valid numeric values for all fields.")
            return render_template('crop_recommendation.html', 
                                 states=states, 
                                 states_and_cities=states_and_cities,
                                 error="Please enter valid numeric values for all fields.")
        except KeyError as e:
            flash(f"❌ Missing required field: {e}")
            return render_template('crop_recommendation.html', 
                                 states=states, 
                                 states_and_cities=states_and_cities,
                                 error=f"Missing required field: {e}")
        except Exception as e:
            flash(f"❌ An error occurred: {str(e)}")            
            return render_template('crop_recommendation.html', 
                                 states=states, 
                                 states_and_cities=states_and_cities,
                                 error=f"An error occurred: {str(e)}")
    
    return render_template('crop_recommendation.html', states=states, states_and_cities=states_and_cities)

def validate_fertility_inputs(temperature, humidity, moisture, nitrogen, phosphorous, potassium, soil_type, crop_type):
    """
    Validate fertility analysis inputs and return warnings for out-of-range values
    
    Returns:
        dict: {
            'valid': bool,
            'warnings': list,
            'errors': list,
            'suggestions': list
        }
    """
    warnings_list = []
    errors_list = []
    suggestions_list = []
    
    # Define acceptable ranges based on agricultural standards
    ranges = {
        'temperature': {'min': -10, 'max': 50, 'ideal_min': 15, 'ideal_max': 35},
        'humidity': {'min': 10, 'max': 100, 'ideal_min': 40, 'ideal_max': 80},
        'moisture': {'min': 10, 'max': 100, 'ideal_min': 30, 'ideal_max': 70},
        'nitrogen': {'min': 0, 'max': 200, 'ideal_min': 10, 'ideal_max': 50},
        'phosphorous': {'min': 0, 'max': 100, 'ideal_min': 5, 'ideal_max': 30},
        'potassium': {'min': 0, 'max': 150, 'ideal_min': 5, 'ideal_max': 40}
    }
    
    # Check temperature
    if temperature < ranges['temperature']['min'] or temperature > ranges['temperature']['max']:
        errors_list.append(f"🌡️ Temperature ({temperature}°C) is outside acceptable range ({ranges['temperature']['min']}°C to {ranges['temperature']['max']}°C)")
    elif temperature < ranges['temperature']['ideal_min'] or temperature > ranges['temperature']['ideal_max']:
        warnings_list.append(f"⚠️ Temperature ({temperature}°C) is outside ideal range ({ranges['temperature']['ideal_min']}°C to {ranges['temperature']['ideal_max']}°C)")
        if temperature < ranges['temperature']['ideal_min']:
            suggestions_list.append("🌡️ Consider greenhouse cultivation or warmer climate for better crop growth")
        else:
            suggestions_list.append("🌡️ Consider shade protection or cooling systems in hot weather")
    
    # Check humidity
    if humidity < ranges['humidity']['min'] or humidity > ranges['humidity']['max']:
        errors_list.append(f"💧 Humidity ({humidity}%) is outside acceptable range ({ranges['humidity']['min']}% to {ranges['humidity']['max']}%)")
    elif humidity < ranges['humidity']['ideal_min'] or humidity > ranges['humidity']['ideal_max']:
        warnings_list.append(f"⚠️ Humidity ({humidity}%) is outside ideal range ({ranges['humidity']['ideal_min']}% to {ranges['humidity']['ideal_max']}%)")
        if humidity < ranges['humidity']['ideal_min']:
            suggestions_list.append("💧 Consider irrigation or humidification systems")
        else:
            suggestions_list.append("💧 Ensure proper ventilation to prevent fungal diseases")
    
    # Check moisture
    if moisture < ranges['moisture']['min'] or moisture > ranges['moisture']['max']:
        errors_list.append(f"💧 Soil moisture ({moisture}%) is outside acceptable range ({ranges['moisture']['min']}% to {ranges['moisture']['max']}%)")
    elif moisture < ranges['moisture']['ideal_min'] or moisture > ranges['moisture']['ideal_max']:
        warnings_list.append(f"⚠️ Soil moisture ({moisture}%) is outside ideal range ({ranges['moisture']['ideal_min']}% to {ranges['moisture']['ideal_max']}%)")
        if moisture < ranges['moisture']['ideal_min']:
            suggestions_list.append("💧 Increase irrigation frequency or improve water retention")
        else:
            suggestions_list.append("💧 Improve drainage to prevent waterlogging and root rot")
    
    # Check nitrogen
    if nitrogen < ranges['nitrogen']['min'] or nitrogen > ranges['nitrogen']['max']:
        errors_list.append(f"🧪 Nitrogen ({nitrogen}) is outside acceptable range ({ranges['nitrogen']['min']} to {ranges['nitrogen']['max']})")
    elif nitrogen < ranges['nitrogen']['ideal_min'] or nitrogen > ranges['nitrogen']['ideal_max']:
        warnings_list.append(f"⚠️ Nitrogen ({nitrogen}) is outside ideal range ({ranges['nitrogen']['ideal_min']} to {ranges['nitrogen']['ideal_max']})")
        if nitrogen < ranges['nitrogen']['ideal_min']:
            suggestions_list.append("🧪 Apply nitrogen-rich fertilizers (Urea, Ammonium Nitrate)")
        else:
            suggestions_list.append("🧪 Reduce nitrogen application to prevent leaf burn and environmental issues")
    
    # Check phosphorous
    if phosphorous < ranges['phosphorous']['min'] or phosphorous > ranges['phosphorous']['max']:
        errors_list.append(f"🧪 Phosphorous ({phosphorous}) is outside acceptable range ({ranges['phosphorous']['min']} to {ranges['phosphorous']['max']})")
    elif phosphorous < ranges['phosphorous']['ideal_min'] or phosphorous > ranges['phosphorous']['ideal_max']:
        warnings_list.append(f"⚠️ Phosphorous ({phosphorous}) is outside ideal range ({ranges['phosphorous']['ideal_min']} to {ranges['phosphorous']['ideal_max']})")
        if phosphorous < ranges['phosphorous']['ideal_min']:
            suggestions_list.append("🧪 Apply phosphate fertilizers (DAP, TSP, SSP)")
        else:
            suggestions_list.append("🧪 Reduce phosphorous to prevent nutrient lockup")
    
    # Check potassium
    if potassium < ranges['potassium']['min'] or potassium > ranges['potassium']['max']:
        errors_list.append(f"🧪 Potassium ({potassium}) is outside acceptable range ({ranges['potassium']['min']} to {ranges['potassium']['max']})")
    elif potassium < ranges['potassium']['ideal_min'] or potassium > ranges['potassium']['ideal_max']:
        warnings_list.append(f"⚠️ Potassium ({potassium}) is outside ideal range ({ranges['potassium']['ideal_min']} to {ranges['potassium']['ideal_max']})")
        if potassium < ranges['potassium']['ideal_min']:
            suggestions_list.append("🧪 Apply potash fertilizers (MOP, SOP)")
        else:
            suggestions_list.append("🧪 Reduce potassium to maintain balanced nutrition")
    
    # Check soil type
    valid_soil_types = ['Clayey', 'Loamy', 'Black', 'Red', 'Sandy']
    if soil_type not in valid_soil_types:
        errors_list.append(f"🏔️ Invalid soil type '{soil_type}'. Must be one of: {', '.join(valid_soil_types)}")
    
    # Check crop type
    valid_crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 
                       'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    if crop_type.title() not in valid_crop_types:
        errors_list.append(f"🌾 Invalid crop type '{crop_type}'. Must be one of: {', '.join(valid_crop_types)}")
    
    # Additional validation rules
    if nitrogen <= 0 or phosphorous <= 0 or potassium <= 0:
        errors_list.append("🧪 All nutrient values (N, P, K) must be greater than zero")
    
    # Special case warnings
    if nitrogen > 0 and phosphorous > 0 and potassium > 0:
        # Check NPK ratios
        total_npk = nitrogen + phosphorous + potassium
        n_ratio = nitrogen / total_npk
        p_ratio = phosphorous / total_npk
        k_ratio = potassium / total_npk
        
        if n_ratio > 0.7:
            warnings_list.append("⚠️ Very high nitrogen ratio - may cause excessive vegetative growth")
        if p_ratio > 0.5:
            warnings_list.append("⚠️ Very high phosphorous ratio - may interfere with other nutrients")
        if k_ratio > 0.6:
            warnings_list.append("⚠️ Very high potassium ratio - may affect calcium and magnesium uptake")
    
    # Determine if inputs are valid (no errors, warnings are okay)
    is_valid = len(errors_list) == 0
    
    return {
        'valid': is_valid,
        'warnings': warnings_list,
        'errors': errors_list,
        'suggestions': suggestions_list
    }

@app.route('/advanced_fertility', methods=['GET', 'POST'])
@login_required
def advanced_fertility():
    states = list(states_and_cities.keys())
    soil_types = ['Clayey', 'Loamy', 'Black', 'Red', 'Sandy']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 
                  'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    
    if request.method == 'POST':
        try:
            # Get form data
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            nitrogen = float(request.form['nitrogen'])
            phosphorous = float(request.form['phosphorous'])
            potassium = float(request.form['potassium'])
            soil_type = request.form['soil_type']
            crop_type = request.form['crop_type'].strip()
            
            # Comprehensive input validation
            validation_result = validate_fertility_inputs(
                temperature, humidity, moisture, nitrogen, phosphorous, potassium, soil_type, crop_type
            )
            
            # Check for validation errors (blocking issues)
            if not validation_result['valid']:
                error_messages = validation_result['errors']
                flash("❌ Input validation failed. Please check your values.")
                
                return render_template('advanced_fertility.html', 
                                     states=states, 
                                     states_and_cities=states_and_cities,
                                     fertility_result="❌ " + " | ".join(error_messages),
                                     validation_errors=error_messages,
                                     soil_types=soil_types, 
                                     crop_types=crop_types,
                                     temperature=temperature,
                                     humidity=humidity,
                                     moisture=moisture,
                                     nitrogen=nitrogen,
                                     phosphorous=phosphorous,
                                     potassium=potassium,
                                     soil_type=soil_type,
                                     crop_type=crop_type,
                                     success=False)
            
            # Show warnings and suggestions (non-blocking)
            warnings_to_show = []
            if validation_result['warnings']:
                warnings_to_show.extend(validation_result['warnings'])
            if validation_result['suggestions']:
                warnings_to_show.extend(validation_result['suggestions'])
            
            # Predict fertilizer and get recommendations using enhanced model
            result = predict_fertilizer_enhanced(
                temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type
            )
            
            if result['success']:
                # Prepare results for display
                npk_analysis = result['npk_analysis']
                recommendations = result['recommendations']
                fertilizer = result['fertilizer']
                probabilities = result['probabilities']
                deficiencies = result['deficiencies']
                
                # Format fertilizer probabilities for display
                fertilizer_probs = [{"name": name, "probability": prob * 100} 
                                    for name, prob in list(probabilities.items())[:3]]
                
                return render_template('advanced_fertility.html',
                                    states=states, 
                                    states_and_cities=states_and_cities,
                                    result=result,
                                    npk_analysis=npk_analysis,
                                    recommendations=recommendations,
                                    fertilizer=fertilizer,
                                    fertilizer_probs=fertilizer_probs,
                                    deficiencies=deficiencies,
                                    validation_warnings=warnings_to_show,  # Add validation warnings
                                    soil_types=soil_types,
                                    crop_types=crop_types,
                                    temperature=temperature,
                                    humidity=humidity,
                                    moisture=moisture,
                                    nitrogen=nitrogen,
                                    phosphorous=phosphorous,
                                    potassium=potassium,
                                    soil_type=soil_type,
                                    crop_type=crop_type,
                                    success=True)
            else:
                # Handle error
                flash(f"❌ {result['error']}")
                return render_template('advanced_fertility.html', 
                                      states=states, 
                                      states_and_cities=states_and_cities,
                                      fertility_result=f"❌ Error: {result['error']}",
                                      soil_types=soil_types,
                                      crop_types=crop_types,
                                      temperature=temperature,
                                      humidity=humidity,
                                      moisture=moisture,
                                      nitrogen=nitrogen,
                                      phosphorous=phosphorous,
                                      potassium=potassium,                                      soil_type=soil_type,
                                      crop_type=crop_type,
                                      success=False)
        except ValueError as e:
            # Handle input conversion errors
            error_message = f"❌ Invalid input values: {str(e)}"
            flash(error_message)
            return render_template('advanced_fertility.html', 
                                  states=states, 
                                  states_and_cities=states_and_cities,
                                  fertility_result=error_message,
                                  soil_types=soil_types,
                                  crop_types=crop_types,
                                  temperature=request.form.get('temperature', '30'),
                                  humidity=request.form.get('humidity', '60'),
                                  moisture=request.form.get('moisture', '40'),
                                  nitrogen=request.form.get('nitrogen', '20'),
                                  phosphorous=request.form.get('phosphorous', '15'),
                                  potassium=request.form.get('potassium', '10'),
                                  soil_type=request.form.get('soil_type', ''),
                                  crop_type=request.form.get('crop_type', ''),
                                  success=False)
        except Exception as e:
            # Handle other unexpected errors
            error_message = f"? An error occurred: {str(e)}"
            flash(error_message)
            return render_template('advanced_fertility.html', 
                                  states=states, 
                                  states_and_cities=states_and_cities,
                                  fertility_result=error_message,
                                  soil_types=soil_types,
                                  crop_types=crop_types,
                                  temperature=request.form.get('temperature', '30'),
                                  humidity=request.form.get('humidity', '60'),
                                  moisture=request.form.get('moisture', '40'),
                                  nitrogen=request.form.get('nitrogen', '20'),
                                  phosphorous=request.form.get('phosphorous', '15'),
                                  potassium=request.form.get('potassium', '10'),
                                  soil_type=request.form.get('soil_type', ''),
                                  crop_type=request.form.get('crop_type', ''),
                                  success=False)

    # Define soil types and crop types
    soil_types = ['Clayey', 'Loamy', 'Black', 'Red', 'Sandy']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 
                  'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    
    return render_template('advanced_fertility.html', 
                          states=states, 
                          states_and_cities=states_and_cities,
                          soil_types=soil_types,
                          crop_types=crop_types)

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
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    if name and email and feedback:
        try:
            if mongo_available:
                # MongoDB implementation
                feedback_doc = {
                    "name": name,
                    "email": email,
                    "feedback": feedback,
                    "timestamp": current_time
                }
                feedback_collection.insert_one(feedback_doc)
            else:
                # SQLite implementation
                sqlite_cursor.execute(
                    "INSERT INTO feedbacks (name, email, feedback, timestamp) VALUES (?, ?, ?, ?)",
                    (name, email, feedback, current_time)
                )
                sqlite_conn.commit()
                
            flash("✅ Feedback submitted successfully!")
        except Exception as e:
            flash(f"❌ Failed to submit feedback: {str(e)}")
    else:
        flash("❌ Please fill out all fields.")

    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    # Store user name for goodbye message before clearing session
    user_name = session.get('name', 'User')
    
    # Clear the session
    session.clear()
    
    flash(f"✅ Goodbye, {user_name}! You have been successfully logged out.")
    return redirect(url_for('home'))

# List of plant diseases will be loaded dynamically from labels file
plant_diseases = []
class_labels = {}

# Disease detection model - we'll use our trained plant disease classification model
disease_model = None

def load_disease_model():
    global disease_model, plant_diseases, class_labels
    try:
        print(f"🔍 Load function called. Current state:")
        print(f"   disease_model is None: {disease_model is None}")
        print(f"   plant_diseases length: {len(plant_diseases)}")
        
        if disease_model is None:
            print("Loading PyTorch plant disease detection model (using app_harvest.py approach)...")
            
            # Get the base directory relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Base directory: {base_dir}")
            
            # Paths to PyTorch model and labels
            pytorch_model_path = os.path.join(base_dir, 'plant_disease_model.pth')
            labels_path = os.path.join(base_dir, 'plant_disease_labels.json')
            
            print(f"Looking for PyTorch model at: {pytorch_model_path}")
            print(f"Looking for labels at: {labels_path}")
            
            # Check if PyTorch model exists
            if os.path.exists(pytorch_model_path) and os.path.exists(labels_path):
                try:
                    # Load disease classes exactly like app_harvest.py
                    with open(labels_path, 'r') as f:
                        class_labels = json.load(f)
                    
                    # Convert to plant_diseases list (same as app_harvest.py)
                    plant_diseases = [class_labels[str(i)] for i in range(len(class_labels))]
                    print(f"✅ Loaded {len(plant_diseases)} disease classes")
                      # Load ResNet9 model exactly like app_harvest.py
                    from utils.model import ResNet9
                    import torch
                    
                    disease_model = ResNet9(3, len(plant_diseases))
                    disease_model.load_state_dict(torch.load(
                        pytorch_model_path, map_location=torch.device('cpu')))
                    disease_model.eval()
                    
                    print(f"✅ PyTorch ResNet9 model loaded successfully with {len(plant_diseases)} classes")
                    print(f"🔍 After loading:")
                    print(f"   disease_model is None: {disease_model is None}")
                    print(f"   plant_diseases length: {len(plant_diseases)}")
                    
                    return True
                        
                except Exception as e:
                    print(f"❌ Error loading PyTorch model: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to TensorFlow model if PyTorch fails
                    return load_tensorflow_model(base_dir)
            else:
                print(f"❌ PyTorch model not found at {pytorch_model_path}")
                print("Falling back to TensorFlow model...")
                return load_tensorflow_model(base_dir)
        else:
            # Model is already loaded
            print("✅ Model already loaded, returning True")
            return True
            
    except Exception as e:
        print(f"❌ Error loading disease model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_tensorflow_model(base_dir):
    """Load TensorFlow/Keras model as fallback"""
    global disease_model, plant_diseases, class_labels
    
    try:
        # Paths to TensorFlow model and labels
        model_path = os.path.join(base_dir, 'plant_disease_model.h5')
        labels_path = os.path.join(base_dir, 'plant_disease_labels.json')
        
        print(f"Looking for TensorFlow model at: {model_path}")
        
        # Check if TensorFlow model exists
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                # Load TensorFlow model with warning suppression
                print("Loading trained TensorFlow plant disease model...")
                
                # Suppress TensorFlow warnings temporarily
                import warnings
                import logging
                old_level = tf.get_logger().level
                tf.get_logger().setLevel(logging.ERROR)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    disease_model = tf.keras.models.load_model(model_path, compile=False)
                    
                    # Manually compile the model for inference
                    disease_model.compile(
                        optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                
                # Restore logging level
                tf.get_logger().setLevel(old_level)
                
                # Load class labels first
                with open(labels_path, 'r') as f:
                    class_labels = json.load(f)
                
                # Convert to plant_diseases list for compatibility
                plant_diseases = [class_labels[str(i)] for i in range(len(class_labels))]
                print(f"✅ Loaded {len(plant_diseases)} disease classes")
                
                # Build the model by calling it with dummy input
                dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
                dummy_output = disease_model.predict(dummy_input, verbose=0)
                print(f"✅ TensorFlow model built and tested successfully")
                print(f"📊 Model input shape: (None, 224, 224, 3)")
                print(f"📊 Model output shape: {dummy_output.shape}")
                print(f"📊 Expected classes: {len(plant_diseases)}, Model output classes: {dummy_output.shape[-1]}")
                
                # Verify model output matches expected classes
                if dummy_output.shape[-1] != len(plant_diseases):
                    raise ValueError(f"Model output classes ({dummy_output.shape[-1]}) don't match label classes ({len(plant_diseases)})")
                
                print(f"✅ TensorFlow model loaded successfully")
                return True
                
            except Exception as e:
                print(f"❌ Error loading TensorFlow model: {str(e)}")
                # Fall back to creating a basic model
                return create_fallback_model(base_dir)
        else:
            print(f"❌ TensorFlow model not found at {model_path}")
            print("Creating fallback EfficientNet model...")
            return create_fallback_model(base_dir)
            
    except Exception as e:
        print(f"❌ Error in TensorFlow model loading: {str(e)}")
        return create_fallback_model(base_dir)

def create_fallback_model(base_dir):
    """Create a fallback EfficientNet model when trained model is not available"""
    global disease_model, plant_diseases, class_labels
    
    try:
        print("Creating fallback EfficientNet model...")
        
        # Use default 19-class plant diseases for fallback
        fallback_diseases = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust', 
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        
        plant_diseases = fallback_diseases
        class_labels = {str(i): disease for i, disease in enumerate(fallback_diseases)}
        
        # Create EfficientNet model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Add classifier on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(len(plant_diseases), activation='softmax')(x)
        
        # Create the full model
        disease_model = Model(inputs=base_model.input, outputs=predictions)
        print(f"✅ Created fallback EfficientNet model with {len(plant_diseases)} classes")
        print("⚠️ Note: Using fallback model - results may not be accurate without trained weights")
        
        # Test model
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        dummy_output = disease_model.predict(dummy_input, verbose=0)
        print(f"✅ Fallback model test passed: output shape {dummy_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating fallback model: {str(e)}")
        return False

# Preprocess image for disease detection with enhanced handling
def preprocess_image(img):
    """
    Preprocess image with enhanced handling of different image types
    and sizes, including upscaling of low-resolution images.
    Works specifically for EfficientNet with TensorFlow/Keras.
    
    Args:
        img: PIL Image object
        
    Returns:
        processed_img: Preprocessed numpy array ready for model input
        upscaled: Boolean indicating whether the image was upscaled
    """
    import numpy as np
    import os
    
    # Target size for the model
    target_size = (224, 224)
    
    # Check if upscaling is needed (low resolution image)
    width, height = img.size
    upscaled = False
    
    if width < 224 or height < 224:
        upscaled = True
        # Before resizing, apply super-resolution if OpenCV is available
        try:
            import cv2
            # Convert PIL image to OpenCV format
            img_cv = np.array(img)
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:  # Ensure it's RGB
                # Try using EDSR super-resolution model if available
                try:
                    # Create SR object
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    # Check if model is available
                    model_path = 'models/EDSR_x4.pb'
                    if os.path.exists(model_path):
                        sr.readModel(model_path)
                        sr.setModel("edsr", 4)  # 4x upscaling
                        img_cv = sr.upsample(img_cv)
                    else:
                        # Fallback to bicubic interpolation
                        scale_factor = 224 / min(width, height)
                        new_size = (int(width * scale_factor), int(height * scale_factor))
                        img_cv = cv2.resize(img_cv, new_size, interpolation=cv2.INTER_CUBIC)
                except:
                    # Fallback to bicubic interpolation if dnn_superres not available
                    scale_factor = 224 / min(width, height)
                    new_size = (int(width * scale_factor), int(height * scale_factor))
                    img_cv = cv2.resize(img_cv, new_size, interpolation=cv2.INTER_CUBIC)
                
                # Convert back to PIL
                img = Image.fromarray(img_cv)
        except (ImportError, AttributeError) as e:
            # If OpenCV is not available, use PIL's high-quality upscaling
            print(f"OpenCV not available for super-resolution: {str(e)}")
            scale_factor = 224 / min(width, height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
    
    # Resize image to target size with high quality
    img = img.resize(target_size, Image.LANCZOS)    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Convert to float32 and scale to [0, 1]
    img_array = img_array.astype(np.float32)
    img_array /= 255.0
    
    # Apply EfficientNet-specific preprocessing
    # EfficientNet expects ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # After scaling to [0,1], apply ImageNet normalization
    img_array[..., 0] = (img_array[..., 0] - 0.485) / 0.229  # Red channel
    img_array[..., 1] = (img_array[..., 1] - 0.456) / 0.224  # Green channel  
    img_array[..., 2] = (img_array[..., 2] - 0.406) / 0.225  # Blue channel
    
    # Add batch dimension for model input (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, upscaled

# Function to perform test-time augmentation for better inference with PyTorch
def test_time_augmentation(img_tensor, model, num_augmentations=5):
    """
    Apply multiple augmentations to the input image and average predictions.
    Works with both PyTorch models and ONNX models.
    
    Args:
        img_tensor: Input image tensor of shape (3, 224, 224) or numpy array
        model: Loaded model for prediction (PyTorch or ONNX)
        num_augmentations: Number of augmentations to perform
    
    Returns:
        avg_predictions: Averaged prediction array or None if error
    """
    try:
        import torch
        import numpy as np
        import random
        
        print("Starting test-time augmentation...")
        
        # Check if model exists
        if model is None:
            print("Error: Model is None")
            return None
            
        # Initialize predictions list
        all_predictions = []
        
        # Check if model is PyTorch or ONNX
        is_pytorch = hasattr(model, 'eval')
        is_onnx = hasattr(model, 'run')
        
        print(f"Model type: PyTorch={is_pytorch}, ONNX={is_onnx}")
        
        # Handle PyTorch model
        if is_pytorch:
            try:
                import torchvision.transforms.functional as F
                
                # Set device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Ensure model is in eval mode
                model.eval()
                
                # Check tensor
                if not isinstance(img_tensor, torch.Tensor):
                    print(f"Warning: Expected PyTorch tensor but got {type(img_tensor)}. Converting...")
                    import torchvision.transforms as transforms
                    transform = transforms.ToTensor()
                    img_tensor = transform(img_tensor)
                
                # Original image prediction
                with torch.no_grad():
                    img_batch = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
                    orig_pred = model(img_batch)
                    # Convert to numpy and check if result is valid
                    orig_numpy = orig_pred.cpu().numpy()
                    if orig_numpy is None or orig_numpy.size == 0:
                        print("Error: Model returned empty prediction")
                        return None
                    all_predictions.append(orig_numpy[0])
                    print(f"Original prediction shape: {orig_numpy.shape}")
                
                # Define augmentation types for PyTorch
                augmentations = [
                    # Horizontal flip
                    lambda x: F.hflip(x),
                    # Vertical flip
                    lambda x: F.vflip(x),
                    # 90 degree rotation
                    lambda x: F.rotate(x, 90),
                    # 180 degree rotation
                    lambda x: F.rotate(x, 180),
                    # 270 degree rotation
                    lambda x: F.rotate(x, 270),
                    # Brightness increase by 10%
                    lambda x: F.adjust_brightness(x, 1.1),
                    # Brightness decrease by 10%
                    lambda x: F.adjust_brightness(x, 0.9),
                ]
                
                # Random selection of augmentation types based on num_augmentations
                selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
                
                # Apply selected augmentations and get predictions
                for i, aug_func in enumerate(selected_augmentations):
                    try:
                        # Apply augmentation
                        aug_img = aug_func(img_tensor.squeeze(0))  # Remove batch dimension
                        
                        # Make prediction with PyTorch model
                        with torch.no_grad():
                            aug_img = aug_img.unsqueeze(0).to(device)  # Add batch dimension
                            pred = model(aug_img)
                            pred_numpy = pred.cpu().numpy()
                            if pred_numpy is not None and pred_numpy.size > 0:
                                all_predictions.append(pred_numpy[0])
                                print(f"Aug {i+1} prediction shape: {pred_numpy.shape}")
                    except Exception as e:
                        print(f"Error during augmentation {i+1}: {str(e)}")
                        continue
            
            except Exception as e:
                import traceback
                print(f"Error in PyTorch augmentation: {str(e)}")
                traceback.print_exc()
                if len(all_predictions) == 0:
                    return None
        
        # Handle ONNX model
        elif is_onnx:
            try:
                # Convert tensor to numpy if needed
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.numpy()
                else:
                    img_np = img_tensor
                    
                print(f"ONNX input shape: {img_np.shape}")
                    
                # Get input name
                input_name = model.get_inputs()[0].name
                
                # Original image prediction
                results = model.run(None, {input_name: img_np[None, ...]})
                if results is None or len(results) == 0 or results[0] is None:
                    print("Error: ONNX model returned None or empty result")
                    return None
                    
                orig_pred = results[0][0]
                all_predictions.append(orig_pred)
                print(f"Original ONNX prediction shape: {orig_pred.shape}")
                
                # Define augmentation types for numpy arrays
                augmentations = [
                    # Horizontal flip
                    lambda x: np.flip(x, axis=2),
                    # Vertical flip
                    lambda x: np.flip(x, axis=1),
                    # 90 degree rotation
                    lambda x: np.rot90(x, k=1, axes=(1, 2)),
                    # 180 degree rotation
                    lambda x: np.rot90(x, k=2, axes=(1, 2)),
                    # 270 degree rotation
                    lambda x: np.rot90(x, k=3, axes=(1, 2)),
                    # Brightness increase by 10%
                    lambda x: np.clip(x * 1.1, 0, 1),
                    # Brightness decrease by 10%
                    lambda x: np.clip(x * 0.9, 0, 1),
                ]
                
                # Random selection of augmentation types based on num_augmentations
                selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
                
                # Apply selected augmentations and get predictions
                for i, aug_func in enumerate(selected_augmentations):
                    try:
                        # Apply augmentation
                        aug_img = aug_func(img_np.copy())
                        
                        # Make prediction with ONNX model
                        results = model.run(None, {input_name: aug_img[None, ...]})
                        if results is not None and len(results) > 0 and results[0] is not None:
                            pred = results[0][0]
                            all_predictions.append(pred)
                            print(f"Aug {i+1} ONNX prediction shape: {pred.shape}")
                    except Exception as e:
                        print(f"Error during ONNX augmentation {i+1}: {str(e)}")
                        continue
                        
            except Exception as e:
                import traceback
                print(f"Error in ONNX augmentation: {str(e)}")
                traceback.print_exc()
                if len(all_predictions) == 0:
                    return None
        else:
            print("Unsupported model type for test-time augmentation")
            return None
           # If we have predictions, average them
        if len(all_predictions) > 0:
            print(f"Averaging {len(all_predictions)} predictions...")
            avg_predictions = np.mean(all_predictions, axis=0)
            print(f"Final prediction shape: {avg_predictions.shape}")
            return avg_predictions
        else:
            print("No valid predictions to average")
            return None
            
    except Exception as e:
        import traceback
        print(f"Unexpected error in test_time_augmentation: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/detect_disease', methods=['POST'])
@login_required
def detect_disease():
    if request.method == 'POST':
        # Check if model is loaded
        if not load_disease_model():
            flash("? Disease detection model failed to load.")
            return redirect(url_for('diseases'))
        
        # Get uploaded file
        if 'image' not in request.files:
            flash("? No image uploaded.")
            return redirect(url_for('diseases'))
        
        file = request.files['image']
        
        if file.filename == '':
            flash("? No image selected.")
            return redirect(url_for('diseases'))
          # Check file size (limit to 10MB)
        file.seek(0, 2)  # Move to end of file
        file_size = file.tell()
        file.seek(0)     # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            flash("? Image file too large. Please upload an image smaller than 10MB.")
            return redirect(url_for('diseases'))
        
        try:
            # Open and process the image with error handling
            try:
                img = Image.open(file.stream)
            except Exception as img_error:
                flash(f"? Invalid image file. Please upload a valid image (JPG, PNG, etc.): {str(img_error)}")
                return redirect(url_for('diseases'))
            
            # Convert to RGB if the image is not in RGB format (e.g. RGBA PNG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Store original image dimensions
            original_width, original_height = img.size
            
            # Create a copy for display
            display_img = img.copy()
            
            # Use the disease model with proper API handling
            import numpy as np
            
            # Initialize variables to prevent undefined errors
            confidence = 0.0
            top_prediction_idx = 0
            avg_predictions = None
            was_upscaled = False
              # Make prediction using the exact same method as app_harvest.py
            try:
                # Use direct prediction method (same as app_harvest.py)
                print("🔄 Using direct PyTorch prediction (app_harvest.py method)...")
                result = predict_image_direct(img)
                
                if result.get("success"):
                    # Extract results
                    top_prediction_idx = result["predicted_index"]
                    confidence = float(result["confidence"])
                    avg_predictions = result.get("all_probabilities", np.zeros(len(plant_diseases)))
                    was_upscaled = False  # No preprocessing like app_harvest.py
                    print(f"✅ Direct prediction successful: {confidence:.2f}% confidence")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"❌ Direct prediction failed: {error_msg}")
                    flash(f"? Prediction failed: {error_msg}")
                    return redirect(url_for('diseases'))
                
                # Validation check
                if avg_predictions is None or len(avg_predictions) != len(plant_diseases):
                    flash(f"? Model output is invalid. Expected {len(plant_diseases)} classes, got: {len(avg_predictions) if avg_predictions is not None else 'None'}")
                    return redirect(url_for('diseases'))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ Prediction error: {str(e)}")
                flash(f"? Error during prediction: {str(e)}")
                return redirect(url_for('diseases'))
            
            # Additional validation
            if avg_predictions is None:
                flash("? Unable to make a prediction. Please try again with a different image.")
                return redirect(url_for('diseases'))
            
            # Confidence thresholds for prediction reliability
            HIGH_CONFIDENCE_THRESHOLD = 65.0
            MEDIUM_CONFIDENCE_THRESHOLD = 45.0
            LOW_CONFIDENCE_THRESHOLD = 25.0
            
            # Get the predicted disease label
            predicted_disease = plant_diseases[top_prediction_idx]
            
            # Confidence level determination and messaging
            confidence_level = "high"
            confidence_message = None
            low_confidence = False
            secondary_message = None
            
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "very_low"
                low_confidence = True
                confidence_message = f"Very low confidence prediction ({confidence:.1f}%). The model is highly uncertain about this diagnosis."
                secondary_message = "This could indicate: (1) The image may not contain a supported plant disease, (2) Poor image quality, or (3) A disease not in our training dataset. Consider consulting an agricultural expert."
            elif confidence < MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_level = "low"
                low_confidence = True
                confidence_message = f"Low confidence prediction ({confidence:.1f}%). Please verify this diagnosis with additional sources."
            elif confidence < HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "medium"
                confidence_message = f"Moderate confidence prediction ({confidence:.1f}%). This diagnosis appears reasonably reliable."
            else:
                confidence_level = "high"
                confidence_message = f"High confidence prediction ({confidence:.1f}%). This diagnosis appears reliable."
            
            # Update secondary message for very low confidence predictions
            if confidence < 30.0:
                # Add additional warning for very uncertain predictions
                secondary_message = "The model is very uncertain about this prediction. Results may be unreliable. This might be because the image does not contain one of our supported plant diseases, or the image quality is poor."
                
                # Add a list of supported plants for reference
                plant_types = set([disease.split('___')[0] for disease in plant_diseases])
                supported_plants = ", ".join(sorted([p.replace('_', ' ') for p in plant_types]))
                secondary_message += f"<br><br>Supported plants: {supported_plants}"
            else:
                secondary_message = None
            
            # Get treatment recommendation
            treatment = get_treatment_recommendation(predicted_disease)
            
            # Convert image to base64 for display
            buffered = io.BytesIO()            
            display_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Get top 3 predictions for display
            top_3_indices = np.argsort(avg_predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    "disease": plant_diseases[idx],
                    "confidence": float(avg_predictions[idx] * 100)
                }
                for idx in top_3_indices            ]
              # Save detection result to history
            if is_logged_in():
                current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                was_upscaled_int = 1 if was_upscaled else 0
                
                if mongo_available:
                    # MongoDB implementation
                    detection_history = {
                        "user_id": session['user_id'],
                        "user_name": session.get('name', 'User'),
                        "disease": predicted_disease,
                        "confidence": confidence,
                        "confidence_level": confidence_level,
                        "treatment": treatment,
                        "image": img_str,
                        "timestamp": current_time,
                        "original_dimensions": f"{original_width}x{original_height}",
                        "was_upscaled": was_upscaled
                    }
                    disease_history_collection.insert_one(detection_history)
                else:
                    # SQLite implementation
                    sqlite_cursor.execute("""
                        INSERT INTO disease_history 
                        (user_id, user_name, disease, confidence, confidence_level, 
                        treatment, image, timestamp, original_dimensions, 
                        was_upscaled)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session['user_id'],
                        session.get('name', 'User'),
                        predicted_disease,
                        confidence,
                        confidence_level,
                        treatment,
                        img_str,
                        current_time,
                        f"{original_width}x{original_height}",
                        was_upscaled_int
                    ))
                    sqlite_conn.commit()
            
            return render_template('diseases.html', 
                                  prediction_result=True,
                                  disease=predicted_disease,
                                  confidence=confidence,
                                  confidence_level=confidence_level,
                                  treatment=treatment,
                                  image=img_str,
                                  low_confidence=low_confidence,
                                  confidence_message=confidence_message,
                                  secondary_message=secondary_message,
                                  top_3_predictions=top_3_predictions,
                                  not_a_plant=False,
                                  was_upscaled=was_upscaled,
                                  original_dimensions=f"{original_width}x{original_height}")
        
        except Exception as e:
            import traceback
            print(f"Disease detection error: {str(e)}")
            print(traceback.format_exc())
            flash(f"? Error processing image: {str(e)}")
            return redirect(url_for('diseases'))
    
    return redirect(url_for('diseases'))

def get_treatment_recommendation(disease):
    """
    Generate structured treatment recommendations for plant diseases
    
    Returns:
        str: Properly formatted HTML treatment recommendation
    """
    
    # Structured treatment recommendation template
    def format_treatment(disease_name, severity="Moderate", 
                        immediate_actions=None, prevention_measures=None, 
                        organic_options=None, chemical_treatments=None,
                        monitoring_tips=None, additional_notes=None):
        """Format treatment recommendation with consistent structure"""
        
        immediate_actions = immediate_actions or []
        prevention_measures = prevention_measures or []
        organic_options = organic_options or []
        chemical_treatments = chemical_treatments or []
        monitoring_tips = monitoring_tips or []
        additional_notes = additional_notes or []
        
        # Determine severity color
        severity_colors = {
            "Low": "text-green-600",
            "Moderate": "text-amber-600", 
            "High": "text-red-600",
            "Critical": "text-red-800"
        }
        severity_color = severity_colors.get(severity, "text-amber-600")
        
        html = f"""
        <div class="treatment-recommendation">
            <div class="disease-header mb-4 p-3 bg-gray-50 rounded-lg">
                <h4 class="text-lg font-bold text-gray-800 mb-1">{disease_name}</h4>
                <div class="flex items-center space-x-2">
                    <span class="text-sm text-gray-600">Severity Level:</span>
                    <span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-200 {severity_color}">{severity}</span>
                </div>
            </div>
        """
        
        if immediate_actions:
            html += """
            <div class="treatment-section mb-4">
                <h5 class="flex items-center text-md font-semibold text-red-700 mb-2">
                    <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
                    </svg>
                    Immediate Actions Required
                </h5>
                <ul class="list-disc pl-6 space-y-1 text-sm text-gray-700">
            """
            for action in immediate_actions:
                html += f"<li>{action}</li>"
            html += "</ul></div>"
        
        if chemical_treatments:
            html += """
            <div class="treatment-section mb-4">
                <h5 class="flex items-center text-md font-semibold text-blue-700 mb-2">
                    <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z"/>
                    </svg>
                    Chemical Treatment Options
                </h5>
                <ul class="list-disc pl-6 space-y-1 text-sm text-gray-700">
            """
            for treatment in chemical_treatments:
                html += f"<li>{treatment}</li>"
            html += "</ul></div>"
        
        if organic_options:
            html += """
            <div class="treatment-section mb-4">
                <h5 class="flex items-center text-md font-semibold text-green-700 mb-2">
                    <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586l1.293-1.293a1 1 0 111.414 1.414L10.414 12l1.293 1.293a1 1 0 01-1.414 1.414L9 13.414l-1.293 1.293a1 1 0 01-1.414-1.414L7.586 12 6.293 10.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                    </svg>
                    Organic & Natural Options
                </h5>
                <ul class="list-disc pl-6 space-y-1 text-sm text-gray-700">
            """
            for option in organic_options:
                html += f"<li>{option}</li>"
            html += "</ul></div>"
        
        if prevention_measures:
            html += """
            <div class="treatment-section mb-4">
                <h5 class="flex items-center text-md font-semibold text-purple-700 mb-2">
                    <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                    </svg>
                    Prevention Measures
                </h5>
                <ul class="list-disc pl-6 space-y-1 text-sm text-gray-700">
            """
            for measure in prevention_measures:
                html += f"<li>{measure}</li>"
            html += "</ul></div>"
        
        if monitoring_tips:
            html += """
            <div class="treatment-section mb-4">
                <h5 class="flex items-center text-md font-semibold text-indigo-700 mb-2">
                    <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                        <path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd"/>
                    </svg>
                    Monitoring & Follow-up
                </h5>
                <ul class="list-disc pl-6 space-y-1 text-sm text-gray-700">
            """
            for tip in monitoring_tips:
                html += f"<li>{tip}</li>"
            html += "</ul></div>"
        
        if additional_notes:
            html += """
            <div class="treatment-section mb-4">
                <div class="bg-blue-50 border-l-4 border-blue-400 p-3 rounded">
                    <h5 class="flex items-center text-md font-semibold text-blue-800 mb-2">
                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
                        </svg>
                        Important Notes
                    </h5>
                    <ul class="space-y-1 text-sm text-blue-700">
            """
            for note in additional_notes:
                html += f"<li>• {note}</li>"
            html += "</ul></div></div>"
        
        html += """
            <div class="treatment-footer mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <p class="text-xs text-amber-800">
                    <strong>Disclaimer:</strong> These recommendations are for general guidance only. 
                    Always read and follow product labels, and consult with local agricultural extension services 
                    or plant pathologists for region-specific advice.
                </p>
            </div>
        </div>
        """
        
        return html
    
    # Disease-specific treatment recommendations
    treatments = {
        'Apple___Apple_scab': format_treatment(
            disease_name="Apple Scab (Venturia inaequalis)",
            severity="Moderate",
            immediate_actions=[
                "Remove and destroy all infected leaves, fruits, and debris from ground and tree",
                "Apply fungicide treatment immediately upon symptom detection",
                "Improve air circulation around affected trees"
            ],
            chemical_treatments=[
                "Myclobutanil (Immunox) - Apply every 7-14 days during wet conditions",
                "Captan - Protective fungicide, apply before rainfall",
                "Sulfur-based fungicides - Effective for early season protection",
                "Propiconazole - Systemic fungicide for established infections"
            ],
            organic_options=[
                "Baking soda spray (1 tsp per quart water) for minor infections",
                "Neem oil applications every 7-10 days",
                "Copper-based fungicides during dormant season",
                "Compost tea applications to boost plant immunity"
            ],
            prevention_measures=[
                "Plant scab-resistant apple varieties (Liberty, Enterprise, Freedom)",
                "Prune trees for proper air circulation and light penetration",
                "Apply preventative fungicide sprays from bud break to petal fall",
                "Maintain proper tree spacing (minimum 6-8 feet apart)",
                "Remove fallen leaves and fruit promptly in autumn"
            ],
            monitoring_tips=[
                "Inspect leaves weekly during spring and early summer",
                "Look for olive-green to brown spots on leaves and fruit", 
                "Monitor weather conditions - disease thrives in cool, wet weather",
                "Check for premature leaf drop, especially from lower branches"
            ],
            additional_notes=[
                "Scab is most active when temperatures are 55-75°F with high humidity",
                "Early season prevention is more effective than treatment after infection",
                "Consider resistant rootstocks for new plantings"
            ]
        ),        
        'Apple___Black_rot': format_treatment(
            disease_name="Apple Black Rot (Botryosphaeria obtusa)",
            severity="High",
            immediate_actions=[
                "Remove and burn all infected fruits, branches, and cankers immediately",
                "Sterilize pruning tools with 70% alcohol between cuts",
                "Apply targeted fungicide treatment to affected areas"
            ],
            chemical_treatments=[
                "Thiophanate-methyl - Systemic fungicide for internal infections",
                "Captan - Protective fungicide for wound protection",
                "Myclobutanil - Apply every 10-14 days during growing season",
                "Copper fungicides during dormant season"
            ],
            organic_options=[
                "Copper sulfate applications during dormancy", 
                "Bordeaux mixture for protective coverage",
                "Proper sanitation and removal of infected material",
                "Lime sulfur sprays during dormant season"
            ],
            prevention_measures=[
                "Prune during dry weather to minimize infection opportunities",
                "Maintain tree vigor through proper fertilization and watering",
                "Remove water sprouts and suckers that provide entry points",
                "Apply dormant season sprays to prevent overwintering spores",
                "Avoid mechanical injuries to bark and branches"
            ],
            monitoring_tips=[
                "Inspect for sunken cankers on branches and trunk",
                "Look for black, rotted fruit with concentric rings",
                "Check for dieback of shoots and branches",
                "Monitor for frog-eye leaf spots in early summer"
            ],
            additional_notes=[
                "Black rot is more severe during hot, humid weather",
                "Stressed trees are more susceptible to infection",
                "Remove infected wood at least 6 inches below visible symptoms"
            ]
        ),
        
        'Apple___healthy': format_treatment(
            disease_name="Healthy Apple Plant - Maintenance Care",
            severity="Low",
            immediate_actions=[
                "Continue current care routine - your plant is healthy!",
                "Conduct weekly visual inspections for early problem detection",
                "Maintain consistent watering and nutrition schedule"
            ],
            prevention_measures=[
                "Apply balanced fertilizer (10-10-10) in early spring",
                "Prune during dormancy for structure and air circulation",
                "Apply preventative fungicide program as per local recommendations", 
                "Maintain 2-3 inch mulch layer around base (not touching trunk)",
                "Water deeply but infrequently, avoiding wetting foliage"
            ],
            monitoring_tips=[
                "Weekly inspection for pest and disease symptoms",
                "Monitor soil moisture - should be consistently moist but not waterlogged",
                "Check for proper fruit development and thinning needs",
                "Observe leaf color and growth patterns for nutrient deficiencies"
            ],
            additional_notes=[
                "Healthy plants are more resistant to diseases and pests",
                "Preventative care is more cost-effective than treating problems",
                "Consider soil testing every 2-3 years for optimal nutrition"
            ]
        ),
        
        'Corn_(maize)___Common_rust': format_treatment(
            disease_name="Common Rust (Puccinia sorghi)",
            severity="Moderate", 
            immediate_actions=[
                "Apply foliar fungicide when first pustules appear",
                "Remove heavily infected lower leaves if practical",
                "Avoid overhead irrigation to reduce leaf wetness"
            ],
            chemical_treatments=[
                "Azoxystrobin - Excellent for rust control, apply at first symptoms",
                "Pyraclostrobin - Systemic protection, 14-day intervals",
                "Triazole fungicides (Propiconazole) - Curative and protective action",
                "Strobilurin + Triazole combinations for enhanced efficacy"
            ],
            organic_options=[
                "Copper-based fungicides for organic production",
                "Sulfur applications in early morning or evening",
                "Bacillus subtilis biological fungicides",
                "Plant essential oil sprays (cinnamon, clove, thyme)"
            ],
            prevention_measures=[
                "Plant rust-resistant corn hybrids when available",
                "Avoid excessive nitrogen fertilization which increases susceptibility",
                "Ensure adequate plant spacing for air circulation",
                "Practice crop rotation with non-host crops",
                "Time planting to avoid peak rust pressure periods"
            ],
            monitoring_tips=[
                "Scout weekly during warm, humid conditions (60-80°F)",
                "Look for small, cinnamon-brown pustules on leaves",
                "Check both leaf surfaces, especially lower leaves first",
                "Monitor weather forecasts - rust spreads rapidly in humid conditions"
            ],
            additional_notes=[
                "Common rust requires an alternate host (Oxalis) to complete lifecycle",
                "Cool, dry weather naturally suppresses rust development",
                "Economic thresholds vary by growth stage and hybrid susceptibility"
            ]
        ),
        
        'Tomato___Late_blight': format_treatment(
            disease_name="Late Blight (Phytophthora infestans)",
            severity="Critical",
            immediate_actions=[
                "URGENT: Remove and destroy all infected plants immediately",
                "Apply protective fungicide to healthy plants within 24 hours",
                "Isolate affected area to prevent spread to other plants",
                "Improve air circulation and reduce humidity around plants"
            ],
            chemical_treatments=[
                "Chlorothalonil - Broad-spectrum protectant, apply immediately",
                "Mancozeb - Preventative fungicide, 7-10 day intervals",
                "Copper fungicides - Effective for bacterial and fungal diseases",
                "Phosphorous acid - Systemic acquired resistance activator"
            ],
            organic_options=[
                "Copper sulfate (Bordeaux mixture) - OMRI approved",
                "Bicarbonate solutions for minor infections",
                "Bacillus amyloliquefaciens biological control",
                "Immediate removal and destruction of infected material"
            ],
            prevention_measures=[
                "Choose late blight resistant varieties (Iron Lady, Defiant PhR)",
                "Space plants adequately for air circulation",
                "Avoid overhead watering - use drip irrigation or soaker hoses",
                "Apply preventative fungicide program in high-risk periods",
                "Remove volunteer tomato and potato plants that harbor disease"
            ],
            monitoring_tips=[
                "Daily inspection during cool, wet weather (50-70°F)",
                "Look for dark, water-soaked spots on leaves and stems",
                "Check for white fuzzy growth on leaf undersides",
                "Monitor local disease alerts and weather conditions"
            ],
            additional_notes=[
                "Late blight can destroy entire crop within days under favorable conditions",
                "This is the same disease that caused the Irish Potato Famine",
                "Report outbreaks to local extension services for area-wide management",
                "Destroy infected material by burial or burning (where permitted)"
            ]
        )
    }
    
    # Enhanced default treatment for unknown diseases
    default_treatment = format_treatment(
        disease_name="General Plant Disease Management",
        severity="Moderate",
        immediate_actions=[
            "Remove and destroy visibly infected plant parts",
            "Isolate affected plants to prevent spread",
            "Apply broad-spectrum fungicide if available",
            "Improve plant growing conditions (drainage, spacing, nutrition)"
        ],
        chemical_treatments=[
            "Broad-spectrum fungicides (Chlorothalonil, Mancozeb)",
            "Copper-based bactericides for bacterial diseases", 
            "Systemic fungicides for established infections",
            "Contact fungicides for protective coverage"
        ],
        organic_options=[
            "Copper sulfate or Bordeaux mixture",
            "Baking soda solutions (1 tsp per quart water)",
            "Neem oil applications",
            "Compost tea for plant immunity boost"
        ],
        prevention_measures=[
            "Practice crop rotation with non-susceptible plants",
            "Use disease-resistant varieties when available",
            "Maintain plant vigor through balanced nutrition",
            "Ensure proper spacing and air circulation",
            "Sanitize tools and equipment between plants"
        ],
        monitoring_tips=[
            "Conduct regular visual inspections",
            "Monitor environmental conditions favoring disease",
            "Keep records of symptoms and treatments applied",
            "Watch for pattern of disease spread"
        ],
        additional_notes=[
            "Consult local agricultural extension services for specific recommendations",
            "Always read and follow pesticide label instructions",
            "Consider professional plant pathology diagnosis for valuable crops"
        ]
    )
    
    # Handle healthy plants
    if disease not in treatments:
        if 'healthy' in disease.lower():
            return format_treatment(
                disease_name="Healthy Plant - Maintenance Care",
                severity="Low",
                immediate_actions=[
                    "Continue current care routine - your plant is healthy!",
                    "Maintain regular monitoring schedule",
                    "Keep consistent watering and nutrition program"
                ],
                prevention_measures=[
                    "Continue monitoring for any signs of disease or pest problems",
                    "Maintain proper watering, avoiding both drought stress and overwatering",
                    "Apply balanced fertilizer according to the plant's specific needs",
                    "Follow good sanitation practices to prevent future problems",
                    "Provide appropriate light levels and temperature for optimal growth"
                ],
                monitoring_tips=[
                    "Weekly visual inspections of leaves, stems, and soil",
                    "Monitor plant growth patterns and vigor",
                    "Check soil moisture levels regularly",
                    "Watch for early signs of nutrient deficiencies"
                ],
                additional_notes=[
                    "Healthy plants are more resistant to diseases and pests",
                    "Preventative care is more effective than reactive treatment",
                    "Keep detailed records of care and observations"
                ]
            )
        else:
            return default_treatment
    
    return treatments.get(disease, default_treatment)

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
    
    user_history = []
    all_detections = []
    
    if mongo_available:
        # MongoDB implementation
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
    else:
        # SQLite implementation
        query_parts = ["user_id = ?"]
        query_params = [session['user_id']]
        
        # Add plant type filter
        if plant_filter != 'all':
            query_parts.append("disease LIKE ?")
            query_params.append(f"{plant_filter}___%")
        
        # Add health status filter
        if health_filter == 'healthy':
            query_parts.append("disease LIKE '%healthy'")
        elif health_filter == 'diseased':
            query_parts.append("disease NOT LIKE '%healthy'")
                
        # Add date range filter
        if date_from:
            if date_to:
                query_parts.append("timestamp BETWEEN ? AND ?")
                query_params.extend([date_from, date_to + " 23:59:59"])
            else:
                query_parts.append("timestamp >= ?")
                query_params.append(date_from)
        elif date_to:
            query_parts.append("timestamp <= ?")
            query_params.append(date_to + " 23:59:59")
        
        # Build the full query
        query_sql = f"SELECT * FROM disease_history WHERE {' AND '.join(query_parts)} ORDER BY timestamp DESC"
        
        # Get filtered history
        sqlite_cursor.execute(query_sql, query_params)
        user_history_rows = sqlite_cursor.fetchall()
        
        # Convert to dictionary format similar to MongoDB
        columns = ['id', 'user_id', 'user_name', 'disease', 'confidence', 'confidence_level', 
                   'treatment', 'image', 'timestamp', 'used_augmentation', 
                   'original_dimensions', 'was_upscaled', 'plant_likelihood']
        
        user_history = []
        for row in user_history_rows:
            user_history.append(dict(zip(columns, row)))
            
        # Get all detections for this user
        sqlite_cursor.execute("SELECT * FROM disease_history WHERE user_id = ?", [session['user_id']])
        all_detection_rows = sqlite_cursor.fetchall()
        all_detections = [dict(zip(columns, row)) for row in all_detection_rows]
    
    # Extract plant types from detections
    plant_types = set()
    for detection in all_detections:
        disease = detection.get('disease', '')
        if '___' in disease:
            plant_type = disease.split('___')[0]
            plant_types.add(plant_type)
    
    # Calculate some statistics
    total_detections = len(all_detections)
    healthy_count = sum(1 for d in all_detections if 'healthy' in str(d.get('disease', '')))
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
        if mongo_available:
            # MongoDB implementation
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
        else:
            # SQLite implementation
            # Check if the detection belongs to the logged-in user
            sqlite_cursor.execute(
                "SELECT user_id FROM disease_history WHERE id = ?", 
                (detection_id,)
            )
            detection = sqlite_cursor.fetchone()
            
            if detection and str(detection[0]) == session['user_id']:
                # Delete the detection
                sqlite_cursor.execute("DELETE FROM disease_history WHERE id = ?", (detection_id,))
                sqlite_conn.commit()
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
        
        if mongo_available:
            # MongoDB implementation
            # Convert string IDs to ObjectId and verify user ownership
            from bson.objectid import ObjectId
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
        else:
            # SQLite implementation
            # Clean and convert detection IDs
            detection_id_list = [id_str.strip() for id_str in detection_ids if id_str.strip()]
            
            # Begin with no successfully deleted records
            deleted_count = 0
            
            # Process each ID individually to check permissions
            for detection_id in detection_id_list:
                # Check if the detection belongs to the logged-in user
                sqlite_cursor.execute(
                    "SELECT id FROM disease_history WHERE id = ? AND user_id = ?", 
                    (detection_id, session['user_id'])
                )
                detection = sqlite_cursor.fetchone()
                
                if detection:
                    # Delete the detection
                    sqlite_cursor.execute("DELETE FROM disease_history WHERE id = ?", (detection_id,))
                    deleted_count += 1
            
            # Commit the transaction
            sqlite_conn.commit()
            
            if deleted_count > 0:
                flash(f"✅ Successfully deleted {deleted_count} detection records.")
            else:
                flash("❌ No records were deleted.")
            
    except Exception as e:
        flash(f"❌ Error deleting detections: {str(e)}")
    
    return redirect(url_for('disease_history'))

# Load the disease model at startup
try:
    print("🚀 Initializing plant disease detection system...")
    if load_disease_model():
        print(f"✅ Disease detection system ready with {len(plant_diseases)} classes")
    else:
        print("⚠️ Disease detection system failed to initialize")
except Exception as e:
    print(f"❌ Error initializing disease detection system: {str(e)}")
    import traceback
    traceback.print_exc()

# Enhanced fertility prediction function using optimized model
def predict_fertilizer_enhanced(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):
    """
    Enhanced fertilizer prediction using optimized model when available
    """
    try:
        if OPTIMIZED_MODEL_AVAILABLE:
            # Use the new optimized model
            result = predict_fertilizer_optimized(
                temperature, humidity, moisture, nitrogen, potassium, phosphorous, 
                soil_type, crop_type
            )
            
            if 'error' not in result:                # Get additional analysis using existing functions
                from soil_fertility_api import get_npk_analysis, suggest_fertilizer_composition
                
                npk_analysis = get_npk_analysis(nitrogen, potassium, phosphorous)
                
                # Create soil data dictionary for the new API
                soil_data = {
                    'nitrogen': nitrogen,
                    'phosphorous': phosphorous,
                    'potassium': potassium,
                    'temperature': temperature,
                    'humidity': humidity,
                    'moisture': moisture,
                    'soil_type': soil_type,
                    'crop_type': crop_type
                }
                
                composition_result = suggest_fertilizer_composition(soil_data)
                recommendations = composition_result['analysis']['recommendations']
                deficiencies = composition_result['analysis']['nutrient_status']['deficiencies']
                
                # Format result to match existing structure
                enhanced_result = {
                    'success': True,
                    'fertilizer': result['fertilizer'],
                    'confidence': result['confidence'],
                    'confidence_percentage': result['confidence_percentage'],
                    'recommendation_quality': result['recommendation_quality'],
                    'npk_analysis': npk_analysis,
                    'recommendations': recommendations,
                    'deficiencies': deficiencies,
                    'model_type': 'optimized',
                    'probabilities': {result['fertilizer']: result['confidence']}  # Simplified for compatibility
                }
                
                return enhanced_result
            else:
                # Fall back to original model if optimized fails
                print(f"Optimized model failed: {result['error']}, falling back to original model")
          # Use original prediction function as fallback
        result = predict_best_fertilizer(
            temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type
        )
        
        if result['success']:
            # Add missing keys for compatibility with the UI
            from soil_fertility_api import get_npk_analysis, suggest_fertilizer_composition
            
            npk_analysis = get_npk_analysis(nitrogen, potassium, phosphorous)
            
            # Create soil data dictionary for the new API
            soil_data = {
                'nitrogen': nitrogen,
                'phosphorous': phosphorous,
                'potassium': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture,
                'soil_type': soil_type,
                'crop_type': crop_type
            }
            
            composition_result = suggest_fertilizer_composition(soil_data)
            recommendations = composition_result['analysis']['recommendations']
            deficiencies = composition_result['analysis']['nutrient_status']['deficiencies']
            
            # Add missing keys to result
            result['npk_analysis'] = npk_analysis
            result['recommendations'] = recommendations
            result['deficiencies'] = deficiencies
            result['model_type'] = 'original'
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Enhanced prediction failed: {str(e)}',
            'model_type': 'error'
        }
        
def predict_image_direct(img):
    """
    Predict disease using the exact same method as app_harvest.py
    This function replicates the predict_image function from app_harvest.py
    """
    import torch
    from torchvision import transforms
    import io
    
    try:
        # Exact same transform as app_harvest.py
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        
        # Convert PIL image to file-like object for app_harvest.py compatibility
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        
        # Exact same processing as app_harvest.py predict_image function
        image = Image.open(io.BytesIO(img_bytes))
        img_t = transform(image)
        img_u = torch.unsqueeze(img_t, 0)

        # Get predictions from model (same as app_harvest.py)
        with torch.no_grad():
            yb = disease_model(img_u)
            # Get probabilities for confidence calculation
            probabilities = torch.nn.functional.softmax(yb, dim=1)
            
            # Pick index with highest probability
            _, preds = torch.max(yb, dim=1)
            predicted_idx = preds[0].item()
            confidence = probabilities[0][predicted_idx].item() * 100  # Convert to percentage
            
            # Get disease name
            prediction = plant_diseases[predicted_idx]
            
            return {
                "success": True,
                "predicted_disease": prediction,
                "predicted_index": predicted_idx,
                "confidence": confidence,
                "all_probabilities": probabilities[0].cpu().numpy()
            }
    
    except Exception as e:
        print(f"❌ Direct prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Direct prediction failed: {str(e)}"
        }

if __name__ == "__main__":
    # Use production-like settings to avoid threading issues with file uploads
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
