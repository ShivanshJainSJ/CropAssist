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
from bson.objectid import ObjectId  # Add this import for MongoDB ObjectId
from functools import wraps
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import io
import base64
import random  # Added for random sampling in test-time augmentation
try:
    import cv2  # Import OpenCV for image quality assessment
except ImportError:
    pass  # CV2 is optional, fallback mechanisms are in place
from soil_fertility_api import predict_best_fertilizer, get_fertilizer_recommendation

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
            flash("? Please log in to access this page.")
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
            flash("? Passwords do not match!")
            return redirect(url_for('register'))
        
        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            flash("? Email already registered!")
            return redirect(url_for('register'))
        
        # Save the user data to MongoDB
        new_user = {
            'name': name,
            'email': email,
            'password': password  # In production, use password hashing!
        }
        users_collection.insert_one(new_user)
        
        flash("? Registration successful! Please log in.")
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
            
            flash("? Login successful!")
            return redirect(url_for('home'))
        else:
            result = "? No feasible fertilizer combination found."
            flash("? Invalid email or password. Please try again.")
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
            crop_type = request.form['crop_type'].strip().lower()
            
            # Validate inputs
            if nitrogen <= 0 or phosphorous <= 0 or potassium <= 0:
                flash("? Nutrient values must be greater than zero.")                
                return render_template('advanced_fertility.html', states=states, states_and_cities=states_and_cities,
                                   fertility_result="? All nutrient values (Nitrogen, Phosphorous, Potassium) must be greater than zero.",
                                   soil_types=soil_types, crop_types=crop_types)
            
            # Predict fertilizer and get recommendations
            result = predict_best_fertilizer(
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
                                    nitrogen=nitrogen,
                                    phosphorous=phosphorous,
                                    potassium=potassium,                                    soil_type=soil_type,
                                    crop_type=crop_type,
                                    success=True)
            else:
                # Handle error
                flash(f"? {result['error']}")
                return render_template('advanced_fertility.html', 
                                      states=states, 
                                      states_and_cities=states_and_cities,
                                      fertility_result=f"? Error: {result['error']}",
                                      success=False)
                
        except ValueError as e:
            # Handle input conversion errors
            error_message = f"? Invalid input values: {str(e)}"
            flash(error_message)
            return render_template('advanced_fertility.html', 
                                  states=states, 
                                  states_and_cities=states_and_cities,
                                  fertility_result=error_message,
                                  success=False)
        except Exception as e:
            # Handle other unexpected errors
            error_message = f"? An error occurred: {str(e)}"
            flash(error_message)
            return render_template('advanced_fertility.html', 
                                  states=states, 
                                  states_and_cities=states_and_cities,
                                  fertility_result=error_message,
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

    if name and email and feedback:
        feedback_doc = {
            "name": name,
            "email": email,
            "feedback": feedback
        }
        try:
            feedback_collection.insert_one(feedback_doc)
            flash("? Feedback submitted successfully!")
        except Exception as e:
            flash(f"? Failed to submit feedback: {str(e)}")
    else:
        flash("? Please fill out all fields.")

    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash("? You have been successfully logged out.")
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

# Disease detection model - we'll use EfficientNetB0 which is more accurate than MobileNetV2
disease_model = None

def load_disease_model():
    global disease_model
    try:
        # Check if model is already loaded
        if disease_model is None:
            print("Loading plant disease detection model...")
            
            # Import EfficientNet only when needed (helps with initial app loading time)
            try:
                from tensorflow.keras.applications import EfficientNetB0
                print("Using EfficientNetB0 architecture")
            except ImportError:
                # Fallback to MobileNetV2 if EfficientNet is not available
                from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
                print("EfficientNetB0 not available, falling back to MobileNetV2")
                base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                model_architecture = "MobileNetV2"
            else:
                # Create an EfficientNetB0 model pretrained on ImageNet
                base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                model_architecture = "EfficientNetB0"
            
            # Add our own classifier on top with dropout for better generalization
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.3)(x)  # Increased dropout to reduce overfitting
            x = tf.keras.layers.Dense(512, activation='relu')(x)  # Wider layer for better feature extraction
            x = tf.keras.layers.BatchNormalization()(x)  # Add batch normalization for more stable training
            x = tf.keras.layers.Dropout(0.3)(x)  # Add another dropout layer
            predictions = tf.keras.layers.Dense(len(plant_diseases), activation='softmax')(x)
            
            # Create the full model
            disease_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
            
            # Load weights if available, otherwise use base model
            try:
                disease_model.load_weights('plant_disease_model.h5')
                print(f"Loaded plant disease model weights for {model_architecture}")
            except:
                print(f"Using base {model_architecture} model - fine-tuning needed for better results")
                
            # Compile the model for better performance
            disease_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Transfer learning: only train the top layers
            for layer in base_model.layers:
                layer.trainable = False
        
        return True
    except Exception as e:
        print(f"Error loading disease model: {str(e)}")
        return False

# Preprocess image for disease detection with enhanced handling
def preprocess_image(img):
    """
    Preprocess image for disease detection with enhanced handling of different image types
    and sizes, including upscaling of low-resolution images.
    
    Args:
        img: PIL Image object
        
    Returns:
        img_array: Preprocessed numpy array ready for model input
        upscaled: Boolean indicating whether the image was upscaled
    """
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
        except (ImportError, AttributeError):
            # If OpenCV is not available, use PIL's high-quality upscaling
            scale_factor = 224 / min(width, height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)
    
    # Resize image to target size with high quality
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(img)
    
    # Make sure image has 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Check if image is too dark or too bright
    mean_brightness = np.mean(img_array)
    if mean_brightness < 50:  # Very dark image
        # Increase brightness
        gamma = 0.7  # Less than 1 brightens the image
        img_array = np.clip(((img_array / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    elif mean_brightness > 220:  # Very bright image
        # Decrease brightness
        gamma = 1.3  # Greater than 1 darkens the image
        img_array = np.clip(((img_array / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    
    # Apply specific preprocessing based on the model architecture being used
    try:
        # Try to use EfficientNet preprocessing
        from tensorflow.keras.applications.efficientnet import preprocess_input
    except ImportError:
        # Fall back to MobileNetV2 preprocessing
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    # Apply model-specific preprocessing
    img_array = preprocess_input(img_array.astype(np.float32))
    
    return img_array, upscaled

# Function to perform test-time augmentation for better inference
def test_time_augmentation(img, model, num_augmentations=5):
    """
    Apply multiple augmentations to the input image and average predictions
    for more robust inference.
    
    Args:
        img: Input image array of shape (224, 224, 3)
        model: Loaded model for prediction
        num_augmentations: Number of augmentations to perform
    
    Returns:
        avg_predictions: Averaged prediction array
    """
    import numpy as np
    from scipy.ndimage import rotate
    
    # Initialize predictions array
    all_predictions = []
    
    # Original image prediction
    orig_img = np.expand_dims(img, axis=0)
    orig_pred = model.predict(orig_img, verbose=0)  # Set verbose=0 to avoid unnecessary output
    all_predictions.append(orig_pred[0])
    
    # Define augmentation types - more varied for better robustness
    augmentations = [
        # Horizontal flip
        lambda x: np.fliplr(x),
        # Vertical flip
        lambda x: np.flipud(x),
        # 90 degree rotation
        lambda x: np.rot90(x, k=1),
        # 180 degree rotation
        lambda x: np.rot90(x, k=2),
        # 270 degree rotation
        lambda x: np.rot90(x, k=3),
        # Brightness increase by 10%
        lambda x: np.clip(x * 1.1, 0, 255),
        # Brightness decrease by 10%
        lambda x: np.clip(x * 0.9, 0, 255),
    ]
    
    # Random selection of augmentation types based on num_augmentations
    import random
    selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
    
    # Apply selected augmentations and get predictions
    for aug_func in selected_augmentations:
        # Apply augmentation
        aug_img = aug_func(img.copy())
        
        # Add to batch
        aug_img = np.expand_dims(aug_img, axis=0)
        
        # Get prediction
        pred = model.predict(aug_img, verbose=0)
        all_predictions.append(pred[0])
    
    # Average predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    
    return avg_predictions

def assess_image_quality(img):
    """
    Assess the quality of an uploaded image to determine if it's suitable for disease detection.
    
    Args:
        img: PIL Image object
        
    Returns:
        dict: Dictionary containing quality assessment results
    """
    import numpy as np
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Initialize result dictionary
    quality = {
        "is_suitable": True,
        "warnings": [],
        "suggestions": [],
        "can_be_enhanced": False
    }
    
    # 1. Check image size
    width, height = img.size
    if width < 224 or height < 224:
        quality["warnings"].append(f"Image resolution is low ({width}x{height} pixels)")
        quality["suggestions"].append("We'll automatically enhance the image, but results may be less accurate")
        quality["can_be_enhanced"] = True
        # Don't set is_suitable to False since we can now handle this case
    
    # 2. Check brightness
    mean_brightness = np.mean(img_array)
    if mean_brightness < 50:
        quality["warnings"].append("Image is too dark")
        quality["suggestions"].append("Take photo in better lighting conditions")
    elif mean_brightness > 220:
        quality["warnings"].append("Image is too bright or overexposed")
        quality["suggestions"].append("Avoid direct sunlight or use diffused lighting")
    
    # 3. Check contrast
    # Calculate standard deviation of pixel values as a simple contrast measure
    std_dev = np.std(img_array)
    if std_dev < 25:
        quality["warnings"].append("Image has low contrast")
        quality["suggestions"].append("Ensure the plant is clearly visible against the background")
    
    # 4. Check blur using Laplacian variance
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # Convert to grayscale for blur detection
        gray = np.mean(img_array[:,:,:3], axis=2).astype(np.uint8)
        
        # Calculate variance of Laplacian to detect blur
        try:
            import cv2
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:
                quality["warnings"].append("Image appears to be blurry")
                quality["suggestions"].append("Hold camera steady and ensure the plant is in focus")
        except ImportError:
            # Skip if OpenCV is not available
            pass
    
    # 5. Check if there's enough green in the image (plant detection)
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # Check if there's enough green in the image
        # Simple approach: calculate ratio of green pixels
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        green_mask = (g > r + 20) & (g > b + 20)
        green_pixel_ratio = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
        
        if green_pixel_ratio < 0.1:
            quality["warnings"].append("Image may not contain enough plant material")
            quality["suggestions"].append("Ensure the plant leaves are clearly visible in the frame")
    
    # Final suitability assessment - more than 2 serious issues indicates a problematic image
    # but exclude resolution issue since we can now handle it
    serious_issues = [w for w in quality["warnings"] if "resolution" not in w.lower()]
    if len(serious_issues) > 2:
        quality["is_suitable"] = False
    
    return quality

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
        
        try:
            # Open and process the image
            img = Image.open(file.stream)
            
            # Convert to RGB if the image is not in RGB format (e.g. RGBA PNG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Assess image quality
            quality = assess_image_quality(img)
            
            # Store original image dimensions
            original_width, original_height = img.size
            
            # Create a copy for display
            display_img = img.copy()
            
            # Preprocess the image - now returns whether upscaling was needed
            processed_img, was_upscaled = preprocess_image(img)
            
            # Update quality assessment if image was upscaled
            if was_upscaled:
                quality["was_upscaled"] = True
            
            # Check if test-time augmentation should be used
            use_augmentation = request.form.get('use_augmentation', 'false') == 'true'
            
            if use_augmentation:
                # Perform test-time augmentation for more robust prediction
                avg_predictions = test_time_augmentation(processed_img, disease_model, num_augmentations=7)
            else:
                # Standard prediction without augmentation
                img_array = np.expand_dims(processed_img, axis=0)
                predictions = disease_model.predict(img_array, verbose=0)
                avg_predictions = predictions[0]
            
            # Get the top prediction and confidence
            top_prediction_idx = np.argmax(avg_predictions)
            confidence = float(avg_predictions[top_prediction_idx] * 100)
            
            # Set confidence thresholds for different levels
            HIGH_CONFIDENCE_THRESHOLD = 75.0
            MEDIUM_CONFIDENCE_THRESHOLD = 60.0
            
            # Get the predicted disease label
            predicted_disease = plant_diseases[top_prediction_idx]
            
            # Determine confidence level and message
            confidence_level = "high"
            confidence_message = None
            low_confidence = False
            
            if confidence < MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_level = "low"
                low_confidence = True
                confidence_message = f"Low confidence prediction ({confidence:.1f}%). Consider taking another photo with better lighting and clear focus."
            elif confidence < HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "medium"
            
            # Check if prediction has very low confidence (below 40%)
            if confidence < 40.0:
                # Add additional warning for very uncertain predictions
                secondary_message = "The model is very uncertain about this prediction. Results may be unreliable."
            else:
                secondary_message = None
                
            # Append any confidence adjustments for enhanced images
            if was_upscaled:
                confidence_adjustment = "Note: This prediction is based on an enhanced low-resolution image, which may affect accuracy."
                if confidence_message:
                    confidence_message += " " + confidence_adjustment
                else:
                    confidence_message = confidence_adjustment
            
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
                for idx in top_3_indices
            ]
            
            # Save detection result to history
            if is_logged_in():
                detection_history = {
                    "user_id": session['user_id'],
                    "user_name": session.get('name', 'User'),
                    "disease": predicted_disease,
                    "confidence": confidence,
                    "confidence_level": confidence_level,
                    "treatment": treatment,
                    "image": img_str,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "used_augmentation": use_augmentation,
                    "original_dimensions": f"{original_width}x{original_height}",
                    "was_upscaled": was_upscaled
                }                
                disease_history_collection.insert_one(detection_history)
              # Include quality warnings in the response if there are any
            quality_warnings = None
            if quality["warnings"]:
                quality_warnings = quality["warnings"]
            
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
                                  quality_warnings=quality_warnings,
                                  
                                  was_upscaled=was_upscaled,
                                  original_dimensions=f"{original_width}x{original_height}")
        
        except Exception as e:
            flash(f"? Error processing image: {str(e)}")
            return redirect(url_for('diseases'))
    
    return redirect(url_for('diseases'))

def get_treatment_recommendation(disease):
    # Dictionary of treatments for various plant diseases
    treatments = {
        'Apple___Apple_scab': """
        <h4>Apple Scab Treatment</h4>
        <h5>Immediate Actions:</h5>
        <ul>
            <li>Apply fungicides containing myclobutanil, captan, or sulfur as soon as symptoms appear</li>
            <li>Remove and destroy infected leaves and fruit from the ground and tree</li>
        </ul>
        <h5>Prevention:</h5>
        <ul>
            <li>Apply preventative fungicide sprays in early spring before symptoms appear</li>
            <li>Ensure proper spacing between trees for adequate air circulation</li>
            <li>Prune trees during dormancy to improve air flow</li>
            <li>Consider planting scab-resistant apple varieties for future plantings</li>
        </ul>
        """,
        'Apple___Black_rot': """
        <h4>Apple Black Rot Treatment</h4>
        <h5>Immediate Actions:</h5>
        <ul>
            <li>Apply fungicides containing thiophanate-methyl, captan, or myclobutanil</li>
            <li>Remove mummified fruits, cankers, and infected branches</li>
        </ul>
        <h5>Prevention:</h5>
        <ul>
            <li>Prune out dead or diseased wood during winter dormancy</li>
            <li>Apply fungicide sprays on a 10-14 day schedule from bud break to mid-summer</li>
            <li>Maintain tree vigor with proper fertilization and watering</li>
            <li>Clean pruning tools with alcohol between cuts</li>
        </ul>
        """,
        'Apple___Cedar_apple_rust': """
        <h4>Cedar Apple Rust Treatment</h4>
        <h5>Immediate Actions:</h5>
        <ul>
            <li>Apply fungicides containing mancozeb, myclobutanil, or propiconazole</li>
            <li>Remove galls from nearby juniper or cedar trees if possible</li>
        </ul>
        <h5>Prevention:</h5>
        <ul>
            <li>Plant apple varieties resistant to cedar apple rust</li>
            <li>Remove nearby juniper or cedar trees (alternate hosts) if practical</li>
            <li>Apply protective fungicides starting at pink bud stage</li>
            <li>Continue applications on a 7-10 day schedule until 2-3 weeks after petal fall</li>
        </ul>
        """,
        'Apple___healthy': """
        <h4>Healthy Apple Plant</h4>
        <p>Your apple plant appears healthy! Here are some tips to maintain its health:</p>
        <ul>
            <li>Continue regular monitoring for early signs of disease or pests</li>
            <li>Maintain a proper pruning schedule to encourage air circulation</li>
            <li>Apply balanced fertilizer based on soil test recommendations</li>
            <li>Water deeply but infrequently, avoiding overhead irrigation</li>
            <li>Apply preventative fungicide sprays according to local extension guidelines</li>
        </ul>
        """,
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': """
        <h4>Gray Leaf Spot (Cercospora) Treatment</h4>
        <h5>Immediate Actions:</h5>
        <ul>
            <li>Apply foliar fungicides containing pyraclostrobin, azoxystrobin, or propiconazole</li>
            <li>Timing of application is critical - apply when disease is first detected</li>
        </ul>
        <h5>Prevention:</h5>
        <ul>
            <li>Rotate crops - avoid corn after corn in the same field</li>
            <li>Choose resistant hybrids for future plantings</li>
            <li>Practice conservation tillage to reduce surface residue</li>
            <li>Improve field drainage to reduce humidity and leaf wetness</li>
        </ul>
        """,
        'Corn_(maize)___Common_rust': """
        <h4>Common Rust Treatment</h4>
        <h5>Immediate Actions:</h5>
        <ul>
            <li>Apply foliar fungicides containing azoxystrobin, pyraclostrobin, or triazoles</li>
            <li>For organic options, consider copper-based fungicides</li>
        </ul>
        <h5>Prevention:</h5>
        <ul>
            <li>Plant rust-resistant corn hybrids</li>
            <li>Early planting can help avoid periods of high rust pressure</li>
            <li>Maintain balanced soil fertility - excessive nitrogen can increase susceptibility</li>
            <li>Avoid overhead irrigation that increases leaf wetness periods</li>
        </ul>
        """,
        'Corn_(maize)___Northern_Leaf_Blight': """
        <h4>Northern Leaf Blight Treatment</h4>
        <h5>Immediate Actions:</h5>
        <ul>
            <li>Apply fungicides containing propiconazole, azoxystrobin, or pyraclostrobin</li>
            <li>Application timing is crucial - apply at disease onset for best control</li>
        </ul>
        <h5>Prevention:</h5>
        <ul>
            <li>Rotate crops - avoid corn after corn for at least one year</li>
            <li>Select hybrids with resistance to Northern Leaf Blight</li>
            <li>Practice deep tillage to bury crop residue that harbors the fungus</li>
            <li>Optimize plant health through proper fertility and irrigation</li>
        </ul>
        """,
        'Corn_(maize)___healthy': """
        <h4>Healthy Corn Plant</h4>
        <p>Your corn plant appears healthy! Here are some tips to maintain its health:</p>
        <ul>
            <li>Maintain optimal fertilization based on soil tests</li>
            <li>Continue monitoring for early disease symptoms or pest damage</li>
            <li>Ensure adequate but not excessive irrigation</li>
            <li>Control weeds that compete for nutrients and can harbor diseases</li>
            <li>Scout regularly for signs of nutrient deficiencies</li>
        </ul>
        """
    }
    
    # Add a default set of general recommendations for diseases not specifically covered
    default_treatment = """
    <h4>General Disease Management</h4>
    <h5>Immediate Actions:</h5>
    <ul>
        <li>Remove and destroy infected plant parts to reduce disease spread</li>
        <li>Apply appropriate fungicide or bactericide based on the specific disease</li>
        <li>Ensure proper plant spacing and ventilation to reduce humidity</li>
    </ul>
    <h5>Prevention:</h5>
    <ul>
        <li>Practice crop rotation with non-susceptible plants</li>
        <li>Use disease-resistant varieties when available</li>
        <li>Maintain plant vigor through balanced nutrition</li>
        <li>Avoid overhead watering to keep foliage dry</li>
        <li>Sanitize tools and equipment between plants</li>
    </ul>
    <p>For specific treatment advice, consult your local agricultural extension service.</p>
    """
    
    # For healthy plants or unknown diseases
    if disease not in treatments:
        if 'healthy' in disease.lower():
            return """
            <h4>Healthy Plant</h4>
            <p>Your plant appears to be healthy! Here are some general care tips:</p>
            <ul>
                <li>Continue monitoring for any signs of disease or pest problems</li>
                <li>Maintain proper watering, avoiding both drought stress and overwatering</li>
                <li>Apply balanced fertilizer according to the plant's specific needs</li>
                <li>Follow good sanitation practices to prevent future problems</li>
                <li>Provide appropriate light levels and temperature for optimal growth</li>
            </ul>
            """
        else:
            return default_treatment
    
    return treatments.get(disease, default_treatment)

@app.route('/disease_history')
@login_required
def disease_history():
    if not is_logged_in():
        flash("? Please log in to view your disease detection history.")
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
        flash("? Please log in to manage your disease detection history.")
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
            flash("? Detection record deleted successfully.")
        else:
            flash("? You don't have permission to delete this record.")
    except Exception as e:
        flash(f"? Error deleting detection: {str(e)}")
    
    return redirect(url_for('disease_history'))

@app.route('/batch_delete_detections', methods=['POST'])
@login_required
def batch_delete_detections():
    if not is_logged_in():
        flash("? Please log in to manage your disease detection history.")
        return redirect(url_for('login'))
    
    try:
        # Get the list of IDs to delete
        detection_ids = request.form.get('detection_ids', '').split(',')
        if not detection_ids or detection_ids[0] == '':
            flash("? No records selected for deletion.")
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
                flash(f"? Successfully deleted {result.deleted_count} detection records.")
            else:
                flash("? No records were deleted.")
        else:
            flash("? You don't have permission to delete these records.")
            
    except Exception as e:
        flash(f"? Error deleting detections: {str(e)}")
    
    return redirect(url_for('disease_history'))

if __name__ == "__main__":
    app.run(debug=True)
