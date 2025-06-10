# Plant Disease Detection & Soil Fertility System

## 📋 Project Overview

This is a comprehensive agricultural technology platform that assists farmers with plant disease detection and soil fertility analysis. The system uses machine learning to provide accurate disease identification and fertilizer recommendations.

## ✨ Features

### 🔍 Plant Disease Detection
- Upload images of plant leaves to identify diseases
- Utilizes EfficientNetB0 deep learning model for accurate disease classification
- Supports 39 different plant disease categories
- Maintains history of previous disease detections

### 💧 Soil Fertility Analysis
- Analyzes soil composition and recommends appropriate fertilizers
- Optimizes nutrient ratios (N, P, K) for specific crops
- Provides cost-effective fertilizer recommendations
- Temperature and humidity-based adjustments

### 📧 Email Notification System
- **Welcome Emails**: Professional welcome messages for new user registrations
- **Login Notifications**: Security alerts for login activities  
- **Responsive Design**: Mobile-friendly HTML email templates
- **Multi-Provider Support**: Works with Gmail, Outlook, Yahoo, and custom SMTP servers
- **Secure Configuration**: Environment variable-based setup with app password support

### 🎯 Input Validation System
- **Real-time Validation**: Instant feedback for form inputs
- **Agricultural Ranges**: Scientifically accurate min/max values for all parameters
- **Visual Feedback**: Color-coded input borders and warning messages
- **Comprehensive Coverage**: Validates crop recommendation, fertility analysis, and advanced forms

## 🛠️ Technology Stack

### Backend
- Flask (Python web framework)
- TensorFlow/Keras (Deep learning models)
- SQLite (Database)
- NumPy, Pandas, SciPy (Data processing)
- PIL (Image processing)

### Frontend
- HTML, CSS, JavaScript
- Bootstrap for responsive design

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MongoDB
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash   git clone https://github.com/yourusername/CropAssist-solution.git
   cd CropAssist-solution
   ```

2. **Create and activate virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MongoDB**
   - Make sure MongoDB is running on your system
   - The application will connect to MongoDB at `mongodb://localhost:27017/`

5. **Set up Email Notifications (Optional)**
   ```bash
   # Copy the email configuration template
   copy .env.template .env
   
   # Edit .env file with your email credentials
   # For Gmail (recommended):
   EMAIL_ADDRESS=your-email@gmail.com
   EMAIL_PASSWORD=your-16-character-app-password
   ```
   
   **Gmail Setup Instructions:**
   - Enable 2-Factor Authentication on your Gmail account
   - Go to Google Account → Security → 2-Step Verification → App passwords
   - Generate a new app password for "Mail"
   - Use the 16-character password in your .env file
   
   **Test Email Configuration:**
   ```bash
   python verify_email.py
   ```

6. **Initialize the database**
   ```bash
   python setup_db.py
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

8. **Access the application**
   - Open your browser and navigate to `http://127.0.0.1:5000/`

## 📊 Machine Learning Models

### Crop Recommendation Model
- Random Forest Classifier trained on crop recommendation dataset
- Features: N, P, K, temperature, humidity, pH, rainfall
- Located in `model.pkl`

### Plant Disease Detection Model
- MobileNetV2 architecture fine-tuned on plant disease images
- Capable of identifying various plant diseases
- Located in `plant_disease_model.h5`

## 📱 Usage Guide

1. **Crop Recommendation**
   - Navigate to the Crop Recommendation page
   - Enter soil nutrient values (N, P, K)
   - Provide environmental data
   - Select your state and city
   - Submit to get personalized crop recommendations

2. **Disease Detection**
   - Go to the Disease Detection page
   - Upload an image of the plant leaf
   - The system will analyze and identify any diseases
   - View previous disease detections in the history section

3. **Soil Fertility Analysis**
   - Access the Fertility page
   - Enter soil composition details
   - Receive optimized fertilizer recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For any questions or suggestions, please reach out to us at [your-email@example.com](mailto:your-email@example.com).

---

Developed with ❤️ for Indian farmers
