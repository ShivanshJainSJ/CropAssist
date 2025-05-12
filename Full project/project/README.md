# CropAssist Solution - Agritech Full Stack Project

![CropAssist Logo](https://cdn-icons-png.flaticon.com/512/2913/2913461.png)

## 📋 Project Overview

CropAssist Solution is a comprehensive agricultural technology platform designed to assist farmers in making data-driven decisions to optimize their farming practices. The application leverages machine learning algorithms to provide personalized recommendations for crop selection, soil fertility analysis, and plant disease detection.

## ✨ Features

### 🌾 Crop Recommendation System
- Suggests optimal crops based on soil nutrient levels (N, P, K)
- Considers environmental factors like temperature, humidity, pH, and rainfall
- Location-specific recommendations for different states and cities in India

### 🔍 Plant Disease Detection
- Upload images of plant leaves to identify diseases
- Utilizes a deep learning model (MobileNetV2) for accurate disease classification
- Maintains history of previous disease detections for tracking

### 💧 Soil Fertility Analysis
- Analyzes soil composition and recommends appropriate fertilizers
- Optimizes nutrient ratios for specific crops
- Provides cost-effective fertilizer recommendations

### 👤 User Authentication System
- Secure login and registration
- Personalized recommendations based on user history
- Save and track historical data

### 🌐 Multilingual Support
- Available in English and Hindi
- Enhances accessibility for farmers across India

## 🛠️ Technology Stack

### Backend
- Flask (Python web framework)
- TensorFlow/Keras (ML models)
- MongoDB (Database)
- NumPy, Pandas, SciPy (Data processing)

### Frontend
- HTML, Tailwind CSS
- JavaScript
- Responsive design for mobile and desktop

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- MongoDB
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CropAssist-solution.git
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

5. **Initialize the database**
   ```bash
   python setup_db.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
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
