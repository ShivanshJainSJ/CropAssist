"""
Soil Fertility API - Machine Learning based fertilizer recommendation system
"""

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and feature columns
try:
    with open('fertilizer_model.pkl', 'rb') as f:
        fertilizer_model = pickle.load(f)
    
    with open('fertilizer_feature_cols.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    print("✅ Fertilizer model and features loaded successfully")
except Exception as e:
    print(f"❌ Error loading fertilizer model: {e}")
    fertilizer_model = None
    feature_columns = None

# Load fertilizer dataset for additional analysis
try:
    fertilizer_data = pd.read_csv('Fertilizer.csv')
    print(f"✅ Fertilizer dataset loaded with {len(fertilizer_data)} records")
except Exception as e:
    print(f"❌ Error loading fertilizer dataset: {e}")
    fertilizer_data = None

def predict_best_fertilizer(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):
    """
    Predict the best fertilizer based on soil and environmental conditions
    """
    try:
        if fertilizer_model is None or feature_columns is None:
            return {
                'success': False,
                'error': 'Fertilizer model not loaded properly'
            }
        
        # Create input data
        input_data = {
            'Temparature': temperature,
            'Humidity': humidity,
            'Moisture': moisture,
            'Nitrogen': nitrogen,
            'Potassium': potassium,
            'Phosphorous': phosphorous
        }
        
        # Add soil type encoding
        soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
        for soil in soil_types:
            input_data[f'Soil_{soil}'] = 1 if soil_type == soil else 0
        
        # Add crop type encoding
        crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 
                     'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']
        for crop in crop_types:
            input_data[f'Crop_{crop}'] = 1 if crop_type == crop else 0
        
        # Create dataframe with correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Get prediction
        if isinstance(fertilizer_model, np.ndarray):
            # If model is a lookup table/array, use nearest neighbor approach
            prediction = predict_from_lookup_table(temperature, humidity, moisture, 
                                                 nitrogen, potassium, phosphorous, 
                                                 soil_type, crop_type)
        else:
            # If it's a trained ML model
            prediction = fertilizer_model.predict(input_df)[0]
        
        # Calculate confidence (simplified)
        confidence = 0.85  # Default confidence
        
        return {
            'success': True,
            'fertilizer': prediction,
            'confidence': confidence,
            'confidence_percentage': round(confidence * 100, 1),
            'recommendation_quality': 'Good' if confidence > 0.8 else 'Moderate',
            'probabilities': {prediction: confidence}
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }

def predict_from_lookup_table(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):
    """
    Predict fertilizer using dataset lookup when model is not available
    """
    if fertilizer_data is None:
        return "Urea"  # Default fallback
    
    # Find similar conditions in dataset
    df = fertilizer_data.copy()
    
    # Calculate similarity score
    df['temp_diff'] = abs(df['Temparature'] - temperature)
    df['humidity_diff'] = abs(df['Humidity'] - humidity)
    df['moisture_diff'] = abs(df['Moisture'] - moisture)
    df['nitrogen_diff'] = abs(df['Nitrogen'] - nitrogen)
    df['potassium_diff'] = abs(df['Potassium'] - potassium)
    df['phosphorous_diff'] = abs(df['Phosphorous'] - phosphorous)
    
    # Filter by soil and crop type
    df_filtered = df[
        (df['Soil Type'] == soil_type) & 
        (df['Crop Type'] == crop_type)
    ]
    
    if len(df_filtered) == 0:
        # Fallback to soil type only
        df_filtered = df[df['Soil Type'] == soil_type]
    
    if len(df_filtered) == 0:
        # Final fallback to full dataset
        df_filtered = df
    
    # Calculate total difference score
    df_filtered['total_diff'] = (
        df_filtered['temp_diff'] + 
        df_filtered['humidity_diff'] + 
        df_filtered['moisture_diff'] + 
        df_filtered['nitrogen_diff'] + 
        df_filtered['potassium_diff'] + 
        df_filtered['phosphorous_diff']
    )
    
    # Get the fertilizer with minimum difference
    best_match = df_filtered.loc[df_filtered['total_diff'].idxmin()]
    return best_match['Fertilizer Name']

def get_fertilizer_recommendation(soil_data):
    """
    Get detailed fertilizer recommendation (legacy function for compatibility)
    """
    return predict_best_fertilizer(
        soil_data.get('temperature', 25),
        soil_data.get('humidity', 60),
        soil_data.get('moisture', 40),
        soil_data.get('nitrogen', 20),
        soil_data.get('potassium', 15),
        soil_data.get('phosphorous', 25),
        soil_data.get('soil_type', 'Loamy'),
        soil_data.get('crop_type', 'Wheat')
    )

def get_npk_analysis(nitrogen, potassium, phosphorous):
    """
    Analyze NPK levels and provide recommendations
    """
    analysis = {
        'nitrogen': {
            'value': nitrogen,
            'status': 'optimal',
            'recommendation': 'Maintain current levels'
        },
        'phosphorous': {
            'value': phosphorous,
            'status': 'optimal',
            'recommendation': 'Maintain current levels'
        },
        'potassium': {
            'value': potassium,
            'status': 'optimal',
            'recommendation': 'Maintain current levels'
        }
    }
    
    # Nitrogen analysis
    if nitrogen < 10:
        analysis['nitrogen']['status'] = 'deficient'
        analysis['nitrogen']['recommendation'] = 'Increase nitrogen content with urea or ammonium fertilizers'
    elif nitrogen > 40:
        analysis['nitrogen']['status'] = 'excess'
        analysis['nitrogen']['recommendation'] = 'Reduce nitrogen to prevent leaf burn and environmental issues'
    
    # Phosphorous analysis
    if phosphorous < 5:
        analysis['phosphorous']['status'] = 'deficient'
        analysis['phosphorous']['recommendation'] = 'Add phosphate fertilizers like DAP or SSP'
    elif phosphorous > 45:
        analysis['phosphorous']['status'] = 'excess'
        analysis['phosphorous']['recommendation'] = 'Reduce phosphorous to prevent nutrient imbalance'
    
    # Potassium analysis
    if potassium < 5:
        analysis['potassium']['status'] = 'deficient'
        analysis['potassium']['recommendation'] = 'Apply potash fertilizers like MOP or SOP'
    elif potassium > 25:
        analysis['potassium']['status'] = 'excess'
        analysis['potassium']['recommendation'] = 'Reduce potassium to maintain balanced nutrition'
    
    return analysis

def suggest_fertilizer_composition(soil_data):
    """
    Suggest optimal fertilizer composition based on soil analysis
    """
    nitrogen = soil_data.get('nitrogen', 20)
    phosphorous = soil_data.get('phosphorous', 25)
    potassium = soil_data.get('potassium', 15)
    temperature = soil_data.get('temperature', 25)
    humidity = soil_data.get('humidity', 60)
    moisture = soil_data.get('moisture', 40)
    
    # Get NPK analysis
    npk_analysis = get_npk_analysis(nitrogen, potassium, phosphorous)
    
    # Determine deficiencies
    deficiencies = []
    recommendations = []
    
    for nutrient, data in npk_analysis.items():
        if data['status'] == 'deficient':
            deficiencies.append(nutrient.capitalize())
            recommendations.append(data['recommendation'])
        elif data['status'] == 'excess':
            recommendations.append(data['recommendation'])
    
    # Environmental recommendations
    if temperature > 35:
        recommendations.append("Consider irrigation due to high temperature")
    if humidity < 40:
        recommendations.append("Increase irrigation frequency due to low humidity")
    if moisture < 30:
        recommendations.append("Immediate irrigation required due to low soil moisture")
    
    # General recommendations
    if not deficiencies:
        recommendations.append("Soil nutrient levels are well balanced")
        recommendations.append("Apply maintenance fertilizer as per crop requirements")
    
    return {
        'success': True,
        'analysis': {
            'npk_status': npk_analysis,
            'nutrient_status': {
                'deficiencies': deficiencies,
                'overall_health': 'Good' if len(deficiencies) == 0 else 'Needs attention'
            },
            'recommendations': recommendations,
            'environmental_factors': {
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture
            }
        }
    }

def get_soil_health_score(nitrogen, phosphorous, potassium, temperature, humidity, moisture):
    """
    Calculate overall soil health score
    """
    scores = []
    
    # NPK scoring (0-100)
    n_score = min(100, max(0, 100 - abs(nitrogen - 25) * 2))  # Optimal around 25
    p_score = min(100, max(0, 100 - abs(phosphorous - 20) * 2))  # Optimal around 20
    k_score = min(100, max(0, 100 - abs(potassium - 15) * 2))  # Optimal around 15
    
    # Environmental scoring
    temp_score = min(100, max(0, 100 - abs(temperature - 28) * 3))  # Optimal around 28°C
    humidity_score = min(100, max(0, 100 - abs(humidity - 60) * 1.5))  # Optimal around 60%
    moisture_score = min(100, max(0, 100 - abs(moisture - 45) * 2))  # Optimal around 45%
    
    # Overall score
    overall_score = (n_score + p_score + k_score + temp_score + humidity_score + moisture_score) / 6
    
    return {
        'overall_score': round(overall_score, 1),
        'nitrogen_score': round(n_score, 1),
        'phosphorous_score': round(p_score, 1),
        'potassium_score': round(k_score, 1),
        'temperature_score': round(temp_score, 1),
        'humidity_score': round(humidity_score, 1),
        'moisture_score': round(moisture_score, 1),
        'grade': get_grade_from_score(overall_score)
    }

def get_grade_from_score(score):
    """Convert numerical score to letter grade"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# Test the functions if run directly
if __name__ == "__main__":
    # Test prediction
    test_result = predict_best_fertilizer(
        temperature=28, humidity=65, moisture=40,
        nitrogen=20, potassium=15, phosphorous=25,
        soil_type="Loamy", crop_type="Wheat"
    )
    print("Test prediction result:", test_result)
    
    # Test NPK analysis
    npk_result = get_npk_analysis(20, 15, 25)
    print("NPK analysis result:", npk_result)