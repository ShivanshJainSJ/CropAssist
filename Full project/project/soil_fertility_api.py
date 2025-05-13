"""
API functions for soil fertility prediction to be used in the Flask app.
This file contains functions to integrate the fertility prediction model with the app.py Flask application.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import linprog

# Paths to model and data files
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fertilizer_model.pkl')
FEATURE_COLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fertilizer_feature_cols.pkl')
IDEAL_DF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FertilizerData1.csv')
FERTILIZER_DF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fertilizer_composition.csv')

# Check if model exists
if not os.path.exists(MODEL_PATH):
    # Add the parent directory to sys.path to import the soil_fertility_predictor module
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    print("Model not found. Please run soil_fertility_predictor.py separately to create the model first.")
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Please run the soil_fertility_predictor.py script to create the model.")
else:
    print(f"Found existing model at {MODEL_PATH}")

# Import prediction functions from soil_fertility_predictor
try:
    # Add the parent directory to sys.path to import from soil_fertility_predictor
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Import specific functions only, not the whole module
    from importlib import import_module
    
    # Use a different approach to avoid executing the whole module
    import importlib.util
    
    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(
        "soil_fertility_predictor_functions", 
        os.path.join(parent_dir, "soil_fertility_predictor.py")
    )
    
    # Create the module
    module = importlib.util.module_from_spec(spec)
    
    # Define the functions we want to import
    def predict_fertilizer(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type, model_path=None):
        """Local wrapper for predict_fertilizer"""
        if model_path is None:
            model_path = MODEL_PATH
        
        try:
            import joblib
            import pandas as pd
            import numpy as np
            
            # Load the model
            pipeline = joblib.load(model_path)
            
            # Load feature column names
            feature_cols_path = os.path.join(os.path.dirname(model_path), 'fertilizer_feature_cols.pkl')
            feature_cols = joblib.load(feature_cols_path)
            
            # Create input dataframe with the same structure as training data
            input_data = pd.DataFrame({
                'Temparature': [temperature],
                'Humidity': [humidity],
                'Moisture': [moisture],
                'Nitrogen': [nitrogen],
                'Potassium': [potassium],
                'Phosphorous': [phosphorous]
            })
            
            # Add soil type dummy columns (initialize all to 0)
            for col in feature_cols:
                if col.startswith('Soil_'):
                    input_data[col] = 0
            
            # Set the appropriate soil type column to 1
            soil_col = f'Soil_{soil_type}'
            if soil_col in feature_cols:
                input_data[soil_col] = 1
            
            # Add crop type dummy columns (initialize all to 0)
            for col in feature_cols:
                if col.startswith('Crop_'):
                    input_data[col] = 0
            
            # Set the appropriate crop type column to 1
            crop_col = f'Crop_{crop_type}'
            if crop_col in feature_cols:
                input_data[crop_col] = 1
            
            # Ensure all features are in the right order
            input_data = input_data.reindex(columns=feature_cols, fill_value=0)
            
            # Make prediction
            fertilizer = pipeline.predict(input_data)[0]
            
            # Get prediction probabilities
            probs = dict(zip(pipeline.classes_, pipeline.predict_proba(input_data)[0]))
            
            return fertilizer, probs
            
        except Exception as e:
            raise Exception(f"Fertilizer prediction error: {str(e)}")
    
    def suggest_fertilizer_composition(nitrogen, phosphorous, potassium, crop_type, ideal_df_path=None, fertilizer_df_path=None):
        """Local wrapper for suggest_fertilizer_composition"""
        if ideal_df_path is None:
            ideal_df_path = IDEAL_DF_PATH
        if fertilizer_df_path is None:
            fertilizer_df_path = FERTILIZER_DF_PATH
            
        try:
            import pandas as pd
            import numpy as np
            from scipy.optimize import linprog
            
            # Load ideal values for different crops
            ideal_df = pd.read_csv(ideal_df_path)
            fertilizer_df = pd.read_csv(fertilizer_df_path)
            
            # Look for the crop in the ideal dataframe
            crop_type_lower = crop_type.lower()
            crop_found = False
            
            # Try exact match
            if crop_type_lower in ideal_df['Crop'].str.lower().values:
                ideal_values = ideal_df[ideal_df['Crop'].str.lower() == crop_type_lower].iloc[0]
                crop_found = True
            
            # Try mapping common crops to those in the dataset
            if not crop_found:
                crop_mapping = {
                    'wheat': 'wheat',
                    'barley': 'barley',
                    'maize': 'maize',
                    'rice': 'paddy',
                    'paddy': 'paddy',
                    'sugarcane': 'sugarcane',
                    'cotton': 'cotton',
                    'groundnut': 'ground nuts',
                    'ground nut': 'ground nuts',
                    'ground nuts': 'ground nuts',
                    'pulse': 'pulses',
                    'pulses': 'pulses',
                    'millet': 'millets',
                    'millets': 'millets',
                    'oil seed': 'oil seeds',
                    'oil seeds': 'oil seeds',
                    'tobacco': 'tobacco'
                }
                
                mapped_crop = crop_mapping.get(crop_type_lower)
                if mapped_crop and mapped_crop in ideal_df['Crop'].str.lower().values:
                    ideal_values = ideal_df[ideal_df['Crop'].str.lower() == mapped_crop].iloc[0]
                    crop_found = True
            
            # If still not found, use a default crop
            if not crop_found:
                # Use wheat as a fallback
                ideal_values = ideal_df[ideal_df['Crop'].str.lower() == 'wheat'].iloc[0]
              # Get ideal NPK values for the crop
            ideal_N = ideal_values['N']
            ideal_P = ideal_values['P']
            ideal_K = ideal_values['K']
            
            # Calculate deficiencies
            deficiency_n = max(ideal_N - nitrogen, 0)
            deficiency_p = max(ideal_P - phosphorous, 0)
            deficiency_k = max(ideal_K - potassium, 0)
            
            deficiencies = {
                'Nitrogen': deficiency_n,
                'Phosphorous': deficiency_p,
                'Potassium': deficiency_k,
                'N': deficiency_n,  # Add short form keys for template access
                'P': deficiency_p,
                'K': deficiency_k
            }
              # If no deficiencies, return an empty recommendation
            if deficiency_n <= 0 and deficiency_p <= 0 and deficiency_k <= 0:
                return ["✅ Your soil has adequate levels of NPK for this crop. No additional fertilizers needed."], deficiencies
            
            # Extract NPK content from fertilizers
            N_content = fertilizer_df['N_content'].values
            P_content = fertilizer_df['P_content'].values
            K_content = fertilizer_df['K_content'].values
            
            # Set up linear programming to find optimal fertilizer mix
            c = np.ones(len(fertilizer_df))  # Minimize total amount of fertilizer
            
            A = [
                -N_content,  # Ensure N requirement is met
                -P_content,  # Ensure P requirement is met
                -K_content   # Ensure K requirement is met
            ]
            b = [
                -deficiency_n,
                -deficiency_p,
                -deficiency_k
            ]
            
            # Fertilizer quantities can't be negative
            bounds = [(0, None) for _ in range(len(fertilizer_df))]
            
            # Solve the optimization problem
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            # Prepare recommendations
            recommendations = []
            
            if res.success:
                fertilizer_quantities = res.x
                
                for i, qty in enumerate(fertilizer_quantities):
                    if qty > 0.5:  # Only include significant quantities
                        fertilizer_name = fertilizer_df['Fertilizer'].iloc[i]
                        recommendations.append(f"Apply {qty:.2f} kg/ha of {fertilizer_name}")
                
                if not recommendations:
                    recommendations.append("✅ No fertilizer recommendation available for your specific requirements.")
            else:
                recommendations.append("❌ Could not determine optimal fertilizer mix. Consider consulting a soil specialist.")
            
            return recommendations, deficiencies
            
        except Exception as e:
            raise Exception(f"Fertilizer composition suggestion error: {str(e)}")
    
    def get_npk_analysis(nitrogen, potassium, phosphorous):
        """Analyze NPK levels and provide recommendations"""
        analysis = {}
        
        # Analyze Nitrogen
        if nitrogen < 15:
            analysis['nitrogen'] = {
                'status': 'Low',
                'recommendation': 'Increase nitrogen application. Consider applying nitrogen-rich fertilizers like Urea or Ammonium Sulfate. Legume cover crops can also help fix nitrogen naturally.'
            }
        elif nitrogen > 30:
            analysis['nitrogen'] = {
                'status': 'High',
                'recommendation': 'Reduce nitrogen application. Excessive nitrogen can lead to vegetative growth at the expense of fruiting and can increase susceptibility to diseases.'
            }
        else:
            analysis['nitrogen'] = {
                'status': 'Optimal',
                'recommendation': 'Maintain current nitrogen management practices.'
            }
        
        # Analyze Potassium
        if potassium < 10:
            analysis['potassium'] = {
                'status': 'Low',
                'recommendation': 'Increase potassium application. Consider potassium-rich fertilizers like Potassium Chloride or Potassium Sulfate. Crop residue incorporation can also help recycle potassium.'
            }
        elif potassium > 25:
            analysis['potassium'] = {
                'status': 'High',
                'recommendation': 'Reduce potassium application. Excessive potassium can interfere with the uptake of other nutrients, especially magnesium and calcium.'
            }
        else:
            analysis['potassium'] = {
                'status': 'Optimal',
                'recommendation': 'Maintain current potassium management practices.'
            }
        
        # Analyze Phosphorous
        if phosphorous < 10:
            analysis['phosphorous'] = {
                'status': 'Low',
                'recommendation': 'Increase phosphorous application. Consider fertilizers like Superphosphate or DAP. Organic matter addition and maintaining proper soil pH can also improve phosphorous availability.'
            }
        elif phosphorous > 25:
            analysis['phosphorous'] = {
                'status': 'High',
                'recommendation': 'Reduce application of phosphorous-rich fertilizers. Excessive phosphorous can lead to water pollution and may inhibit micronutrient uptake.'
            }
        else:
            analysis['phosphorous'] = {
                'status': 'Optimal',
                'recommendation': 'Maintain current phosphorous management practices.'
            }
        
        return analysis
            
    print("Successfully defined local versions of prediction functions")
except ImportError as e:
    # If import fails, define the functions here
    print(f"Could not import from soil_fertility_predictor: {str(e)}")
    print("Using local function definitions")
    
    def predict_fertilizer(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type, model_path=None):
        """Local implementation of predict_fertilizer"""
        if model_path is None:
            model_path = MODEL_PATH
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        pipeline = joblib.load(model_path)
        
        # Load feature columns
        try:
            feature_cols = joblib.load(FEATURE_COLS_PATH)
        except:
            # If not available, use a default set of columns
            feature_cols = None
        
        # Create a DataFrame with the input parameters
        input_data = pd.DataFrame({
            'Temparature': [temperature],
            'Humidity': [humidity],
            'Moisture': [moisture],
            'Nitrogen': [nitrogen],
            'Potassium': [potassium],
            'Phosphorous': [phosphorous]
        })
        
        # Handle soil type and crop type
        soil_types = ['Clayey', 'Loamy', 'Black', 'Red', 'Sandy']
        crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 
                    'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
        
        # Add dummy variables for soil type
        for soil in soil_types:
            input_data[f'Soil_{soil}'] = 1 if soil_type == soil else 0
        
        # Add dummy variables for crop type
        for crop in crop_types:
            input_data[f'Crop_{crop}'] = 1 if crop_type == crop else 0
        
        # If we have feature columns list, ensure we have all required columns
        if feature_cols is not None:
            for col in feature_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns to match training data
            input_data = input_data[feature_cols]
        
        # Make prediction
        fertilizer = pipeline.predict(input_data)[0]
        
        # Get probabilities for all fertilizers
        proba = pipeline.predict_proba(input_data)[0]
        fertilizer_classes = pipeline.classes_
        
        # Create a dictionary of fertilizer probabilities
        fertilizer_probs = {fertilizer_classes[i]: proba[i] for i in range(len(fertilizer_classes))}
        fertilizer_probs = {k: v for k, v in sorted(fertilizer_probs.items(), key=lambda item: item[1], reverse=True)}
        
        return fertilizer, fertilizer_probs
    
    def get_npk_analysis(nitrogen, potassium, phosphorous):
        """Local implementation of get_npk_analysis"""
        analysis = {}
        
        # Nitrogen analysis
        if nitrogen < 10:
            analysis['nitrogen'] = {
                'status': 'Low',
                'recommendation': 'Consider increasing the application of nitrogen-rich fertilizers like Urea, Ammonium Sulfate, or Ammonium Nitrate.'
            }
        elif nitrogen > 30:
            analysis['nitrogen'] = {
                'status': 'High',
                'recommendation': 'Reduce application of nitrogen-rich fertilizers. High nitrogen can cause excessive vegetative growth at the expense of flowering and fruiting.'
            }
        else:
            analysis['nitrogen'] = {
                'status': 'Optimal',
                'recommendation': 'Maintain current nitrogen management practices.'
            }
        
        # Potassium analysis
        if potassium < 10:
            analysis['potassium'] = {
                'status': 'Low',
                'recommendation': 'Consider increasing the application of potassium-rich fertilizers like Potassium Chloride (MOP), Potassium Sulfate, or NPK fertilizers with high K.'
            }
        elif potassium > 30:
            analysis['potassium'] = {
                'status': 'High',
                'recommendation': 'Reduce application of potassium-rich fertilizers. Excessive potassium can interfere with the uptake of other nutrients.'
            }
        else:
            analysis['potassium'] = {
                'status': 'Optimal',
                'recommendation': 'Maintain current potassium management practices.'
            }
        
        # Phosphorous analysis
        if phosphorous < 10:
            analysis['phosphorous'] = {
                'status': 'Low',
                'recommendation': 'Consider increasing the application of phosphorous-rich fertilizers like DAP, SSP, or NPK fertilizers with high P.'
            }
        elif phosphorous > 30:
            analysis['phosphorous'] = {
                'status': 'High',
                'recommendation': 'Reduce application of phosphorous-rich fertilizers. Excessive phosphorous can lead to water pollution and may inhibit micronutrient uptake.'
            }
        else:
            analysis['phosphorous'] = {
                'status': 'Optimal',
                'recommendation': 'Maintain current phosphorous management practices.'
            }
        
        return analysis
    
    def suggest_fertilizer_composition(current_n, current_p, current_k, crop, ideal_df_path=None, fertilizer_df_path=None):
        """Local implementation of suggest_fertilizer_composition"""
        if ideal_df_path is None:
            ideal_df_path = IDEAL_DF_PATH
        
        if fertilizer_df_path is None:
            fertilizer_df_path = FERTILIZER_DF_PATH
        
        # Load datasets
        ideal_df = pd.read_csv(ideal_df_path)
        fertilizer_df = pd.read_csv(fertilizer_df_path)
        
        crop = crop.lower()
        
        # Default values for crops not in the dataset
        ideal_n = 60  # Default nitrogen
        ideal_p = 40  # Default phosphorous 
        ideal_k = 40  # Default potassium
        
        # Check if crop exists in ideal data
        crop_found = False
        if crop in ideal_df['Crop'].str.lower().values:
            # Get ideal NPK values for crop
            ideal_values = ideal_df[ideal_df['Crop'].str.lower() == crop].iloc[0]
            ideal_n = ideal_values['N']
            ideal_p = ideal_values['P']
            ideal_k = ideal_values['K']
            crop_found = True
        elif crop in ['wheat', 'paddy', 'barley', 'millets', 'sugarcane', 'tobacco', 'oil seeds', 'pulses', 'ground nuts']:
            # Map similar crops or use defaults for common crops not in the ideal dataset
            if crop == 'wheat' or crop == 'barley' or crop == 'millets':
                # Similar to rice
                ideal_n = 80
                ideal_p = 40
                ideal_k = 40
            elif crop == 'paddy':
                # Rice equivalent
                ideal_n = 80
                ideal_p = 40
                ideal_k = 40
            elif crop == 'sugarcane':
                ideal_n = 100
                ideal_p = 50
                ideal_k = 50
            elif crop == 'oil seeds':
                ideal_n = 40
                ideal_p = 60
                ideal_k = 30
            elif crop == 'pulses':
                # Similar to chickpea
                ideal_n = 40
                ideal_p = 60
                ideal_k = 80
            elif crop == 'ground nuts':
                ideal_n = 30
                ideal_p = 60
                ideal_k = 30
            elif crop == 'tobacco':
                ideal_n = 60
                ideal_p = 30
                ideal_k = 40
            crop_found = True
        else:
            crop_found = False
        
        # Calculate deficiencies        deficiency_n = max(ideal_n - current_n, 0)
        deficiency_p = max(ideal_p - current_p, 0)
        deficiency_k = max(ideal_k - current_k, 0)
        
        deficiencies = {
            'N': deficiency_n,
            'P': deficiency_p,
            'K': deficiency_k,
            'Nitrogen': deficiency_n,  # Add long form keys for consistency
            'Phosphorous': deficiency_p,
            'Potassium': deficiency_k
        }
        
        # If crop not found, include a note in the recommendations
        if not crop_found:
            deficiency_note = [f"Note: Specific data for '{crop}' was not found. Using default values for a general field crop."]
        else:
            deficiency_note = []
        
        # No deficiencies case
        if deficiency_n == 0 and deficiency_p == 0 and deficiency_k == 0:
            return deficiency_note + ["No nutrient deficiencies detected. Your soil has adequate NPK levels for this crop."], deficiencies
            
        # Extract fertilizer composition data
        n_content = fertilizer_df['N_content'].values
        p_content = fertilizer_df['P_content'].values
        k_content = fertilizer_df['K_content'].values
        
        # Optimization setup
        c = np.ones(len(fertilizer_df))  # Minimize total amount of fertilizer
        
        A = [
            -n_content,  # Ensure N requirement is met
            -p_content,  # Ensure P requirement is met
            -k_content   # Ensure K requirement is met
        ]
        
        b = [
            -deficiency_n,
            -deficiency_p,
            -deficiency_k
        ]
        
        bounds = [(0, None) for _ in range(len(fertilizer_df))]
        
        # Solve linear programming problem
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        # Process results
        if res.success:
            fertilizer_quantities = res.x
            recommendations = []
            for i, qty in enumerate(fertilizer_quantities):
                if qty > 0.01:  # Use a small threshold to filter out negligible quantities
                    fertilizer_name = fertilizer_df['Fertilizer'].iloc[i]
                    recommendations.append(f"Apply {qty:.2f} kg/ha of {fertilizer_name}")
            
            if not recommendations:
                recommendations = ["The optimization suggests very small quantities of fertilizers. Your soil may have near-optimal NPK levels."]
        else:
            recommendations = ["No feasible fertilizer combination found to address the deficiencies."]
        
        return deficiency_note + recommendations, deficiencies

def predict_best_fertilizer(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):
    """
    Predict the best fertilizer based on soil and crop parameters
    
    Args:
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        moisture: Soil moisture percentage 
        nitrogen: Nitrogen content in soil (kg/ha)
        potassium: Potassium content in soil (kg/ha)
        phosphorous: Phosphorous content in soil (kg/ha)
        soil_type: Type of soil (Clayey, Loamy, Black, Red, Sandy)
        crop_type: Type of crop being grown
        
    Returns:
        result_dict: Dictionary containing fertilizer prediction and analysis
    """
    try:
        # Predict fertilizer
        fertilizer, probs = predict_fertilizer(
            temperature, humidity, moisture, nitrogen, potassium, phosphorous, 
            soil_type, crop_type, MODEL_PATH
        )
        
        # Get NPK analysis
        npk_analysis = get_npk_analysis(nitrogen, potassium, phosphorous)
        
        # Get fertilizer composition recommendation
        recommendations, deficiencies = suggest_fertilizer_composition(
            nitrogen, phosphorous, potassium, crop_type, 
            IDEAL_DF_PATH, FERTILIZER_DF_PATH
        )
        
        # Prepare result
        result_dict = {
            'success': True,
            'fertilizer': fertilizer,
            'probabilities': {k: float(v) for k, v in probs.items()},  # Convert numpy types to Python native types
            'npk_analysis': npk_analysis,
            'deficiencies': {k: float(v) for k, v in deficiencies.items()},  # Convert numpy types to Python native types
            'recommendations': recommendations
        }
        
        return result_dict
        
    except Exception as e:
        print(f"Error in predict_best_fertilizer: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def get_fertilizer_recommendation(nitrogen, phosphorous, potassium, crop_type):
    """
    Get fertilizer recommendation based on NPK levels and crop type
    This function is designed to be used with the existing fertility route in app.py
    
    Args:
        nitrogen: Nitrogen content in soil (kg/ha)
        phosphorous: Phosphorous content in soil (kg/ha)
        potassium: Potassium content in soil (kg/ha)
        crop_type: Type of crop being grown
        
    Returns:
        recommendations: List of fertilizer recommendations
        result: String with formatted recommendations (for backward compatibility)
    """
    try:
        recommendations, deficiencies = suggest_fertilizer_composition(
            nitrogen, phosphorous, potassium, crop_type, 
            IDEAL_DF_PATH, FERTILIZER_DF_PATH
        )
        
        # Format result as a string (for backward compatibility with existing app.py)
        result = " | ".join(recommendations)
        
        return recommendations, result
        
    except Exception as e:
        print(f"Error in get_fertilizer_recommendation: {str(e)}")
        return [], f"❌ Error calculating fertilizer recommendation: {str(e)}"

# Test the functions if run directly
if __name__ == "__main__":
    # Example parameters
    temperature = 30
    humidity = 60
    moisture = 40
    nitrogen = 20
    potassium = 10
    phosphorous = 15
    soil_type = "Loamy"
    crop_type = "Wheat"
    
    # Test prediction
    result = predict_best_fertilizer(
        temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type
    )
    
    print("\n--- Fertilizer Prediction Test ---")
    if result['success']:
        print(f"Recommended Fertilizer: {result['fertilizer']}")
        print("Top 3 Fertilizer Probabilities:")
        for fert, prob in list(result['probabilities'].items())[:3]:
            print(f"{fert}: {prob:.4f}")
        
        print("\nNutrient Analysis:")
        for nutrient, analysis in result['npk_analysis'].items():
            print(f"- {nutrient.capitalize()}: {analysis['status']}")
            print(f"  {analysis['recommendation']}")
        
        print("\nFertilizer Composition Recommendations:")
        for recommendation in result['recommendations']:
            print(f"- {recommendation}")
    else:
        print(f"Error: {result['error']}")
    
    # Test recommendation function
    print("\n--- Fertilizer Recommendation Test ---")
    recommendations, result_str = get_fertilizer_recommendation(nitrogen, phosphorous, potassium, crop_type)
    print(f"Result string: {result_str}")
    print("Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
