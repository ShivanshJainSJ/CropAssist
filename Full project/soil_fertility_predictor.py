import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define the project directory path
project_dir = 'c:\\Users\\Shivansh\\Desktop\\New folder\\Full project\\project'

# Load the dataset
df = pd.read_csv(os.path.join(project_dir, 'Fertilizer.csv'))

# Data preprocessing
# Check for missing values
print(f"Missing values in the dataset:\n{df.isnull().sum()}")

# Remove any rows with missing values
df = df.dropna()

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Unique fertilizer classes: {df['Fertilizer Name'].unique()}")
print(f"Class distribution:\n{df['Fertilizer Name'].value_counts()}")

# Extract features and target
X = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer Name']

# Visualize correlation between features
plt.figure(figsize=(10, 8))
correlation = X.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(project_dir, 'static', 'images', 'feature_correlation.png'))
plt.close()

# Convert soil type and crop type to dummy variables
soil_dummies = pd.get_dummies(df['Soil Type'], prefix='Soil')
crop_dummies = pd.get_dummies(df['Crop Type'], prefix='Crop')

# Combine features
X = pd.concat([X, soil_dummies, crop_dummies], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for grid search
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Perform grid search
print("Performing hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the model on test data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.named_steps['model'].feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(project_dir, 'static', 'images', 'feature_importance.png'))
plt.close()

# Create confusion matrix
plt.figure(figsize=(14, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(project_dir, 'static', 'images', 'confusion_matrix.png'))
plt.close()

# Save the model and pipeline
print("Saving model and pipeline...")
model_path = os.path.join(project_dir, 'fertilizer_model.pkl')
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# Save feature column names
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, os.path.join(project_dir, 'fertilizer_feature_cols.pkl'))

# Function to predict fertilizer based on soil parameters
def predict_fertilizer(temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type, model_path=None):
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
        model_path: Path to the saved model (if None, uses the default path)
        
    Returns:
        fertilizer: The recommended fertilizer
        fertilizer_probs: Dictionary of fertilizer probabilities
    """
    if model_path is None:
        model_path = os.path.join(project_dir, 'fertilizer_model.pkl')
    
    # Load the model
    pipeline = joblib.load(model_path)
    
    # Load feature columns if available
    try:
        feature_cols = joblib.load(os.path.join(project_dir, 'fertilizer_feature_cols.pkl'))
    except:
        # If not available, we'll try to reconstruct based on the input data
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

# Function to suggest fertilizer composition based on NPK deficiencies
def suggest_fertilizer_composition(current_n, current_p, current_k, crop, ideal_df_path=None, fertilizer_df_path=None):
    """
    Suggests fertilizer composition based on current NPK levels and crop requirements
    
    Args:
        current_n: Current Nitrogen level in soil
        current_p: Current Phosphorous level in soil
        current_k: Current Potassium level in soil
        crop: Crop type
        ideal_df_path: Path to ideal NPK values dataset
        fertilizer_df_path: Path to fertilizer composition dataset
        
    Returns:
        recommendations: List of fertilizer recommendations
        deficiencies: Dictionary with NPK deficiencies
    """
    if ideal_df_path is None:
        ideal_df_path = os.path.join(project_dir, 'FertilizerData1.csv')
    
    if fertilizer_df_path is None:
        fertilizer_df_path = os.path.join(project_dir, 'fertilizer_composition.csv')
    
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
    
    # Calculate deficiencies
    deficiency_n = max(ideal_n - current_n, 0)
    deficiency_p = max(ideal_p - current_p, 0)
    deficiency_k = max(ideal_k - current_k, 0)
    
    deficiencies = {
        'N': deficiency_n,
        'P': deficiency_p,
        'K': deficiency_k
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
    from scipy.optimize import linprog
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
    
    return recommendations, deficiencies

# Provide detailed fertilizer recommendations based on NPK analysis
def get_npk_analysis(nitrogen, potassium, phosphorous):
    """
    Generate qualitative analysis of NPK levels
    
    Args:
        nitrogen: Nitrogen content in soil
        potassium: Potassium content in soil
        phosphorous: Phosphorous content in soil
        
    Returns:
        analysis: Dictionary with analysis for each nutrient
    """
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

# Example usage
if __name__ == "__main__":
    print("\n--- Example Fertilizer Prediction ---")
    # Example soil and crop parameters
    temperature = 30
    humidity = 60
    moisture = 40
    nitrogen = 20
    potassium = 10
    phosphorous = 15
    soil_type = "Loamy"
    crop_type = "Wheat"
    
    # Use model for prediction
    try:
        fertilizer, probs = predict_fertilizer(
            temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type
        )
        
        print(f"Recommended Fertilizer: {fertilizer}")
        print("Fertilizer Probabilities:")
        for fert, prob in probs.items():
            print(f"{fert}: {prob:.4f}")
    except Exception as e:
        print(f"Error in fertilizer prediction: {str(e)}")
    
    # Get NPK recommendations
    print("\n--- NPK Analysis and Recommendations ---")
    npk_analysis = get_npk_analysis(nitrogen, potassium, phosphorous)
    
    for nutrient, analysis in npk_analysis.items():
        print(f"- {nutrient.capitalize()}: {analysis['status']}")
        print(f"  {analysis['recommendation']}")
    
    # Get optimal fertilizer composition
    print("\n--- Fertilizer Composition Recommendation ---")
    try:
        recommendations, deficiencies = suggest_fertilizer_composition(nitrogen, phosphorous, potassium, crop_type)
        
        if deficiencies:
            print("Nutrient deficiencies detected:")
            for nutrient, amount in deficiencies.items():
                if amount > 0:
                    print(f"- {nutrient}: {amount:.2f} kg/ha required")
        
        print("\nRecommended fertilizer application:")
        for recommendation in recommendations:
            print(f"- {recommendation}")
    except Exception as e:
        print(f"Error in fertilizer composition suggestion: {str(e)}")