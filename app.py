import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_file
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model and configuration
model = joblib.load('model.joblib')
with open('feature_list.json', 'r') as f:
    features = json.load(f)
with open('threshold.json', 'r') as f:
    threshold_config = json.load(f)
    BEST_THRESHOLD = threshold_config['best_threshold']

def get_risk_category(probability):
    """Convert probability to risk category"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def prepare_input_data(data_dict):
    """Prepare input data for prediction"""
    df = pd.DataFrame([data_dict])
    # Ensure all features are present
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    return df[features]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/individual')
def individual_scoring():
    return render_template('individual.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_df = prepare_input_data(data)
        
        # Get prediction probability
        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability >= BEST_THRESHOLD)
        risk_category = get_risk_category(probability)
        
        return jsonify({
            'probability': float(probability),
            'prediction': prediction,
            'risk_category': risk_category,
            'threshold': BEST_THRESHOLD
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch')
def batch_scoring():
    return render_template('batch.html')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Prepare data
        input_data = df[features]
        
        # Get predictions
        probabilities = model.predict_proba(input_data)[:, 1]
        predictions = (probabilities >= BEST_THRESHOLD).astype(int)
        risk_categories = [get_risk_category(p) for p in probabilities]
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Probability'] = probabilities
        results_df['Prediction'] = predictions
        results_df['Risk_Category'] = risk_categories
        
        # Convert to CSV for download
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        return jsonify({
            'success': True,
            'total_records': len(df),
            'high_risk': sum(1 for r in risk_categories if r == 'High Risk'),
            'medium_risk': sum(1 for r in risk_categories if r == 'Medium Risk'),
            'low_risk': sum(1 for r in risk_categories if r == 'Low Risk'),
            'csv_data': output.getvalue()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/interpretability')
def interpretability():
    return render_template('interpretability.html', features=features)

@app.route('/explain', methods=['POST'])
def explain_prediction():
    try:
        data = request.json
        input_df = prepare_input_data(data)
        
        # Get prediction
        probability = model.predict_proba(input_df)[0][1]
        risk_category = get_risk_category(probability)
        
        # Get feature importances from the model
        # Handle both direct models and pipelines
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'named_steps'):
            # It's a pipeline, get the final estimator
            final_estimator = list(model.named_steps.values())[-1]
            if hasattr(final_estimator, 'feature_importances_'):
                feature_importance = final_estimator.feature_importances_
            else:
                # Fallback: create dummy importances
                feature_importance = np.ones(len(features)) / len(features)
        else:
            # Fallback: create dummy importances
            feature_importance = np.ones(len(features)) / len(features)
        
        # Create feature importance dictionary
        importance_dict = dict(zip(features, feature_importance))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 10 most important features and convert to Python floats
        top_features = [(f[0], float(f[1])) for f in sorted_importance[:10]]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        feature_names = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        plt.barh(range(len(feature_names)), importances)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features')
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Generate recommendations
        recommendations = generate_recommendations(data, risk_category, top_features)
        
        return jsonify({
            'probability': float(probability),
            'risk_category': risk_category,
            'top_features': top_features,
            'feature_plot': img_str,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generate_recommendations(data, risk_category, top_features):
    """Generate recommendations based on risk category and feature values"""
    recommendations = []
    
    if risk_category == "High Risk":
        recommendations.append("⚠️ High risk detected. Consider additional verification and monitoring.")
        
        # Check specific features and provide targeted advice
        if data.get('CurrentArrears', 0) > 0:
            recommendations.append("• Clear current arrears to improve credit standing")
        
        if data.get('MaxArrears', 0) > 5:
            recommendations.append("• Work on reducing maximum arrears history")
        
        if data.get('NetIncome', 0) < 50000:
            recommendations.append("• Consider ways to increase net income")
        
        if data.get('Age', 0) < 25:
            recommendations.append("• Young age may be a factor - consider building credit history")
        
    elif risk_category == "Medium Risk":
        recommendations.append("⚡ Medium risk. Monitor closely and consider risk mitigation measures.")
        
        if data.get('SavingAcctDepositCount', 0) < 10:
            recommendations.append("• Increase savings account activity to demonstrate financial stability")
        
        if data.get('CustomerTenure', 0) < 2:
            recommendations.append("• Build longer relationship history with the institution")
        
    else:
        recommendations.append("✅ Low risk profile. Good candidate for approval.")
        recommendations.append("• Maintain current financial behavior")
        recommendations.append("• Continue regular savings and loan payments")
    
    # Add feature-specific recommendations based on top important features
    for feature, importance in top_features[:3]:
        if feature in data:
            value = data[feature]
            if feature == 'NetIncome' and value < 100000:
                recommendations.append(f"• Consider increasing {feature} (current: {value:,.0f})")
            elif feature == 'Age' and value < 30:
                recommendations.append(f"• Age factor: {value} years - building credit history over time will help")
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)