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

def decision_from_pd(pd):
    """DECISION POLICY — ALIGNED TO EXCEL (OPTION A)"""
    if pd > 0.10:
        return {
            "Risk_Class": "HIGH RISK",
            "Decision": "DECLINE"
        }
    elif pd > 0.03:
        return {
            "Risk_Class": "MEDIUM RISK",
            "Decision": "REFER / APPROVE WITH CONTROLS"
        }
    else:
        return {
            "Risk_Class": "LOW RISK",
            "Decision": "AUTO APPROVE"
        }

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
        decision_info = decision_from_pd(probability)
        
        return jsonify({
            'probability': float(probability),
            'prediction': prediction,
            'risk_category': decision_info['Risk_Class'],
            'decision': decision_info['Decision'],
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
        decisions = [decision_from_pd(p) for p in probabilities]
        risk_categories = [d['Risk_Class'] for d in decisions]
        decision_labels = [d['Decision'] for d in decisions]
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Probability'] = probabilities
        results_df['Prediction'] = predictions
        results_df['Risk_Category'] = risk_categories
        results_df['Decision'] = decision_labels
        
        # Convert to CSV for download
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        return jsonify({
            'success': True,
            'total_records': len(df),
            'high_risk': sum(1 for r in risk_categories if r == 'HIGH RISK'),
            'medium_risk': sum(1 for r in risk_categories if r == 'MEDIUM RISK'),
            'low_risk': sum(1 for r in risk_categories if r == 'LOW RISK'),
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
        decision_info = decision_from_pd(probability)
        risk_category = decision_info['Risk_Class']
        
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
    """Generate intelligent recommendations based on risk category and top feature importance"""
    recommendations = []
    
    # Extract top feature names for analysis
    top_feature_names = [f[0] for f in top_features[:5]]
    
    if risk_category == "LOW RISK":
        recommendations.append({
            'type': 'success',
            'title': 'Excellent Credit Profile - AUTO APPROVE',
            'message': 'Congratulations! Your credit profile demonstrates strong financial responsibility with minimal default risk. Continue these positive behaviors to maintain your excellent standing.'
        })
        
        # Analyze top features and provide specific guidance
        for feature, importance in top_features[:4]:
            if feature in data:
                value = data[feature]
                
                if feature == 'SECTOR':
                    recommendations.append({
                        'type': 'maintain',
                        'title': 'Industry Sector Stability',
                        'message': f'Continue building expertise and stability in your industry to maintain favorable credit terms.'
                    })
                
                elif feature == 'Age':
                    if value >= 30:
                        recommendations.append({
                            'type': 'maintain',
                            'title': 'Mature Credit Profile',
                            'message': f'At {int(value)} years, your age reflects financial maturity. This demographic stability positively impacts your creditworthiness.'
                        })
                    else:
                        recommendations.append({
                            'type': 'maintain',
                            'title': 'Building Credit History',
                            'message': f'At {int(value)} years, continue building your credit history. Your current profile is strong - maintain this trajectory as you mature financially.'
                        })
                
                elif feature == 'Exposure_Amount':
                    if value <= 300000:
                        recommendations.append({
                            'type': 'maintain',
                            'title': 'Conservative Loan Request',
                            'message': 'Your loan request demonstrates responsible borrowing aligned with your financial capacity. This conservative approach strengthens your application.'
                        })
                    else:
                        recommendations.append({
                            'type': 'maintain',
                            'title': 'Loan Amount Assessment',
                            'message': 'Your loan request has been evaluated against your financial profile. Ensure the amount aligns with your repayment capacity for optimal approval.'
                        })
                
                elif feature == 'Max_REMAINING_TENOR(AllLoans)':
                    if value <= 24:
                        recommendations.append({
                            'type': 'maintain',
                            'title': 'Optimal Loan Tenure',
                            'message': 'Your loan tenure strategy shows good debt management. Shorter repayment periods reduce long-term risk and demonstrate strong commitment to lenders.'
                        })
                    else:
                        recommendations.append({
                            'type': 'tip',
                            'title': 'Loan Tenure Optimization',
                            'message': 'Consider shorter loan tenures when possible to reduce interest costs and improve future credit applications. This demonstrates stronger financial discipline.'
                        })
                
                elif feature == 'NetIncome':
                    recommendations.append({
                        'type': 'maintain',
                        'title': 'Strong Income Base',
                        'message': f'Your net income of KES {value:,.0f} provides solid financial foundation. Continue diversifying income sources for even greater stability.'
                    })
                
                elif feature == 'CurrentArrears' and value == 0:
                    recommendations.append({
                        'type': 'maintain',
                        'title': 'Perfect Payment Record',
                        'message': 'Zero arrears is excellent! Continue making all payments on time to preserve your pristine credit history.'
                    })
        
        recommendations.append({
            'type': 'tip',
            'title': 'Maintain Your Excellence',
            'message': 'Keep up your timely payments, maintain low debt-to-income ratio, and continue building savings. Consider applying for higher credit limits as your profile strengthens further.'
        })
    
    elif risk_category == "MEDIUM RISK":
        recommendations.append({
            'type': 'warning',
            'title': 'Moderate Risk - REFER / APPROVE WITH CONTROLS',
            'message': 'Your profile shows moderate risk. Implement these targeted improvements based on the most influential factors to move to low-risk category and unlock better terms.'
        })
        
        # Analyze top features for improvement areas
        for feature, importance in top_features[:5]:
            if feature in data:
                value = data[feature]
                
                if feature == 'CurrentArrears' and value > 0:
                    recommendations.append({
                        'type': 'critical',
                        'title': 'Clear Current Arrears Immediately',
                        'message': f'You have KES {value:,.0f} in arrears. This is the #1 priority - clear all outstanding arrears within 30 days to significantly improve your credit score.'
                    })
                
                elif feature == 'MaxArrears' and value > 3:
                    recommendations.append({
                        'type': 'improve',
                        'title': 'Reduce Arrears History',
                        'message': f'Your maximum arrears of {int(value)} months is concerning. Focus on 6+ months of consistent, on-time payments to rebuild trust with lenders.'
                    })
                
                elif feature == 'NetIncome' and value < 80000:
                    recommendations.append({
                        'type': 'improve',
                        'title': 'Increase Income Streams',
                        'message': f'Current income: KES {value:,.0f}. Target: KES 80,000+. Consider side income, salary negotiation, or skill development to boost earning capacity.'
                    })
                
                elif feature == 'Exposure_Amount':
                    recommendations.append({
                        'type': 'improve',
                        'title': 'Optimize Loan Amount',
                        'message': f'Requested amount: KES {value:,.0f}. Consider reducing by 20-30% to improve approval odds and demonstrate conservative borrowing while you strengthen your profile.'
                    })
                
                elif feature == 'RUNNING_LOANS_COUNT' and value > 2:
                    recommendations.append({
                        'type': 'improve',
                        'title': 'Reduce Active Loans',
                        'message': f'You have {int(value)} active loans. Pay off at least one loan before applying for new credit to reduce debt burden and improve approval chances.'
                    })
                
                elif feature == 'SavingAcctDepositCount' and value < 15:
                    recommendations.append({
                        'type': 'improve',
                        'title': 'Increase Savings Activity',
                        'message': f'Current deposits: {int(value)}. Target: 15+ monthly deposits. Regular savings demonstrate financial discipline and improve creditworthiness.'
                    })
                
                elif feature == 'CustomerTenure' and value < 3:
                    recommendations.append({
                        'type': 'improve',
                        'title': 'Build Banking Relationship',
                        'message': f'Tenure: {value:.1f} years. Target: 3+ years. Maintain active relationship with your bank through regular transactions and savings to build trust.'
                    })
        
        recommendations.append({
            'type': 'tip',
            'title': '90-Day Action Plan',
            'message': 'Focus on: (1) Clear all arrears, (2) Make 3 months of on-time payments, (3) Increase savings deposits to 15+, (4) Reduce loan amount if possible. Reapply after 90 days for better terms.'
        })
    
    else:  # HIGH RISK
        recommendations.append({
            'type': 'danger',
            'title': 'High Risk Profile - DECLINE',
            'message': 'Your application shows high default risk and requires immediate corrective action. Follow this recovery plan based on the critical factors affecting your score.'
        })
        
        # Critical actions based on top features
        for feature, importance in top_features[:5]:
            if feature in data:
                value = data[feature]
                
                if feature == 'CurrentArrears' and value > 0:
                    recommendations.append({
                        'type': 'critical',
                        'title': 'URGENT: Clear All Arrears',
                        'message': f'Outstanding arrears: KES {value:,.0f}. This is critically damaging your credit. Negotiate payment plan with lender and clear within 60 days - this is non-negotiable.'
                    })
                
                elif feature == 'MaxArrears' and value > 5:
                    recommendations.append({
                        'type': 'critical',
                        'title': 'URGENT: Rebuild Payment History',
                        'message': f'Maximum arrears of {int(value)} months indicates severe payment issues. You need 12+ months of perfect payment history before reapplying. Set up auto-payments immediately.'
                    })
                
                elif feature == 'NetIncome' and value < 50000:
                    recommendations.append({
                        'type': 'critical',
                        'title': 'URGENT: Increase Income',
                        'message': f'Income of KES {value:,.0f} is insufficient for requested loan. Delay application until income reaches KES 80,000+ through additional work or business opportunities.'
                    })
                
                elif feature == 'RUNNING_LOANS_COUNT' and value > 3:
                    recommendations.append({
                        'type': 'critical',
                        'title': 'URGENT: Reduce Debt Burden',
                        'message': f'{int(value)} active loans is excessive. Pay off at least 2 loans completely before considering new credit. Focus on debt consolidation if possible.'
                    })
                
                elif feature == 'NonPerforming' and value > 0:
                    recommendations.append({
                        'type': 'critical',
                        'title': 'URGENT: Resolve Non-Performing Loans',
                        'message': f'You have {int(value)} non-performing loan(s). This severely damages creditworthiness. Resolve immediately through restructuring or settlement.'
                    })
                
                elif feature == 'Exposure_Amount':
                    recommendations.append({
                        'type': 'critical',
                        'title': 'Loan Amount Too High',
                        'message': f'Requested KES {value:,.0f} exceeds your capacity. Reduce by 50%+ or delay application for 6-12 months while improving financial position.'
                    })
        
        recommendations.append({
            'type': 'tip',
            'title': '6-Month Recovery Plan',
            'message': 'Do NOT apply for new credit now. Instead: (1) Clear all arrears within 60 days, (2) Reduce active loans to maximum 2, (3) Build 6 months of perfect payment history, (4) Increase income to 80K+, (5) Save 10% of income monthly. Reapply after 6 months.'
        })
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)