# Credit Risk Scoring System 🏦

A comprehensive Flask web application for credit risk assessment using XGBoost machine learning model. This system provides three main functionalities: individual scoring, batch processing, and model interpretability.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.1.2-green.svg)
![XGBoost](https://img.shields.io/badge/xgboost-v3.1.3-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### 1. Individual Scoring
- **Real-time Assessment**: Score individual loan applications instantly
- **Interactive Form**: User-friendly interface for entering customer details
- **Risk Categorization**: Automatic classification into Low, Medium, or High risk
- **Probability Scores**: Detailed probability scores with threshold information

### 2. Batch Processing
- **Bulk Upload**: Process hundreds of applications via CSV upload
- **Visual Analytics**: Risk distribution charts and summaries
- **Downloadable Results**: Export results with risk scores and categories
- **Progress Tracking**: Real-time processing status updates

### 3. Model Interpretability
- **Feature Importance**: Understand which factors drive the risk score
- **Visual Explanations**: Interactive charts showing feature contributions
- **Personalized Recommendations**: Actionable insights for risk improvement
- **Transparency**: Clear explanations of model decisions

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: XGBoost, scikit-learn
- **Frontend**: Bootstrap 5, HTML5, JavaScript
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/J-MUGAMBI/scoring_project.git
   cd scoring_project
   ```

2. **Create virtual environment**
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

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
scoring_project/
├── app.py                      # Main Flask application
├── model.joblib               # Trained XGBoost model
├── feature_list.json          # Required features configuration
├── threshold.json             # Model threshold settings
├── requirements.txt           # Python dependencies
├── templates/                 # HTML templates
│   ├── index.html            # Home page
│   ├── individual.html       # Individual scoring interface
│   ├── batch.html           # Batch processing interface
│   └── interpretability.html # Model explanation interface
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## 🎯 Usage Guide

### Individual Scoring
1. Navigate to the **Individual Scoring** page
2. Fill in customer details (age, income, loan amount, etc.)
3. Click **"Calculate Risk Score"**
4. View risk assessment with probability and category

### Batch Processing
1. Go to **Batch Scoring** page
2. Prepare CSV file with required columns (see template below)
3. Upload file and click **"Process Batch"**
4. Download results with risk scores

### Model Interpretability
1. Access **Interpretability** page
2. Enter customer information
3. Click **"Analyze & Explain"**
4. Review feature importance and recommendations

## 📊 Required CSV Format

For batch processing, your CSV must include these columns:

| Column Name | Type | Description |
|-------------|------|-------------|
| CRR_NARRATION | String | Credit risk rating |
| Age | Numeric | Customer age |
| NetIncome | Numeric | Monthly net income |
| Exposure_Amount | Numeric | Loan amount requested |
| CurrentArrears | Numeric | Current arrears amount |
| MaxArrears | Numeric | Maximum historical arrears |
| Gender | String | MALE/FEMALE |
| SECTOR | String | Industry sector |
| EMPLOYEMENT_STATUS | String | Employment status |
| ... | ... | (See full list in app) |

## 🔒 Model Information

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Optimization**: F1 score maximization with Recall ≥ 0.75
- **Threshold**: 0.66 (configurable)
- **Risk Categories**:
  - 🟢 **Low Risk**: Probability < 0.3
  - 🟡 **Medium Risk**: 0.3 ≤ Probability < 0.6
  - 🔴 **High Risk**: Probability ≥ 0.6

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Using Docker (create Dockerfile)
docker build -t credit-risk-scorer .
docker run -p 5000:5000 credit-risk-scorer
```

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/individual` | GET | Individual scoring form |
| `/predict` | POST | Individual prediction API |
| `/batch` | GET | Batch processing form |
| `/batch_predict` | POST | Batch prediction API |
| `/interpretability` | GET | Model explanation form |
| `/explain` | POST | Model explanation API |

## 🛡️ Security Features

- File upload size limits (16MB)
- Input validation and sanitization
- Secure file handling
- Error handling and logging
- CSRF protection ready

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**J-MUGAMBI**
- GitHub: [@J-MUGAMBI](https://github.com/J-MUGAMBI)
- Project: [scoring_project](https://github.com/J-MUGAMBI/scoring_project)

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting framework
- Flask community for the lightweight web framework
- Bootstrap team for the responsive UI components

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

⭐ **Star this repository if you find it helpful!**