Student Performance Prediction 🎓

This project predicts students' math scores based on their demographics, parental education level, lunch type, and test preparation status.
The goal is to demonstrate end-to-end ML engineering skills — from data preprocessing to model deployment with CI/CD.

🚀 Features

Data Preprocessing: Handling missing values, scaling numeric features, and encoding categorical features.

Model Training: Trained with multiple ML algorithms and selected the best-performing model.

Prediction Pipeline: Clean user input → preprocess → predict using saved model.

Flask API: Simple web interface for real-time predictions.

Error Handling: Custom exception logging for debugging.

CI/CD Ready: Configured for smooth deployment using GitHub Actions.

🛠️ Tech Stack

Programming: Python 3.10+

ML Frameworks: scikit-learn, pandas, numpy

Web Framework: Flask

Logging & Exception Handling: Custom modules (src.logger, src.exception)

CI/CD: GitHub Actions for automated testing, linting, and deployment

Deployment Options: Heroku / AWS / Docker

📂 Project Structure
mlproject/
│
├── app.py                        # Flask app entrypoint
├── artifacts/                    # Stores trained model & preprocessor
├── src/
│   ├── pipeline/                  # Training & prediction pipelines
│   ├── components/                # Data ingestion, transformation, model trainer
│   ├── exception.py               # Custom exception handler
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions (save/load objects)
│
├── templates/                     # HTML files for Flask frontend
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
└── .github/workflows/ci.yml       # CI/CD pipeline

⚙️ Workflow

Data Ingestion

Load dataset

Train/test split

Data Transformation

Impute missing values

Scale numeric features

Encode categorical variables (OneHotEncoder)

Model Training

Train multiple models (e.g., Linear Regression, RandomForest, XGBoost)

Evaluate with RMSE/MAE/R²

Save best model to artifacts/model.pkl

Prediction Pipeline

Load saved preprocessor.pkl and model.pkl

Replace "Unknown" with NaN

Transform features

Generate predictions

Deployment with Flask

API endpoint for predictions

HTML form for user input

🌐 CI/CD Pipeline

CI/CD is set up using GitHub Actions. The workflow (.github/workflows/ci.yml) runs automatically on every push/PR:

Install Dependencies

- name: Install dependencies
  run: pip install -r requirements.txt


Code Linting (flake8)

- name: Lint with flake8
  run: flake8 .


Run Unit Tests (pytest)

- name: Run tests
  run: pytest


Build & Deploy (Optional)

Can be extended to deploy on Heroku, AWS EC2, or with Docker + GitHub Packages.

This ensures your project is always tested, clean, and production-ready.

🖥️ How to Run Locally
# Clone repo
git clone https://github.com/yourusername/mlproject.git
cd mlproject

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py


Open your browser → http://127.0.0.1:5000/

📊 Example Input & Output

Input:

Gender: Female

Race/Ethnicity: Group C

Parental Education: Bachelor's degree

Lunch: Standard

Test Preparation: Completed

Reading Score: 72

Writing Score: 74

Output:
👉 Predicted Math Score: 69.4

🎯 Key Takeaways for Recruiters

Designed a production-ready ML pipeline with modular code.

Implemented robust preprocessing to handle unknown categories and missing values.

Built end-to-end system: Data → Model → API → Deployment → CI/CD.

Demonstrated software engineering practices: logging, exception handling, unit testing.

Ready for real-world ML engineering / Data Scientist workflows.

🔥 This project proves I can build, deploy, and maintain ML applications at scale.
