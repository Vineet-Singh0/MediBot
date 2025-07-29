# MediPredict: AI-Powered Medical Disease Prediction

MediPredict is an advanced, user-friendly platform for medical disease prediction and health advice. It combines machine learning, interactive web technology, and a rule-based chatbot to help users understand their symptoms and receive actionable recommendations. The system is designed for both healthcare professionals and the general public, making medical insights accessible, transparent, and easy to use.

## What Does MediPredict Do?
- **Predicts likely diseases** based on user-reported symptoms and their severity.
- **Suggests precautions** and provides mapped doctor recommendations for each predicted disease.
- **Offers fallback advice** to visit a general physician if no specific doctor is mapped.
- **Allows users to download a PDF summary** of their results for sharing or future reference.
- **Includes a medical chatbot** for rule-based health queries and information.
- **Supports model training and evaluation** via interactive Jupyter notebooks, enabling experimentation with multiple machine learning algorithms.

## Tech Stack
- **Python 3.8+** – Core programming language
- **Streamlit** – Interactive web app for disease prediction
- **Pandas, NumPy, scikit-learn, XGBoost** – Data processing and machine learning
- **FPDF** – PDF generation for downloadable results
- **Jupyter Notebooks** – Data exploration, model training, and evaluation
- **Azure App Service** – Cloud deployment for scalable, secure access
- **Other libraries:** PIL, joblib, difflib, etc.

## Project Structure
- `data/` – Contains raw and processed medical datasets in CSV format.
- `notebooks/` – Jupyter notebooks for data exploration, model training, and evaluation. Includes workflows for KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, Logistic Regression, and hyperparameter tuning.
- `src/` – Source code for the Streamlit app, chatbot, and model training scripts.
- `models/` – Pre-trained models and encoders used for predictions.
- `features/` – Feature files for model input.
- `outputs/` – Generated outputs such as plots, reports, and prediction results.

## Key Features
- **Modern Streamlit App:** Glassmorphism UI, stepper navigation, sidebar branding, health tips, and privacy notice.
- **Symptom Input:** Autocomplete, cluster-based suggestions, and per-symptom severity selection.
- **Results Display:** Top 3 disease predictions with confidence scores, mapped precautions, and doctor recommendations.
- **PDF Download:** Users can download their prediction results as a PDF summary.
- **Medical Chatbot:** Rule-based system for answering health-related queries.
- **Robust Mapping:** Ensures correct display of precautions and doctor advice, with fallback to general physician.
- **Notebook Workflows:** Train, evaluate, and tune models interactively.
- **Azure Deployment:** The app is deployed on Azure for scalable, secure, and real-time access.

## How to Use MediPredict
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Explore data and train models:**
   - Open notebooks in the `notebooks/` directory for EDA and model experimentation.
3. **Run the Streamlit app:**
   ```bash
   streamlit run src/app.py
   ```
   - Enter your symptoms, specify severity, and receive predictions and advice.
   - Download your results as a PDF for sharing or record-keeping.
4. **Use the chatbot:**
   - Run the chatbot script in `src/` for rule-based medical queries.

## Data Exploration, Cleaning, and Model Training

- **Exploratory Data Analysis (EDA):**
  - Use the Jupyter notebooks in the `notebooks/` directory to explore the medical datasets, visualize distributions, and understand relationships between symptoms and diseases.
  - EDA helps identify missing values, outliers, and feature importance, guiding further data processing and modeling.

- **Data Cleaning:**
  - The project includes scripts and notebook cells for handling missing data, correcting inconsistent entries, encoding categorical variables, and normalizing features.
  - Cleaned data is saved in the `outputs/` directory for use in model training.

- **Model Training:**
  - Multiple machine learning models are supported, including KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, and Logistic Regression.
  - Notebooks provide code for training, evaluating, and tuning models using cross-validation and hyperparameter search (e.g., RandomizedSearchCV).
  - Trained models are saved in the `models/` directory and used by the Streamlit app and chatbot for predictions.

## Deployment
MediPredict is deployed on Azure, making it accessible from anywhere with secure, scalable infrastructure. The deployment process involves:

- **Containerization:** The Streamlit app and backend are packaged into a Docker container for portability and consistency.
- **Azure App Service:** The container is deployed to Azure App Service, which provides managed hosting, scaling, and HTTPS security.
- **Continuous Integration:** Updates to the codebase can be automatically built and deployed using Azure Pipelines or GitHub Actions.
- **Environment Variables:** Sensitive information and configuration are managed via Azure portal settings.
- **Access:** Users can interact with the app in real time via the provided Azure URL, benefiting from cloud-based predictions and advice.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

---

*For more details, explore the notebooks, try the app, or reach out for collaboration!*
