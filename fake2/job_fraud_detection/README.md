# Fake Job Post Detection (Random Forest + NLP Explainability)

A production-ready **Fake Job Post Detection** portfolio project. Users paste/upload a job posting; the app predicts whether it is **Fraudulent** or **Not Fraudulent** using an NLP pipeline with a **Random Forest Classifier**.

It also:
- Shows a **confidence score**
- Highlights **suspicious keywords/phrases** (rules-based)
- Extracts key **features** (company name, location, salary, requirements, and a description preview)
- Provides a best-effort **model top-feature explanation**
- Generates training visualizations (fraud distribution, confusion matrix, top TF-IDF features)

---

## Project Structure

- `src/` — training, inference, and Flask deployment code
- `dataset/` — dataset CSV (`job_postings.csv`)
- `trained_models/` — saved model artifact (`random_forest_job_fraud.joblib`)
- `screenshots/` — training plots/images
- `templates/` — Flask HTML templates

---

## Dataset Format

`dataset/job_postings.csv` must contain:
- `job_description` (text)
- `location` (string)
- `fraudulent` (0/1 integer label)

---

## How It Works

### Training (`src/train.py`)
1. TF-IDF vectorization of `job_description`
2. Location encoding with `LabelEncoder`
3. Combine features and train `RandomForestClassifier`
4. Evaluate with accuracy/precision/recall/F1/ROC-AUC
5. Save:
   - Model + artifacts to `trained_models/random_forest_job_fraud.joblib`
   - Plots to `screenshots/`

### Inference (`src/inference.py`)
1. Load model artifacts
2. Transform job text to TF-IDF features
3. Encode location (fallback suggestions if unseen)
4. Predict class probability and confidence
5. Apply rules-based fraud keyword detection and return:
   - suspicious contributions
   - extracted features
   - best-effort model top features

---

## Run the App

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train the model (once)
```bash
python -m src.train
```

### 3) Start Flask
```bash
python app.py
```

Open:
- `http://localhost:5000`

---

## Notes
- If you don’t provide a dataset, inference will fail because the trained model artifact is required.
- For explainability, Random Forest is not coefficient-based; the project uses:
  - **rules-based suspicious phrase matching**
  - **top TF-IDF features approximation** using TF-IDF + RandomForest feature importances

---

## Screenshots

After training, the following files are generated in `screenshots/`:
- `fraud_distribution.png`
- `top_tfidf_features.png`
- `confusion_matrix.png`

