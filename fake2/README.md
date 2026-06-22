# Job Fraud Detection System

A Flask web application that detects potentially fraudulent job postings using machine learning.

## Features
- User authentication (registration/login)
- Job posting analysis
- Fraud prediction with confidence score
- Common location suggestions

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Generate the dataset (optional):
   ```bash
   python generate_job_dataset.py
   ```

2. Train the model:
   ```bash
   python job_fraud_detection/model.py
   ```

3. Run the application:
   ```bash
   python job_fraud_detection/app.py
   ```

4. Access the web interface at `http://localhost:5000`

## Data Generation

The `generate_job_dataset.py` script creates synthetic job postings with:
- 80% legitimate postings
- 20% fraudulent postings
- Common Indian locations
- Realistic job descriptions

## Model Training

The Random Forest classifier is trained on:
- Job description text (TF-IDF features)
- Location information
- Balanced class weights

## License

MIT
