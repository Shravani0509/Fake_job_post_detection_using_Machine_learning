Fake Job Fraud Detection System

The Fake Job Fraud Detection System is a machine learning-powered web application developed to identify potentially fraudulent job postings and protect job seekers from recruitment scams. The application allows users to submit job details and receive an instant prediction indicating whether a job posting is likely legitimate or fraudulent.

Key Features:

• User Authentication:
Implemented secure user registration and login functionality to provide personalized access and protect user data.

• Job Posting Analysis:
Users can enter job descriptions and location information, which are analyzed by the machine learning model to detect suspicious patterns commonly found in fraudulent job advertisements.

• Fraud Prediction:
The system predicts whether a job posting is legitimate or fraudulent using a trained Random Forest classification model.

• Confidence Score:
Displays the prediction confidence percentage, helping users understand the reliability of the model's decision.

• Location-Based Analysis:
Incorporates location information as an additional feature to improve fraud detection accuracy and identify suspicious geographic patterns.

• Synthetic Dataset Generation:
Developed a custom dataset generation script to create realistic job postings containing both legitimate and fraudulent examples for training and testing purposes.

Dataset Characteristics:

• 80% legitimate job postings
• 20% fraudulent job postings
• Common Indian city locations
• Realistic job titles and descriptions
• Balanced data distribution for model training

Machine Learning Pipeline:

1. Data Collection and Generation

   * Generated synthetic job posting data.
   * Created labels for legitimate and fraudulent postings.

2. Data Preprocessing

   * Cleaned textual job descriptions.
   * Handled missing values.
   * Processed location information.

3. Feature Engineering

   * Applied TF-IDF Vectorization to convert job descriptions into numerical features.
   * Encoded location-related information.

4. Model Training

   * Trained a Random Forest Classifier.
   * Used balanced class weights to handle class imbalance.
   * Evaluated model performance on test data.

5. Prediction and Deployment

   * Integrated the trained model into a Flask web application.
   * Enabled real-time fraud prediction through a user-friendly interface.

Technologies Used:

• Programming Language: Python
• Framework: Flask
• Machine Learning: Scikit-learn
• Data Processing: Pandas, NumPy
• NLP: TF-IDF Vectorization
• Database: SQLite
• Frontend: HTML, CSS
• Model: Random Forest Classifier

Outcome:

The system successfully automates the identification of suspicious job postings, helping users make informed decisions while applying for jobs. The project demonstrates skills in machine learning, natural language processing, data preprocessing, model deployment, web development, and full-stack integration.
