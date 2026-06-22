import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import joblib


data = pd.read_csv('job_postings.csv')


common_indian_locations = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
    "Chennai", "Kolkata", "Pune", "Jaipur", "Surat",
    "Lucknow", "Kanpur", "Nagpur", "Visakhapatnam", "Patna"
]


X = data[['job_description', 'location']]
y = data['fraudulent']


text_transformer = TfidfVectorizer(max_features=1000, stop_words='english')


location_encoder = LabelEncoder()
location_encoder.fit(common_indian_locations + list(data['location'].unique()))


X_text = text_transformer.fit_transform(X['job_description'])
X_loc = location_encoder.fit_transform(X['location'])


from scipy.sparse import hstack
X_combined = hstack([X_text, X_loc.reshape(-1, 1)])


X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

def predict_job(job_description, location):
    """
    Predict if a job posting is fraudulent
    
    Args:
        job_description (str): The job description text
        location (str): The job location
        
    Returns:
        tuple: (prediction, confidence)
            prediction: 'Fraudulent' or 'Not Fraudulent'
            confidence: Prediction confidence percentage
    """
   
    text_features = text_transformer.transform([job_description])
    
    
    try:
        location_encoded = location_encoder.transform([location])
    except ValueError:
        location_encoded = [-1]  
    
   
    from scipy.sparse import hstack
    features = hstack([text_features, location_encoded.reshape(1, -1)])
    
    
    proba = model.predict_proba(features)
    result = 'Fraudulent' if proba[0][1] > 0.5 else 'Not Fraudulent'
    confidence = round(max(proba[0]) * 100, 1)
    
    return result, confidence


joblib.dump({
    'model': model,
    'text_transformer': text_transformer,
    'location_encoder': location_encoder
}, 'model.joblib')
