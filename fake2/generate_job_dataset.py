import pandas as pd
from faker import Faker
import random
from collections import defaultdict

fake = Faker()

job_titles = [
    "Software Engineer", "Data Analyst", "Marketing Manager",
    "Sales Representative", "HR Coordinator", "Product Manager",
    "Financial Analyst", "Customer Support", "Graphic Designer"
]

skills = [
    "Python", "SQL", "Excel", "Communication", "Leadership",
    "Project Management", "Data Visualization", "Marketing",
    "Sales", "Customer Service"
]

def generate_description(is_fraud):
    description = f"We are looking for a {random.choice(job_titles)} "
    description += f"with experience in {random.choice(skills)}"
    
    if is_fraud:
        
        red_flags = [
            "No interview required - immediate hiring!",
            "Earn $1000+ weekly from home!",
            "No experience needed - we train you!",
            "Provide bank details to receive payments",
            "Transfer funds for international clients",
            "Get rich quick with minimal effort",
            "Guaranteed high income opportunity",
            "No background check required",
            "Send personal documents to get started",
            "First payment bonus available immediately"
        ]
        description += f". {random.choice(red_flags)}"
    else:
        
        legit_requirements = [
            "Bachelor's degree required",
            "2+ years experience",
            "Competitive salary",
            "Full benefits package",
            "On-site position",
            "Background check required",
            "Professional references needed"
        ]
        description += f". {random.choice(legit_requirements)}"
    
    return description


legitimate_postings = [
    {
        'job_description': 'Senior Data Scientist (5+ years) at TechAnalytics Bangalore. Requires: Python, TensorFlow, SQL. Formal hiring process with technical screening. Benefits include health insurance and stock options.',
        'location': 'Bangalore',
        'fraudulent': 0
    },
    {
        'job_description': 'Front-End Developer at DesignHub Mumbai. 3+ years React experience required. On-site position with competitive salary. Must provide portfolio and references.',
        'location': 'Mumbai',
        'fraudulent': 0 
    },
    {
        'job_description': 'Marketing Manager at BrandGrowth Delhi. Requires MBA and 4+ years experience. Office-based role with performance bonuses. No upfront fees - all recruitment costs covered by company.',
        'location': 'Delhi',
        'fraudulent': 0
    },
    {
        'job_description': 'Java Developer at FinServ Hyderabad. 4+ years Spring Boot experience required. Background check and educational verification mandatory. Comprehensive benefits package.',
        'location': 'Hyderabad',
        'fraudulent': 0
    },
    {
        'job_description': 'DevOps Engineer at CloudSolutions Pune. AWS/GCP certification required. Full-time office position. Formal interview process with HR and technical rounds.',
        'location': 'Pune',
        'fraudulent': 0
    }
]


def generate_dataset(num_samples=1000):
    data = defaultdict(list)
    fraud_count = int(num_samples * 0.2) 
    
    
    for posting in legitimate_postings:
        data['job_description'].append(posting['job_description'])
        data['location'].append(posting['location'])
        data['fraudulent'].append(posting['fraudulent'])
    
    
    remaining_samples = num_samples - len(legitimate_postings)
    for i in range(remaining_samples):
        is_fraud = i < (fraud_count - len(legitimate_postings))
        data['job_description'].append(generate_description(is_fraud))
        data['location'].append(fake.city())
        data['fraudulent'].append(int(is_fraud))
    
    return pd.DataFrame(data)


dataset = generate_dataset(1000)
dataset.to_csv('job_postings.csv', index=False)
print("Dataset generated and saved as job_postings.csv")
