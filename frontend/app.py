from flask import Flask, render_template, request
import joblib
import requests
from bs4 import BeautifulSoup

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to extract job details from a URL
def extract_job_details_from_url(job_url):
    try:
        response = requests.get(job_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else ''
        description = soup.find('div', {'class': 'description'}).get_text(strip=True) if soup.find('div', {'class': 'description'}) else ''
        requirements = soup.find('div', {'class': 'requirements'}).get_text(strip=True) if soup.find('div', {'class': 'requirements'}) else ''
        benefits = soup.find('div', {'class': 'benefits'}).get_text(strip=True) if soup.find('div', {'class': 'benefits'}) else ''
        
        return f"{title} {description} {requirements} {benefits}"
    except Exception as e:
        return f"Error: {e}"

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting job authenticity
@app.route('/predict', methods=['POST'])
def predict():
    job_url = request.form['job_url']  # Get the job URL from the form
    job_text = extract_job_details_from_url(job_url)
    
    if "Error:" in job_text or not job_text:
        return render_template('index.html', result="Could not extract job details. Please check the URL.")
    
    # Transform the text using the vectorizer
    job_text_tfidf = vectorizer.transform([job_text])
    
    # Predict using the trained model
    prediction = model.predict(job_text_tfidf)
    
    # Determine result
    result = "Fake Job" if prediction[0] else "Real Job"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
