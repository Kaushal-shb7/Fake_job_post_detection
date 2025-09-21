import requests
from bs4 import BeautifulSoup
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# Load the trained model (Random Forest)
rf_model = joblib.load("random_forest_model.pkl")

# Load the trained TF-IDF Vectorizer
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean and preprocess job description
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()  # Convert to lowercase
    return text

# Function to scrape job description from URL
def get_job_description(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try different selectors
            job_desc = soup.find("div", {"class": "job-description"})
            if not job_desc:
                job_desc = soup.find("div", {"class": "description"})
            if not job_desc:
                job_desc = soup.find("section", {"class": "job-content"})
            if not job_desc:
                job_desc = soup.find("div", {"class": "job-details"})
            
            if job_desc:
                return clean_text(job_desc.get_text())
    except Exception as e:
        print("Requests scraping failed:", e)
    
    print("Trying Selenium...")
    return get_job_description_selenium(url)

# Function to scrape job description using Selenium (for JavaScript-rendered pages)
def get_job_description_selenium(url):
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    # Set up WebDriver
    service = Service("chromedriver.exe")  # Update path to ChromeDriver
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        # Try extracting job description
        try:
            job_desc = driver.find_element(By.CLASS_NAME, "job-description")
            text = job_desc.text
        except:
            job_desc = driver.find_element(By.CLASS_NAME, "description")
            text = job_desc.text

        driver.quit()
        return clean_text(text)
    
    except Exception as e:
        print("Selenium scraping failed:", e)
        driver.quit()
        return None

# Function to predict if job post is fake or real
def predict_fake_job():
    job_url = input("Enter the job posting URL: ").strip()
    
    job_text = get_job_description(job_url)
    if job_text is None:
        return "Unable to fetch job description. Please check the URL."

    # Convert text into numerical features using TF-IDF
    job_text_vectorized = tfidf_vectorizer.transform([job_text])

    # Predict using Random Forest
    prediction = rf_model.predict(job_text_vectorized)

    return "Fake Job Posting" if prediction[0] == 1 else "Real Job Posting"

# Run the prediction with user input
print(predict_fake_job())
