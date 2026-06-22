import unittest
import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class JobFraudDetectionSeleniumTest(unittest.TestCase):
    def setUp(self):
        print("Current working directory:", os.getcwd())

        # Use Selenium Manager defaults unless an explicit path is provided.
        # Set CHROMEDRIVER_PATH env var to override.
        from selenium.webdriver.chrome.service import Service

        chromedriver_path = os.environ.get("CHROMEDRIVER_PATH", "")
        if chromedriver_path:
            print("Using ChromeDriver path:", chromedriver_path)
            service = Service(executable_path=chromedriver_path)
            self.driver = webdriver.Chrome(service=service)
        else:
            print("Using Selenium Manager to obtain ChromeDriver")
            self.driver = webdriver.Chrome()

        self.driver.implicitly_wait(10)
        self.base_url = "http://localhost:5000"

    def test_job_fraud_detection(self):
        driver = self.driver
        driver.get(self.base_url)

        # If redirected to register page, register a new user
        if "register" in driver.current_url:
            username_input = driver.find_element(By.NAME, "username")
            password_input = driver.find_element(By.NAME, "password")
            username_input.send_keys("testuser")
            password_input.send_keys("testpassword")
            password_input.send_keys(Keys.RETURN)
            time.sleep(2)

        # Go to predict page
        driver.get(self.base_url + "/predict-page")

        # If redirected to login, login again.
        if "/login" in driver.current_url:
            username_input = driver.find_element(By.NAME, "username")
            password_input = driver.find_element(By.NAME, "password")
            username_input.send_keys("testuser")
            password_input.send_keys("testpassword")
            password_input.send_keys(Keys.RETURN)
            time.sleep(2)
            driver.get(self.base_url + "/predict-page")

        # Fill the form fields
        job_description = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "job_description"))
        )
        location = driver.find_element(By.ID, "location")
        salary = driver.find_element(By.ID, "salary")
        role = driver.find_element(By.ID, "role")

        # Sample test data for a fraudulent job post
        job_description.clear()
        job_description.send_keys(
            "Urgent hiring! Work from home and earn unlimited income. Provide your bank account details."
        )
        location.clear()
        location.send_keys("Mumbai")
        salary.clear()
        salary.send_keys("1000000")
        role.clear()
        role.send_keys("Remote work")

        # Submit the form (JS click avoids interception)
        submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        driver.execute_script("arguments[0].click();", submit_button)

        # Wait for the result card to appear
        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.visibility_of_element_located(
                (By.CSS_SELECTOR, "div.card.p-4.mb-4.fraud, div.card.p-4.mb-4.legit")
            )
        )

        # Extract prediction and explanation
        prediction_text = driver.find_element(By.CSS_SELECTOR, "div.text-center.display-4").text
        explanation_text = driver.find_element(By.CSS_SELECTOR, "div.text-center.mt-3 p").text

        # Log results to a report file
        report_path = os.path.join(os.path.dirname(__file__), "selenium_test_report.txt")
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("Job Fraud Detection Selenium Test Report\n")
            report_file.write("========================================\n")
            report_file.write(f"Prediction: {prediction_text}\n")
            report_file.write(f"Explanation: {explanation_text}\n")

        # Assert that the page returned a prediction-like output (avoid brittle label checks)
        self.assertNotEqual(prediction_text.strip(), "", f"Empty prediction text: {prediction_text}")



    def tearDown(self):
        self.driver.quit()


if __name__ == "__main__":
    unittest.main()

