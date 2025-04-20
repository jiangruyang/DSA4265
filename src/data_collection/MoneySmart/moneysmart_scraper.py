import os
import requests
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
import time

# Ensure output directory exists
output_dir = "./data/card_tcs/json"
os.makedirs(output_dir, exist_ok=True)

# URL of the Citi PremierMiles Card page
### CHANGE THIS
url = "https://www.moneysmart.sg/credit-cards/uob-lazada-card"

# Set up Selenium with Chrome
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Temporarily disable headless mode for debugging
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)
# driver.get(url)
# Extract the title after JavaScript renders it
### CHANGE THIS
card_name = "Lazada-UOB Card"
card_type = "Cash Rebate"
issuer = "UOB"

def expand_section(section_title):
    """Helper function to expand a section."""
    try:
        # Find and click the section with exact text match, handling potential whitespace
        section = wait.until(EC.element_to_be_clickable((By.XPATH, f"//div[normalize-space(text())='{section_title}']")))
        
        # Scroll the section into view
        driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", section)
        time.sleep(1)  # Wait for scroll to complete
        
        # Try to click the section
        try:
            section.click()
        except ElementClickInterceptedException:
            # If direct click fails, try JavaScript click
            driver.execute_script("arguments[0].click();", section)
        
        # Wait for the content to be visible
        time.sleep(1)
        return True
    except Exception as e:
        print(f"Error expanding section {section_title}: {e}")
        return False

def non_expand_content(section_title):
    """Helper function to check if a non-expandable section exists."""
    try:
        # Find the section by exact text match
        section = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/h2'))) 
        #section = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="promotion-eligibility--null-0"]/div[1]/h2/div')))
        return True
    except Exception as e:                                                  
        print(f"Error finding section '{section_title}': {str(e)}")
        return False

try:
    # Load the page
    driver.get(url)
    
    # Wait for the page to load completely
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
    time.sleep(2)  # Additional wait for dynamic content
    
    # Dictionary to store all extracted information
    card_info = {}
    card_info["card_name"] = card_name
    card_info["card_type"] = card_type
    card_info["issuer"] = issuer
    
    # Initialize rewards and benefits dictionary
    card_info["rewards_and_benefits"] = {}
    
    # Extract non-expandable sections first
    if non_expand_content("ARE YOU ELIGIBLE?"):
        card_info["eligibility"] = []
        card_info["eligibility"].append("Minimum age 21 years old")
        #card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[1]/div/p'))).text) 
        card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="//*[@id="promotion-eligibility--null-0"]/div[1]/ul/li/div/ul/li[1]'))).text)
        #card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[2]'))).text)
        #card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[3]'))).text)
        #card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[4]'))).text) 
        #card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[5]'))).text)
        #card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[6]'))).text)
        # card_info["eligibility"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="campaign-eligibility--null-0"]/div[1]/ul/li/div/ul/li[7]'))).text)

    
    # Extract Key Features
    if expand_section("Key Features"):
        card_info["rewards_and_benefits"]["key_features"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[1]/ul/li/p/span'))).text) 
        card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[2]/ul/li/p/span'))).text)
        card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[3]/ul/li/p/span'))).text)
        card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[4]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[5]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[6]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[7]/ul/li/p/span'))).text)
        #card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[8]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[9]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[10]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[11]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[12]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[13]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[14]/ul/li/p/span'))).text)
        # card_info["rewards_and_benefits"]["key_features"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="key-features-panel"]/div[15]/ul/li/p/span'))).text)

    


    if expand_section("Air Miles"):
        card_info["rewards_and_benefits"]["air_miles"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[1]/p/span'))).text)
        card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[5]/p/span'))).text)
        # card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[6]/p/span'))).text)
        #card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[7]/p/span'))).text)
        # card_info["rewards_and_benefits"]["air_miles"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[8]/p/span'))).text)
   
    if expand_section("Overseas Spending"):
        card_info["rewards_and_benefits"]["overseas_spending"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[1]/p/span'))).text)
        card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[5]/p/span'))).text)
        #card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["overseas_spending"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-7"]/div/ul/li[7]/p/span'))).text)


    if expand_section("Bill Payment"):
        card_info["rewards_and_benefits"]["bill_payment"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["bill_payment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[1]/p/span'))).text) 
        card_info["rewards_and_benefits"]["bill_payment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["bill_payment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[3]/p/span'))).text)
        card_info["rewards_and_benefits"]["bill_payment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[4]/p/span'))).text)
        card_info["rewards_and_benefits"]["bill_payment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[5]/p/span'))).text)
        card_info["rewards_and_benefits"]["bill_payment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[6]/p/span'))).text)
   
    if expand_section("Petrol"):
        card_info["rewards_and_benefits"]["petrol"] = []  # Initialize as empty list
        #card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/p'))).text) 
        card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-9"]/div/ul/li[1]/p/span'))).text) 
        card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-9"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-9"]/div/ul/li[3]/p/span'))).text)
        #card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[5]/p/span'))).text)
        # card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[6]/p/span'))).text)
        # card_info["rewards_and_benefits"]["petrol"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[7]/p/span'))).text)

    if expand_section("Cash Back"):
        card_info["rewards_and_benefits"]["cash_back"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[1]/p/span'))).text)  
        card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[2]/p/span'))).text) 
        card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[3]/p/span'))).text)
        card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[5]/p/span'))).text)
        # card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[6]/p/span'))).text)
        # card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[7]/p/span'))).text)
        # card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[8]/p/span'))).text)
        # card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[9]/p/span'))).text)
        # card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[10]/p/span'))).text)
    #     card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[11]/p/span'))).text)
    #     card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[12]/p/span'))).text)
    #     card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[13]/p/span'))).text)
    #     card_info["rewards_and_benefits"]["cash_back"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[14]/p/span'))).text)

    if expand_section("Rewards"):
        card_info["rewards_and_benefits"]["rewards"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[1]/p/span'))).text) 
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[3]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[4]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[5]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[6]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[7]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[8]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[9]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[10]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[11]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[12]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[13]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[14]/p/span'))).text)
        card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[15]/p/span'))).text)
        # card_info["rewards_and_benefits"]["rewards"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[16]/p/span'))).text)
        

    if expand_section("Buffet Promotion"):
        card_info["rewards_and_benefits"]["buffet_promotion"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["buffet_promotion"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[1]/p/span'))).text) 
        card_info["rewards_and_benefits"]["buffet_promotion"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[2]/p/span'))).text)
        #card_info["rewards_and_benefits"]["buffet_promotion"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[3]/p/span'))).text)
        #card_info["rewards_and_benefits"]["buffet_promotion"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[2]/p/span'))).text)
        #card_info["rewards_and_benefits"]["buffet_promotion"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[3]/p/span'))).text)

    if expand_section("Entertainment"):
        card_info["rewards_and_benefits"]["entertainment"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["entertainment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[1]/p/span'))).text) 
        #card_info["rewards_and_benefits"]["entertainment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[2]/p/span'))).text)
        #card_info["rewards_and_benefits"]["entertainment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["entertainment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[4]/p/span'))).text)
        #card_info["rewards_and_benefits"]["entertainment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[5]/p/span'))).text)

    if expand_section("Grocery"):
        card_info["rewards_and_benefits"]["grocery"] = []  # Initialize as empty list
        #card_info["rewards_and_benefits"]["grocery"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/p'))).text) 
        card_info["rewards_and_benefits"]["grocery"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-4"]/div/ul/li[1]/p'))).text) 
        #card_info["rewards_and_benefits"]["grocery"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[2]/p'))).text)
        #card_info["rewards_and_benefits"]["grocery"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[3]/p'))).text)
        # card_info["rewards_and_benefits"]["grocery"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[4]/p'))).text)

    if expand_section("Dining"):
        card_info["rewards_and_benefits"]["dining"] = []  # Initialize as empty list
        #card_info["rewards_and_benefits"]["grocery"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/p'))).text) 
        card_info["rewards_and_benefits"]["dining"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-3"]/div/ul/li[1]/p/span'))).text) 
        #card_info["rewards_and_benefits"]["dining"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-7"]/div/ul/li[2]/p/span'))).text)
        #card_info["rewards_and_benefits"]["dining"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-7"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["dining"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-8"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["dining"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[5]/p/span'))).text)
        # card_info["rewards_and_benefits"]["dining"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[6]/p/span'))).text)

    if expand_section("Online Shopping"):
        card_info["rewards_and_benefits"]["online_shopping"] = []  # Initialize as empty list
        #card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/p'))).text) 
        card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[1]/p/span'))).text)  
        card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[2]/p/span'))).text)
        #card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-4"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[5]/p/span'))).text)
        # card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[6]/p/span'))).text)
        # card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[7]/p/span'))).text)
        # card_info["rewards_and_benefits"]["online_shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[8]/p/span'))).text)


    if expand_section("Shopping"):
        card_info["rewards_and_benefits"]["shopping"] = []  # Initialize as empty list
        #card_info["rewards_and_benefits"]["shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/p'))).text) 
        card_info["rewards_and_benefits"]["shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-6"]/div/ul/li[1]/p/span'))).text)
        card_info["rewards_and_benefits"]["shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-6"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-6"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-4"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["shopping"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[5]/p/span'))).text)

    if expand_section("Student"):
        card_info["rewards_and_benefits"]["student"] = []  # Initialize as empty list
        #card_info["rewards_and_benefits"]["student"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/p'))).text) 
        card_info["rewards_and_benefits"]["student"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[1]/p/span'))).text)
        card_info["rewards_and_benefits"]["student"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[2]/p/span'))).text)
        card_info["rewards_and_benefits"]["student"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[3]/p/span'))).text)
        card_info["rewards_and_benefits"]["student"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[4]/p/span'))).text)
        # card_info["rewards_and_benefits"]["student"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[5]/p/span'))).text)

    if expand_section("0% Instalment"):
        card_info["rewards_and_benefits"]["installment"] = []  # Initialize as empty list
        card_info["rewards_and_benefits"]["installment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-8"]/div/ul/li[1]/p/span'))).text)
        #card_info["rewards_and_benefits"]["installment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-2"]/div/ul/li[2]/p/span'))).text)
        #card_info["rewards_and_benefits"]["installment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-0"]/div/ul/li[3]/p/span'))).text)
        # card_info["rewards_and_benefits"]["installment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-1"]/div/ul/li[4]/p/span'))).text)
    #     card_info["rewards_and_benefits"]["installment"].append(wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="category-panel-5"]/div/ul/li[5]/p/span'))).text)

    # Extract Annual Interest Rate and Fees
    if expand_section("Annual Interest Rate and Fees"):
        card_info["fees"] = {
            "annual_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[1]/td'))).text, 
            "supplementary_card_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[2]/td'))).text, 
            "annual_fee_waiver": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[3]/td'))).text,
            "interest_free_period": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[4]/td'))).text,
            "annual_interest_rate": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[5]/td'))).text,
            "late_payment_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[6]/td'))).text,
            "minimum_monthly_repayment": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[7]/td'))).text,
            "foreign_transaction_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[8]/td'))).text,
            #"foreign_check_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[9]/td'))).text,
            "cash_advance_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[9]/td'))).text,
            "overlimit_fee": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[10]/td'))).text,
            #"annual_interest_rate_on_cash_advance": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[6]/td'))).text
        }
    
    # Extract Minimum Income Requirements
    if expand_section("Minimum Income Requirements"):
        card_info["minimum_income"] = {
            "singaporean_pr": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-1"]/div/table/tbody/tr[1]'))).text,  
            "non_singaporean": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-1"]/div/table/tbody/tr[2]'))).text
        }
    
    # Extract Documents Required
    if expand_section("Documents Required"):
        card_info["documents_required"] = {
            "sg_citizens_prs": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-2"]/div/table/tbody/tr[1]/td/span'))).text,
            "passport_validity": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-2"]/div/table/tbody/tr[2]/td/span'))).text,
            "work_permit_validity": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-2"]/div/table/tbody/tr[3]/td/span'))).text,
            "utility_bill": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-2"]/div/table/tbody/tr[4]/td/span'))).text,
            #"utility_bill": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="utility_or_telephone_bill_-1765864212"]'))).text,
            "income_tax_notice": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-2"]/div/table/tbody/tr[5]/td/span'))).text,
            "payslip": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-2"]/div/table/tbody/tr[6]/td/span'))).text
            #"income_tax_notice": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="income_tax_notice_of_assessment_-1669441137"]'))).text,
            #"payslip": wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="latest_original_computerised_payslip_801513154"]'))).text
        }
    
    # Extract Card Association
    if expand_section("Card Association"):
        card_info["card_association"] = 'Mastercard' ### CHANGE THIS
    
    
        
    # # Extract Wireless Payment
    # if expand_section("Wireless Payment"):
    #     card_info["wireless_payment"] = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="feature-panel-0"]/div/table/tbody/tr[1]/td'))).text
    
    # Print all extracted information
    # print("\nExtracted Card Information:")
    # for section, info in card_info.items():
    #     print(f"\n{section.upper()}:")
    #     if isinstance(info, dict):
    #         for key, value in info.items():
    #             print(f"  {key}: {value}")
    #     else:
    #         print(f"  {info}")
    
    # Save the extracted information to a JSON file
    output_file = os.path.join(output_dir, f"{card_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(card_info, f, indent=2, ensure_ascii=False)
    print(f"\nData saved to {output_file}")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the browser session
    driver.quit()
