from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
try:
    s = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=s)
    driver.get('http://google.com')
except WebDriverException as e:
    print(f"WebDriverException: {e}")
# driver = webdriver.Chrome(ChromeDriverManager().install())
# browser = webdriver.Chrome()
# br
# from webdriver_manager.chrome import ChromeDriverManager


# browser = webdriver.Chrome()
# browser.get('http://localhost:8000')