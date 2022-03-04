import time
import warnings

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup
from joblib import Parallel, delayed


def link_parser(brand):
    warnings.simplefilter('ignore')

    s = Service('D:\pythonProject\Pr_6\chromedriver.exe')
    driver = webdriver.Chrome(service=s)

    href = []
    url = f'https://auto.ru/moskva/cars/{brand}/used/?page=1'

    driver.get(url)

    submit_button = driver.find_element(By.ID, 'confirm-button')
    submit_button.click()

    time.sleep(2)

    page = driver.execute_script('return document.body.innerHTML;')
    soup = BeautifulSoup(page, 'html.parser')

    pages_num = int(soup.find('span', class_='ControlGroup ControlGroup_responsive_no ControlGroup_size_s '
                                             'ListingPagination__pages').get_text(',').split(',')[-1])


    for page_num in range(1, pages_num + 1):
        if page_num != 1:
            url = f'https://auto.ru/moskva/cars/{brand}/used/?page={page_num}'
            driver.get(url)
        items = driver.find_elements_by_xpath('//a[@class="Link ListingItemTitle__link"]')
        for item in range(len(items)):
            href.append(items[item].get_attribute('href'))
        time.sleep(0.5)

    driver.close()
    return href


brand_list = ['skoda', 'audi', 'honda', 'volvo', 'bmw', 'nissan', 'infiniti', 'mercedes',
              'toyota', 'lexus', 'volkswagen', 'mitsubishi']

links = []
results = Parallel(n_jobs=6)(delayed(link_parser)(brand) for brand in brand_list)
for result in results:
    links.extend(result)

with open('links.txt', 'w') as file:
    for i in links:
        file.write(i + '\n')
    file.close()
