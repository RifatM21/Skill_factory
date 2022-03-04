import time
import warnings

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup
from joblib import Parallel, delayed

pd.set_option('display.max_columns', None)

broken = []

def parser(link):
    warnings.simplefilter('ignore')

    global broken
    info = {}
    s = Service('D:\pythonProject\Pr_6\chromedriver.exe')

    driver = webdriver.Chrome(service=s)
    driver.get(link)
    try:
        submit_button = driver.find_element(By.ID, 'confirm-button')
        submit_button.click()
    except:
        broken.append(link)
        driver.close()

    time.sleep(2)
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH,
                    "//div[@class='ComplectationGroupsDesktop__cut']/span[@class='Link SpoilerLink SpoilerLink_type_default']"
                )
            )
        ).click()
    except:
        pass

    try:
        page_main = driver.execute_script('return document.body.innerHTML;')
        soup = BeautifulSoup(page_main, 'html.parser')

        price = soup.find('span', class_='OfferPriceCaption__price')
        if price:
            info['price'] = int(price.get_text().replace('\xa0', '').replace('₽', ''))
            info['model_name'] = soup.find('h1', class_='CardHead__title').get_text()

            for brand in brand_list:
                if link.find(brand) != -1:
                    info['brand'] = brand

                # info['description'] = soup.find('div', class_='CardDescriptionHTML').get_text()

            for child in soup.find('ul', class_='CardInfo').children:
                child_for_dict = child.get_text(':').replace('\xa0', '').split(':')
                if child_for_dict[0] == 'Двигатель':
                    try:
                        info['engineDisplacement'] = child_for_dict[1].split(' / ')[0]
                        info['enginePower'] = child_for_dict[1].split(' / ')[1]
                        info['fuelType'] = child_for_dict[2]
                    except:
                        info['fuelType'] = 0
                elif child_for_dict[0] == 'Комплектация':
                    pass
                else:
                    info[child_for_dict[0]] = child_for_dict[1]

            complectation = []
            complect_html = soup.find_all('li', class_='ComplectationGroupsDesktop__item')
            for item in complect_html:
                complectation.append(item.get_text().replace('•', ''))
            info['Complectation'] = complectation

            link = driver.find_elements_by_xpath(
                '//a[@class="Link SpoilerLink CardCatalogLink SpoilerLink_type_default"]')[0].get_attribute('href')
            driver.get(link)

            page_catalog = driver.execute_script('return document.body.innerHTML;')
            soup = BeautifulSoup(page_catalog, 'html.parser')
            try:
                catalog_info = soup.find_all('dl', class_='list-values clearfix')
                base_info = catalog_info[0].get_text('/').split('/')
                for i in range(0, 7, 2):
                    info[base_info[i]] = base_info[i + 1]
                if info['fuelType'] == 0:
                    info['fuelType'] = catalog_info[-1].get_text('/').split('/')[1]
            except:
                info['Страна марки'] = info['Класс автомобиля'] = info['Количество дверей'] = info['Количество месте'] = 0

            driver.close()
    except:
        pass
    return info


brand_list = ['skoda', 'audi', 'honda', 'volvo', 'bmw', 'nissan', 'infiniti', 'mercedes',
              'toyota', 'lexus', 'volkswagen', 'mitsubishi']

links = list(open('links.txt', 'r').read().splitlines())

results = Parallel(n_jobs=8)(delayed(parser)(link) for link in links)
data = pd.DataFrame(results)
data = data.drop(['Госномер', 'Гарантия', 'Владение', 'VIN'], axis=1)

data.to_csv('train.csv', index=False)
print(data.info())
