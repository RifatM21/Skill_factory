import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore')


def owners(cell):
    if cell in ['1владелец', '1 владелец']:
        return 1
    elif cell in ['2владельца', '2 владельца']:
        return 2
    else:
        return 3


def vendor(cell):
    if cell in ['bmw', 'mercedes', 'volkswagen', 'audi']:
        return 'Германия'
    elif cell == 'volvo':
        return 'Швеция'
    elif cell == 'skoda':
        return 'Чехия'
    else:
        return 'Япония'


pd.set_option('display.max_columns', None)

data_test = pd.read_csv('test.csv')
data = pd.read_csv('train.csv')

data = data.drop(
    ['Статус', 'Кузов №', 'Количество месте', 'Запас хода', 'Количество мест', 'Налог', 'Обмен',
     'Класс автомобиля', 'Таможня'], axis=1
                 )

data = data.rename(
    columns={'год выпуска': 'productionDate', 'Пробег': 'mileage', 'Кузов': 'bodyType', 'Цвет': 'color',
             'Страна марки': 'vendor', 'Коробка': 'vehicleTransmission', 'Количество дверей': 'numberOfDoors',
             'Complectation': 'complectation_dict'}
                   )

data = data[['bodyType', 'brand', 'color', 'complectation_dict', 'engineDisplacement', 'enginePower', 'fuelType',
             'mileage', 'model_name', 'numberOfDoors', 'productionDate', 'vehicleTransmission', 'vendor', 'Владельцы',
             'ПТС', 'Привод', 'Руль', 'Состояние', 'price']]

columns = list(data.columns)[:-1]

data = data.dropna(how='all').reset_index(drop=True)

data['mileage'] = data['mileage'].apply(lambda x: int(x[:x.find('км')]))

data['engineDisplacement'] = data['engineDisplacement'].apply(lambda x: float(x[:x.find('л')]))
data['enginePower'] = data['enginePower'].apply(
    lambda x: int(float(x[:x.find('кВт')]) * 1.36) if x.find('л.') == -1 else int(x[:x.find('л')]))

check_list = data[data['numberOfDoors'] == 0]['model_name'].to_list()

for model in check_list:
    data[(data['model_name'] == model) & (data['numberOfDoors'] == 0)] = \
        data[(data['model_name'] == model) & (data['numberOfDoors'] != 0)].mode()

data = data.dropna(how='all').reset_index(drop=True)

data['fuelType'] = data['fuelType'].apply(lambda x: 'электро' if len(str(x)) == 3 else x.lower())
data['fuelType'] = data['fuelType'].apply(lambda x: 'газ' if x.find('газобаллонное') != -1 else x)

for row in range(data.shape[0]):
    if data['fuelType'].iloc[row] == 'электро':
        data['engineDisplacement'].iloc[row] = 0

data['Владельцы'] = data['Владельцы'].apply(owners).astype(object)

data['complectation_dict'] = data['complectation_dict'].apply(
    lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', '))
data['complectation'] = data['complectation_dict'].apply(lambda x: len(x))
data = data.drop(['complectation_dict'], axis=1)

data['price'] = data['price'].apply(lambda x: x / 100)

# ================================================================================
# Test data preparing

data_test = data_test[columns]
data_test['brand'] = data_test['brand'].apply(lambda x: x.lower())

data_test['complectation_dict'] = data_test['complectation_dict'].fillna(0)
data_test['complectation'] = data_test['complectation_dict'].apply(
    lambda x: len(x[x.find('[') + 1:x.find(']')].split('","')) if x else x)
data_test = data_test.drop(['complectation_dict'], axis=1)

data_test['engineDisplacement'] = data_test['engineDisplacement'].apply(
    lambda x: float(x[:x.find('LTR')]) if len(x) > 4 else 0)

data_test['enginePower'] = data_test['enginePower'].apply(lambda x: int(x[:x.find('N')]))

data_test['vendor'] = data_test['brand'].apply(vendor)

data_test['Владельцы'] = data_test['Владельцы'].apply(lambda x: owners(x.replace('\xa0', ''))).astype(object)

# data.to_csv('train_prepared.csv', index=False)
# data_test.to_csv('test_prepared.csv', index=False)

