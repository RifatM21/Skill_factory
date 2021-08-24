import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
warnings.simplefilter('ignore')


def fuel_cost_per_month(date):
    month = date.month
    if month == 1:
        cost = 48.9
    elif month == 2:
        cost = 46.7
    elif month == 12:
        cost = 55.6
    return cost


pd.set_option('display.max_columns', None)

data = pd.read_csv('data_sql.csv')

print('Изначальный датасет:')
print(data.head(10))
print('\nИнформация о датасете:')
print(data.info())

data['tickets_sold'] = data['tickets_economy'] + data['tickets_business']
data['occupancy'] = round((data['tickets_sold'] / data['seats_all']), 2) * 100
data['fuel_consumption'] = data['model'].apply(lambda x: 43.34 if x == 'Boeing 737-300' else 28.34)
data['actual_departure'] = data['actual_departure'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
data['actual_arrival'] = data['actual_arrival'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))
data['fuel_price'] = data['actual_departure'].apply(fuel_cost_per_month)
data['fuel_costs'] = data['fuel_consumption'] * data['fuel_price'] * data['flight_time_min']
data['profit'] = data['total_amount'] - data['fuel_costs']
data['profit_14%'] = data['total_amount'] * 0.14 - data['fuel_costs']

print('\nПромежуточный датасет:')
print(data.head(10))
print('\nИнформация о промежуточном датасете:')
print(data.info())

data_novokuznetsk = data[data['arrival_city'] == 'Novokuznetsk']
print('\nДатасет по городу Новокузнецк:')
print(data_novokuznetsk[['flight_no', 'arrival_city', 'seats_all', 'total_amount', 'tickets_sold']])

data = data[~(data['arrival_city'] == 'Novokuznetsk')]

final_data = data.drop(['departure_airport', 'actual_arrival', 'actual_departure', 'tickets_economy',
                        'tickets_business', 'fuel_consumption', 'fuel_price'], axis=1)

print('\nИтоговый датасет:')
print(final_data.head(10))
print('\nИнформация об итоговом датасете:')
print(final_data.info())

print('\nПрибыль по напралвениям:')
print(final_data.groupby(['arrival_city'])['profit'].sum())
print(final_data.groupby(['arrival_city'])['profit_14%'].sum())


losses = data[data['profit_14%'] < 10000].sort_values(by='profit_14%', ascending=True)
print('\nУбыточные рейсы')
print(losses)
