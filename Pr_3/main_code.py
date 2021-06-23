import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split  # Инструмент для разбивки
from sklearn.ensemble import RandomForestRegressor  # Инструмент для создания и обучения модели
from sklearn import metrics  # Инструменты для оценки точности модели
import warnings
warnings.simplefilter('ignore')


def review_date(row):  # Функция, возвращающая даты отзывов
    dates = row[-1]
    dates = dates.replace(']', '')
    dates = dates.replace("'", '')
    dates = dates.strip()
    if len(dates) <= 2:
        return None
    else:
        dates = dates.split(',')
        real_dates = []
        for date in dates:
            date = date.strip()
            dt = datetime.strptime(date, '%m/%d/%Y')
            real_dates.append(dt)
        return real_dates


def dates_dif(row):  # Функция, считающая разницу между первым и последним отзывом
    if row is None:
        return None
    elif len(row) == 2:
        first = row[0]
        second = row[1]
        if first > second:
            dif = first - second
        else:
            dif = second - first
        return dif.days
    else:
        return None


def price_change(row):  # Функция для замены обозначений Price Range
    if row == '$':
        return 'Low'
    elif row == '$$ - $$$':
        return 'Middle'
    elif row == '$$$$':
        return 'High'
    else:
        return None


def find_good_words(row):  # Функция, находящая положительные отзывы
    good_words = ['good', 'yummy', 'fine', 'great', 'tasty',
                  'satisfaction', 'amazing', 'nice', 'best',
                  'friendly', 'pleasant', 'excellent', 'loved',
                  'love', 'lovely', 'welcoming', 'wonderful',
                  'perfect', 'delicious', 'favourite', 'sweet',
                  'yum', 'adequate', 'happy', 'beautiful', 'liked', 'like']
    count = 0
    for word in good_words:
        if row is None:
            count = 0
        elif word in row:
            count += 1
    if count >= 1:
        result = 1
    else:
        result = 0
    return result


def number_reviews(row):  # Функция, заменяющая пропуски в Number of Reviews, в зависимости от Reviews
    if pd.isnull(row['Number of Reviews']):
        if row['Reviews'] is None:
            return 0
        return 1
    return row['Number of Reviews']


def analysis(data):  # Основная функция
    # Замена пропусков в 'Price Range' на самое частое - '$$ - $$$' (средняя цена)
    data['Price Range'] = data['Price Range'].fillna('$$ - $$$')

    # Замена обозначений 'Price Range'
    data['Price Range'] = data['Price Range'].apply(price_change)

    # Замена пропусков 'Cuisine Style' на 'European'
    data['Cuisine Style'] = data['Cuisine Style'].fillna('European')

    # Обработка 'Cuisine Style'
    data['Cuisine Style'] = data['Cuisine Style'].str.replace('[', '')
    data['Cuisine Style'] = data['Cuisine Style'].str.replace(']', '')
    data['Cuisine Style'] = data['Cuisine Style'].str.replace('""', '')
    data['Cuisine Style'] = data['Cuisine Style'].str.split(',')

    # Новый критерий 'Number of Cuisine' - количество кухонь в ресторане
    data['Number of Cuisine'] = data['Cuisine Style'].apply(len)

    # Обработка 'Reviews'
    data['Reviews'] = data['Reviews'].str.replace('[', '')
    data['Reviews'] = data['Reviews'].str.split('],')

    # Заполнение пропусков в 'Reviews'
    data['Reviews'] = data['Reviews'].fillna('],')

    # Новый критерий 'Dates of Reviews' - даты отзывов
    data['Dates of Reviews'] = data['Reviews'].apply(review_date)

    # Удалений дат отзывов из столбца 'Reviews' и замена пустых строк на None
    data['Reviews'] = data['Reviews'].apply(lambda x: x[0] if len(x[0]) > 5 else None)

    # Заполнение пропусков в 'Number of Reviews' с учётом 'Reviews'
    data['Number of Reviews'] = data.apply(number_reviews, axis=1)

    # Новый критерий 'Good Reviews' - показатель хорошего отзыва
    data['Good Reviews'] = data['Reviews'].str.lower().apply(find_good_words)

    # Новый показатель 'Difference between Dates' - разница между первым и последним отзывом
    data['Difference between Dates'] = data['Dates of Reviews'].apply(dates_dif)

    # Заполнение пропусков в 'Difference between Dates' 0-ми
    data['Difference between Dates'] = data['Difference between Dates'].fillna(0)

    # Удаление лишних критериев из датасета
    data = data.drop(['URL_TA', 'ID_TA', 'Dates of Reviews', 'Reviews', 'Restaurant_id',
                      'Cuisine Style', 'Price Range'], axis=1)

    # Создание dummy-переменных по 'City'
    data = pd.get_dummies(data, columns=['City'], dummy_na=True)
    return data


pd.set_option('display.max_columns', 10)

data = pd.read_csv('main_task_new.csv')
kaggle_task = pd.read_csv('kaggle_task.csv')

solution = pd.DataFrame(columns=['Restaurant_id', 'Rating'])
solution['Restaurant_id'] = kaggle_task['Restaurant_id']

print(data.head())
print()
print(data.info())
print()

data = analysis(data)

# Построим таблицу корреляций для полученных признаков
print(data[['Ranking', 'Rating', 'Number of Reviews', 'Number of Cuisine',
            'Difference between Dates', 'Good Reviews']].corr())
print()

# Построим график корреляций новых признаков
fig, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(data[['Ranking', 'Rating', 'Number of Reviews', 'Number of Cuisine', 'Difference between Dates',
                  'Good Reviews']].corr().round(2), annot=True, mask=np.triu(data[['Ranking', 'Rating',
                                                                                   'Number of Reviews',
                                                                                   'Number of Cuisine',
                                                                                   'Difference between Dates',
                                                                                   'Good Reviews']].corr()), ax=ax)
plt.show()

kaggle_task = analysis(kaggle_task)

# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = data.drop(['Rating'], axis=1)
y = data['Rating']

# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования используется 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Модель
regr = RandomForestRegressor(n_estimators=100)

# Обучение модели на тестовом наборе данных
regr.fit(X_train, y_train)

# Предсказанные, на основе тестовой выборки, данные рейтинга записываются в переменную y_pred
y_pred = regr.predict(X_test)

# Предсказываем значения выборки с Kaggle
kaggle_pred = regr.predict(kaggle_task)

# Округление предсказанных значений с точностью 0.5
y_pred = np.round(y_pred * 2) / 2
kaggle_pred = np.round(kaggle_pred * 2) / 2

# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

# Записываем значения результатов рейтингов для выборки с Kaggle в итоговый датасет
solution['Rating'] = kaggle_pred
print(solution)
print()

# Сохраняем итоговый датасет
# solution.to_csv('solution.csv', index=None)


