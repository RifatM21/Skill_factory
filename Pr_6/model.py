import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_percentage_error as mape

from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')


def test_data_for_model(data):
    cat_list = ['bodyType', 'brand', 'color', 'fuelType', 'vehicleTransmission', 'vendor', 'Владельцы', 'Привод']
    bin_list = ['ПТС', 'Руль', 'Состояние']
    data = data.drop(['model_name', 'mileage', 'engineDisplacement'], axis=1)

    # Среднее количество деталей комплектации в зависимости от страны
    median_complectation_vendor = data.groupby('vendor')['complectation'].median().to_dict()
    data['median_complectation_vendor'] = data['vendor'].map(median_complectation_vendor)
    # Среднее количество владельцов в зависиомсти от
    median_owners_brand = data.groupby('brand')['Владельцы'].median().to_dict()
    data['median_owners_brand'] = data['brand'].map(median_owners_brand)
    # Средняя мощность двигателя с учётом брэнда автомобиля
    median_power_brand = data.groupby('brand')['enginePower'].median().to_dict()
    data['median_engine_brand'] = data['brand'].map(median_power_brand)

    le = LabelEncoder()
    for column in bin_list:
        data[column] = le.fit_transform(data[column])

    data = pd.get_dummies(data, columns=cat_list, drop_first=True)

    return data


def compute_meta_feature(clf, X_train, X_test, y_train, cv):
    X_meta_train = np.zeros((len(y_train), 1), dtype=np.float32)

    splits = cv.split(X_train)
    for train_fold_index, predict_fold_index in splits:
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]

        folded_clf = clone(clf)
        folded_clf.fit(X_fold_train, y_fold_train)

        y_pred = folded_clf.predict(X_fold_predict)

        X_meta_train[predict_fold_index, 0: 1] = np.reshape(y_pred, (-1, 1))

    meta_clf = clone(clf)
    meta_clf.fit(X_train, y_train)

    X_meta_test = np.reshape(meta_clf.predict(X_test), (-1, 1))

    return X_meta_train, X_meta_test


def generate_meta_features(classifiers, X_train, X_test, y_train, cv):

    features = [
        compute_meta_feature(clf, X_train, X_test, y_train, cv)
        for clf in tqdm(classifiers)
    ]

    stacked_features_train = np.hstack([
        features_train for features_train, features_test in features
    ])

    stacked_features_test = np.hstack([
        features_test for features_train, features_test in features
    ])

    return stacked_features_train, stacked_features_test


def compute_metric(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    return np.round(mape(y_test, y_test_pred) * 100, 6)


pd.set_option('display.max_columns', None)

data_test = pd.read_csv('test_prepared.csv')
data = pd.read_csv('train_prepared.csv')
data_submission = pd.read_csv('sample_submission.csv')

add = {'bodyType': 'фастбек', 'brand': 'nissan', 'color': 'коричневый', 'engineDisplacement': 2.7, 'enginePower': 179,
       'fuelType': 'бензин', 'mileage': 147000, 'model_name': 'nissan 280ZX', 'numberOfDoors': 3,
       'productionDate': 1980, 'vehicleTransmission': 'автоматическая', 'vendor': 'Япония', 'Владельцы': 2,
       'ПТС': 'Дубликат', 'Привод': 'задний', 'Руль': 'Левый', 'Состояние': 'Не требует ремонта', 'price': 20020,
       'complectation': 12}

data = data.append(add, ignore_index=True)

# В тренеровочном датасете целевая переменная price выражена в $ по текущему курсу ($ = 100 руб.)

print(data.info())
print()
print(data.describe())

# Удалим столбцы с названия моделей автомобилей, т.к названия, полученные при парсинге, сильно отличаются от тех,
# которые находятся в тестовом датасете

data = data.drop(['model_name'], axis=1)

# ============================================================================
# Exploratory data analysis

# Логарифмируем целевую переменную для нормального распределения цены
data['price'] = data['price'].apply(lambda x: np.log(x))

# Распределение целевой переменной (логарифмированной)
# plt.figure(figsize=(12, 6))
# sns.histplot(data=data, x='price')
# plt.title('Распределние целевой переменной (цены)')
# plt.xlabel('Price')
# plt.show()

box_plot_list = ['brand', 'vendor', 'Привод', 'fuelType', 'Владельцы', 'Руль', 'ПТС']
num_list = ['engineDisplacement', 'enginePower', 'mileage', 'numberOfDoors', 'productionDate', 'complectation', 'price']
cat_list = ['bodyType', 'brand', 'color', 'fuelType', 'vehicleTransmission', 'vendor', 'Владельцы', 'Привод']
bin_list = ['ПТС', 'Руль', 'Состояние']


# Box-plot'ы для основных категориальных признаков
# for feature in box_plot_list:
#     plt.figure(figsize=(12, 7))
#     sns.boxplot(data=data, x=feature, y='price')
#     plt.title(f'{feature.capitalize()}-Price', fontsize=20)
#     plt.xlabel(f'{feature.capitalize()}', fontsize=14)
#     plt.ylabel('Price', fontsize=14)
#     plt.show()

print("\nПо построенным box-plot'ам можно заметить, что немецкие авто стоят в среднем дороже остальных, самые дорогие\n"
      "машины марки - Mercedes. Также, у машин с полным приводом в среднем цена значительно выше, чем у других.\n"
      "Электро-мобили в среднем стоят дороже, чем автомобили в любым другим типом топлива. Чем больше владельцев\n"
      "было у автомобиля, тем он дешевле. Леворукие авто значительно дороже праворуких. Аналогично для автомобилей\n"
      "с оригинальным ПТС.")


# Построим график корреляциия численных признаков
# plt.figure(figsize=(12, 7))
# sns.heatmap(data[num_list].corr(), annot=True, fmt='.1f')
# plt.show()

print('\nПо графику корреляций можно заметить, что год выпуска и пробег автомобиля сильно скоррелированы между собой,\n'
      'также оба эти показателя сильно скоррелированные с целевой переменной. Оставим более скоррелированный с\n'
      'целевой переменной показатель - productionDate (год выпуска автомобиля). Также довольно сильно скоррелированы\n'
      'мощность двигателя и объём двигателя. Оставим - enginePower (мощность двигателя), т.к данный показатлей имеет\n'
      'меньше нулевых значений и он более скоррелирован с целевой переменной.')

data = data.drop(['mileage', 'engineDisplacement'], axis=1)

# ========================================================================
# Feature engineering

# Среднее количество деталей комплектации в зависимости от страны
median_complectation_vendor = data.groupby('vendor')['complectation'].median().to_dict()
data['median_complectation_vendor'] = data['vendor'].map(median_complectation_vendor)
# Среднее количество владельцов в зависиомсти от
median_owners_brand = data.groupby('brand')['Владельцы'].median().to_dict()
data['median_owners_brand'] = data['brand'].map(median_owners_brand)
# Средняя мощность двигателя с учётом брэнда автомобиля
median_power_brand = data.groupby('brand')['enginePower'].median().to_dict()
data['median_engine_brand'] = data['brand'].map(median_power_brand)

le = LabelEncoder()
for column in bin_list:
    data[column] = le.fit_transform(data[column])

data_label = pd.get_dummies(data, columns=cat_list, drop_first=True)

# ========================================================================
# Modeling
print(data_label.info())

RAND = 42

X = data_label.drop('price', axis=1).values
y = data_label['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RAND)

# Базовая линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print('\nMAPE for Linear Regression:', np.round(mape(y_test, y_pred_lr) * 100, 6), '%')

# Случайный лес
rfr = RandomForestRegressor(n_estimators=2000, max_features='auto', max_depth=10, min_samples_leaf=5,
                            min_samples_split=2, bootstrap=True, random_state=RAND)
rfr.fit(X_train, y_train)

y_pred_rfr = rfr.predict(X_test)
print('MAPE for Random Forest Regression:', np.round(mape(y_test, y_pred_rfr) * 100, 6), '%\n')

# Stacking

# cv = KFold(n_splits=5, shuffle=True, random_state=42)

# stacked_features_train, stacked_features_test = generate_meta_features([
#     LinearRegression(),
#     RandomForestRegressor(n_estimators=1000, max_features='auto', max_depth=10, min_samples_leaf=5,
#                           min_samples_split=2, bootstrap=True, random_state=42)],
#     X_train, X_test, y_train, cv)

# clf_stack = LinearRegression()
# clf_stack.fit(stacked_features_train, y_train)
# y_pred_stack = clf_stack.predict(stacked_features_test)
# print('MAPE for stacking:', mape(y_test, y_pred_stack), '%')

# ==============================================================================
# Predicting for kaggle test sample

data_kaggle = test_data_for_model(data_test)

kaggle_pred = rfr.predict(data_kaggle)

# kaggle_pred = np.reshape(kaggle_pred, (-1, 1))

data_submission['price'] = kaggle_pred
data_submission['price'] = data_submission['price'].apply(lambda x: int(np.exp(x) * 74))

data_submission.to_csv('data_submission.csv', index=False)
