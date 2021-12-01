# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, roc_auc_score, roc_curve
import warnings

warnings.simplefilter('ignore')


# Функция замен дат на категорию месяца
def seasons(date):
    date = date[2:5]
    if date == 'JAN':
        date = 1
    elif date == 'FEB':
        date = 2
    elif date == 'MAR':
        date = 3
    elif date == 'APR':
        date = 4
    return date


# Функция для нахождения межквантильного размаха
def get_iqr_stats(df, column, iqr_descr=False):
    perc25 = df[column].quantile(0.25)
    perc75 = df[column].quantile(0.75)
    IQR = perc75 - perc25
    low = perc25 - 1.5 * IQR
    high = perc75 + 1.5 * IQR
    if iqr_descr:
        print('\n25-й перцентиль: {},'.format(perc25),
              '\n75-й перцентиль: {},'.format(perc75),
              '\nIQR: {}, '.format(IQR),
              '\nГраницы выбросов: [{l}, {h}].\n'.format(l=low, h=high))
    return low, high


# Функция для нахождения выбросов в колонке
def mistakes(df, column):
    low, high = get_iqr_stats(df, column, iqr_descr=False)
    if df[column].min() < low or df[column].max() > high:  # Нахождение выбросов
        df_mistakes = df.loc[
            (~df.loc[:, column].between(low, high)) & pd.notnull(df.loc[:, column])]  # Выделение выбросов
        print('Найдено {} выбросов.'.format(df_mistakes.shape[0]))
        print('\nДатасет выбросов:')
        print(df_mistakes.head())
        return True
    else:
        print('Выбросов не обнаружено.')


# Функция первичного анализа данных
def pre_analysis(df, column, description, for_num_cols=True):
    print('\n', str(column).upper(), sep='')
    if column in list(description.keys()):  # Отображаем описание параметра
        print(description[column])
    if for_num_cols:
        sample = pd.DataFrame({'Распределение выборки': df[column]})
        if mistakes(df, column):  # Если есть выбросы, записываем "чистые" данные в новый DF
            low, high = get_iqr_stats(df, column, iqr_descr=True)
            sample['Распределение выборки в границе выбросов'] = df.loc[df.loc[:, column].between(low, high)][column]
        miss = (1 - (df[column].count() / df.shape[0]))  # Количество пропусков а колонке
        print('Процент пропусков = {} %'.format(round(miss * 100, 2)))
        fig, ax = plt.subplots(figsize=(6 * sample.shape[1], 5))  # Построение гистограммы для "чистых" данных
        sample.hist(ax=ax)
        plt.suptitle('Распределение параметра {} в обучающей выборке'.format(column))
        plt.show()
    else:
        miss = (1 - (df[column].count() / df.shape[0]))  # Количество пропусков а колонке
        print('Процент пропусков = {} %'.format(round(miss * 100, 2)))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(df.loc[:, column], ax=ax)  # Построение графика countplot для колонок с типом object
        plt.title('Распределение категориального признака {} в обучающей выборке'.format(column))
        plt.show()


def calc_and_plot_roc(y_true, y_pred_proba):  # Функция для отрисовки ROC кривой
    # Посчитать значения ROC кривой и значение площади под кривой AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print('\nLogistic Regression ROC AUC for test sample = %0.4f' % roc_auc)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], label='Baseline', linestyle='--')
    plt.plot(fpr, tpr, label='Regression')
    plt.title('Logistic Regression ROC AUC = %0.4f' % roc_auc)
    plt.xlabel('False positive rate (FPR)', fontsize=15)
    plt.ylabel('True positive rate (TPR)', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()


def analysis(data, train_mode=False):
    if train_mode:
        print('\nИнформация о датасете:')
        print(data.info())
        print('\nНачальный датасет:')
        print(data.head)

    # Для удобства выделения признаковы по группам их принадлежности
    bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
    cat_cols = ['education', 'home_address', 'work_address', 'sna', 'first_time']
    num_cols = ['age', 'decline_app_cnt', 'income', 'bki_request_cnt', 'score_bki', 'region_rating']

    # Заменим отдельные даты подачи заявки на категориальный, где 1 - январь, 2 - февраль, 3 - март, 4 - апрель
    data['app_date'] = data['app_date'].apply(seasons)

    # Добавим новый категориальный признак в список других
    cat_cols.append('app_date')

    # Переведём численный показатель количества отказанных прошлых заявок в категориальный,
    # где 0 - 0 отказанных, 1 - 1, 2 - больше 1
    data['decline_app_cnt'] = data['decline_app_cnt'].apply(lambda x: 2 if x > 2 else x)

    # Переместим этот показатель в список других категориальных показателей
    cat_cols.append('decline_app_cnt')
    num_cols.remove('decline_app_cnt')

    # Прологорифмируем чиленные показатели, имеющие правый хвост
    for col in set(num_cols) - {'age', 'score_bki', 'region_rating'}:
        data[col] = data[col].apply(lambda x: np.log(x + 1))

    if train_mode:
        # Первичный анализ для численных параметров
        for col in set(num_cols) - {'region_rating'}:
            pre_analysis(data, col, description)

        # Удалим выбросы в численных показателях, кроме score_bki,
        # так как данный показатель итак имеет нормальное распределение
        data = data[data['income'].between(get_iqr_stats(data, 'income')[0], get_iqr_stats(data, 'income')[1])]
        data = data[data['bki_request_cnt'].between(get_iqr_stats(data, 'bki_request_cnt')[0],
                                                    get_iqr_stats(data, 'bki_request_cnt')[1])]

        # Подсчёт значений f-статистики численных показателей (чем больше, тем показатель лучше)
        imp_num = pd.Series(f_classif(data[num_cols], data['default'])[0], index=num_cols)
        imp_num.sort_values(inplace=True)
        print('\nЗначения f-статистики для численных показателей:\n', imp_num, sep='')

        # Построение теплокарты корреляции числовых признаков с целевым
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data[num_cols + ['default']].corr(), vmin=0, vmax=1, ax=ax, annot=True, fmt=".2f")
        plt.title('График корреляции численных параметров обучающей выборки')
        plt.show()
        print('\nПо данной теплокарте корреляции численных показателей можно заметить, что ни одна пара параметров\n'
              'не имеет высокого показателя корреляции, благодаря чему можно сделать вывод, что ни один из\n'
              'показателей не является "лишним"')

        # Построение графика, отображающего f-статистику численных показателей
        fig, ax = plt.subplots(figsize=(10, 6))
        imp_num.plot(kind='barh', ax=ax)
        plt.title('f-статистика для чиленных показателей')
        plt.show()
        print('По данной статистике можно заметить, что наибольшее влияние на целевой показатель\n'
              'среди численных параметров оказывает параметр score_bki')

        # Первичный анализ категориальных признаков
        for col in bin_cols + cat_cols:
            pre_analysis(data, col, description, for_num_cols=False)

    # Заполнение пропусков самым частовстречающимся значением
    data['education'] = data['education'].fillna(data['education'].value_counts().index[0])

    # Переведём бинарные показатели, записанные словами, в числа
    label_encoder = LabelEncoder()
    for column in bin_cols:
        data[column] = label_encoder.fit_transform(data[column])

    # Перевод категориального признака 'education' в численный
    data['education'] = label_encoder.fit_transform(data['education'])

    if train_mode:
        # Подсчёт значений оценки значимости категориальных и бинарных показателей
        imp_cat = pd.Series(mutual_info_classif(data[bin_cols + cat_cols], data['default'],
                                                discrete_features=True), index=bin_cols + cat_cols)
        imp_cat.sort_values(inplace=True)
        print('\nОценка значимости категориальных и бинарных признаков:\n', imp_cat, sep='')

        # Построение графики, отображающего оценки значимости категорикльных и бинарных признаков
        fig, ax = plt.subplots(figsize=(12, 6))
        imp_cat.plot(kind='barh', ax=ax)
        plt.title('Оценка значимости категориальных и бинарных признаков:')
        plt.show()
        print('По данной оценке можно заметить, что наибольшее влияние на целевой показатель\n'
              'среди категориальных параметров оказывает параметр sna')

    # Создание dummy-переменных при помощи метода OneHotEncoding
    x_cat = OneHotEncoder(sparse=False).fit_transform(data[cat_cols].values)
    print('\nКоличество категориальных признаков после OneHotEncoding:', x_cat.shape[1])

    # Стандартизация числовых переменных
    x_num = StandardScaler().fit_transform(data[num_cols].values)

    # Объединение стандартизированных числовых переменных, бинарных и закодированных категориальных
    x = np.hstack([x_num, data[bin_cols].values, x_cat])

    if train_mode:
        y = data['default'].values

        # Разделение данных на обучающий и тестовые
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20, random_state=50)

        # Обучаем модель
        lr_tm = LogisticRegression()
        lr_tm.fit(x_train, y_train)

        y_pred = lr_tm.predict_proba(x_valid)
        probs = y_pred[:, 1]

        # Построние ROC AUC - кривой
        calc_and_plot_roc(y_valid, probs)

        return x, y
    else:
        return x


data_train = pd.read_csv('train_kaggle.csv')
data_test = pd.read_csv('test.csv')

solution = pd.DataFrame(columns=['client_id', 'default'])
solution['client_id'] = data_test['client_id']

description = {
    'client_id': 'Идентификатор клиента',
    'education': 'Уровень образования',
    'sex': 'Пол заёмщика',
    'age': 'Возраст заёмщика',
    'car': 'Флаг наличия автомобиля',
    'car_type': 'Флаг автомобиля-иномарки',
    'decline_app_cnt': 'Количество отказанных прошлых заявок',
    'good_work': 'Флаг наличия «хорошей» работы',
    'bki_request_cnt': 'Количество запросов в БКИ',
    'home_address': 'Категоризатор домашнего адреса',
    'work_address': 'Категоризатор рабочего адреса',
    'income': 'Доход заёмщика',
    'foreign_passport': 'Наличие загранпаспорта',
    'sna': 'Связь заемщика с клиентами банка',
    'first_time': 'Давность наличия информации о заемщике',
    'score_bki': 'Скоринговый балл по данным из БКИ',
    'region_rating': 'Рейтинг региона',
    'app_date': 'Дата подачи заявки',
    'default': 'Наличие дефолта'
}

X, Y = analysis(data_train, train_mode=True)

# Разделение данных на обучающий и тестовые
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.20, random_state=50)

# Обучаем модель
lr = LogisticRegression()
lr.fit(X_train, y_train)

X_test = analysis(data_test)

y_pred = lr.predict_proba(X_test)
probs = y_pred[:, 1]

# Записываем значения вероятности default для выборки с Kaggle в итоговый датасет
solution['default'] = probs
print(solution)
print()

# Сохраняем итоговый датасет
solution.to_csv('solution.csv', index=False)
