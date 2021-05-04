# Испорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
import warnings
warnings.simplefilter('ignore')

pd.set_option('display.max_rows', 50)  # показывать больше строк
pd.set_option('display.max_columns', 50)  # показывать больше колонок

puple = pd.read_csv('stud_math.csv')  # загрузка датасета


# Функция для нахождения межквантильного размаха
def get_iqr_stats(df, column, iqr_descr=True):
    perc25 = df[column].quantile(0.25)
    perc75 = df[column].quantile(0.75)
    IQR = perc75 - perc25
    low = perc25 - 1.5 * IQR
    high = perc75 + 1.5 * IQR
    if iqr_descr:
        print('25-й перцентиль: {},'.format(perc25),
          '\n75-й перцентиль: {},'.format(perc75),
          '\nIQR: {}, '.format(IQR),
          '\nГраницы выбросов: [{l}, {h}].\n'.format(l=low, h=high))
    return low, high


# Функция для нахождения выбросов в колонке
def mistakes(df, column, with_borders):
    if column not in list(with_borders.keys()):  # Проверка на наличие известных границ выбросов
        low, high = get_iqr_stats(df, column, iqr_descr=False)
        if df[column].min() < low or df[column].max() > high:  # Нахождение выбросов
            df_mistakes = df.loc[(~df.loc[:, column].between(low, high)) & pd.notnull(df.loc[:, column])]  # Выделение выбросов
            print('Найдено {} выбросов.'.format(df_mistakes.shape[0]))
            print(df_mistakes.head())
            return True
        else:
            print('Выбросов не обнаружено.')
    else:
        low, high = with_borders[column][0], with_borders[column][1]
        if df[column].min() < low or df[column].max() > high:
            df_mistakes = df.loc[(~df.loc[:, column].between(low, high)) & pd.notnull(df.loc[:, column])]
            print('Найдено {} выбросов.'.format(df_mistakes.shape[0]))
            print(df_mistakes.head())
            return True
        else:
            print('Выбросов не обнаружено.')


# Функция для первичного анализа данных
def pre_analysis(df, column, description, with_borders):
    print('\n', str(column).upper(), sep='')
    if column in list(description.keys()):  # Отображаем описание параметра
        print(description[column])
    if df.loc[:, column].dtypes == np.dtype('O'):  # Проверка на тип данных object
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(df.loc[:, column], ax=ax)  # Построение графика countplot для колонок с типом object
        plt.show()
        miss = (1 - (df[column].count() / df.shape[0]))  # Количество пропусков а колонке
        print('Процент пропусков = {} %'.format(round(miss * 100, 2)))
    else:
        sample = pd.DataFrame({'Распределение выборки': df[column]})
        if mistakes(df, column, with_borders):  # Если есть выбросы, записываем "чистые" данные в новый DF
            low, high = get_iqr_stats(df, column, iqr_descr=True)
            sample['Распределение выборки в границе выбросов'] = df.loc[df.loc[:, column].between(low, high)][column]
        fig, ax = plt.subplots(figsize=(6 * sample.shape[1], 5))  # Построение гистограммы для "чистых" данных
        sample.hist(ax=ax)
        plt.show()
        miss = (1 - (df[column].count() / df.shape[0]))  # Количество пропусков а колонке
        print('Процент пропусков = {} %'.format(round(miss * 100, 2)))


# Функция для построения графиков box-plot
def get_boxplot(df, column):
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(x=column, y='score',
                data=df.loc[df.loc[:, column].isin(df.loc[:, column].value_counts().index[:10])], ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# Функция для проведения теста Стьюдента
def get_stat_dif(df, column):
    cols = df.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'score'].dropna(),
                     df.loc[df.loc[:, column] == comb[1], 'score'].dropna()).pvalue <= 0.075/len(combinations_all):
            print('Найдены статистически значимые различия для колонки:', column)
            break


# Функция для заполнения пропусков
def passes_replacement(df, column):
    if df[column].dtypes == np.dtype('O'):
        mode = df[column].mode()
        return df[column].fillna(mode)
    else:
        median = df[column].median()
        return df[column].fillna(median)


# Словарь с описанием параметров
description = {
    'school': "Аббревиатура школы, в которой учится ученик",
    'sex': "Пол ученика ('F' - женский, 'M' - мужской)",
    'age': "Возраст ученика (от 15 до 22)",
    'address': "Тип адреса ученика ('U' - городской, 'R' - за городом)",
    'famsize': "Размер семьи('LE3' <= 3, 'GT3' >3)",
    'Pstatus': "Статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)",
    'Medu': "Образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)",
    'Fedu': "Образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)",
    'Mjob': "Работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)",
    'Fjob': "Работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)",
    'reason': "Причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)",
    'guardian': "Опекун ('mother' - мать, 'father' - отец, 'other' - другое)",
    'traveltime': "Время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)",
    'studytime': "Время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)",
    'failures': "Количество внеучебных неудач (n, если 1<=n<=3, иначе 0)",
    'schoolsup': "Дополнительная образовательная поддержка (yes или no)",
    'famsup': "Семейная образовательная поддержка (yes или no)",
    'paid': "Дополнительные платные занятия по математике (yes или no)",
    'activities': "Дополнительные внеучебные занятия (yes или no)",
    'nursery': "Посещал детский сад (yes или no)",
    'higher': "Хочет получить высшее образование (yes или no)",
    'internet': "Наличие интернета дома (yes или no)",
    'romantic': "В романтических отношениях (yes или no)",
    'famrel': "Семейные отношения (от 1 - очень плохо до 5 - очень хорошо)",
    'freetime': "Свободное время после школы (от 1 - очень мало до 5 - очень мого)",
    'goout': "Проведение времени с друзьями (от 1 - очень мало до 5 - очень много)",
    'health': "Текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)",
    'absences': "Количество пропущенных занятий",
    'score': "Баллы по госэкзамену по математике"
}

# Словарь с границами некоторых параметров
with_borders = {
    'age': [15, 22],
    'Medu': [0, 4],
    'Fedu': [0, 4],
    'traveltime': [1, 4],
    'studytime': [1, 4],
    'failures': [0, 3],
    'famrel': [1, 5],
    'freetime': [1, 5],
    'goout': [1, 5],
    'health': [1, 5],
    'score': [0, 100]
}

# Выведём начальные данные
print('Изначальная информация:')
print(puple.head(10))
print('Информация о датасете:')
print(puple.info())

# Расчитаем количество пропусков в каждой строке
nan_rows_count = puple.apply(lambda x: sum(x.isnull()), axis=1).value_counts()  # Количество строк с разными кол-вом пропусков
nan_rows_percent = round(pd.Series(nan_rows_count.index / puple.shape[1] * 100)).sort_values(ascending=False).astype(str) + ' %'
print(pd.DataFrame({'Случаи пропусков': nan_rows_count, 'Количество пропусков': nan_rows_count.index,
              'Процент пропусков': nan_rows_percent}).sort_values('Количество пропусков', ascending=False))

for col in puple.columns:
    pre_analysis(puple, col, description, with_borders)

print('\nПо предворительному анализу, можно заметить:\n'
      '1. Имеется разрыв в возростах учеников - 16 лет, также имеются несколько вылековозрастных учеников.\n'
      'Возможное решение: разделить учеников на 2 группы: до 17 и старше 17.\n'
      '2. Имеется выброс в параметре "Fedu" - 40.\n'
      'Можно предположить, что это опечатка и заменить 40 на 4.\n'
      '3. Имеется выброс в показателе "famrel" - -1.\n'
      'Можно предположить, что это опечатка и заменить на 1.\n'
      '4. Показатель "absences" имеет 17 выбросов.\n'
      'Возможное решение: сильно выбивающиеся значения (> 100) заменить на медиану.\n'
      '5. Распределение показателей "studytime" и "studytime, granular" имеют схожий характер.\n'
      '6. Показатель "studytime, granular" имеет выбросы.\n'
      '7. В переменных "school", "Pstatus", "higher", "famrel" имеет дисбаланс классов.\n'
      '8. Целевая переменная "Score" имеет 1.52 % пропусков.\n'
      '9. Также по графику распределения целевой переменной видно, что есть пробел между 0 и 20.\n'
      'Возможное объяснение: существует пороговое значение = 20, меньше которого, результаты приравниваются к 0.\n')

# Произведём предобработку на основе сделанных выводов
puple['age_group'] = puple.age.apply(lambda x: 'old' if x > 17 else 'young')
puple.Fedu = puple.Fedu.replace(40, 4)
puple.famrel = puple.famrel.replace(-1, 1)
puple.absences = puple.absences.apply(lambda x: puple.absences.median() if x > 100 else x)
puple.dropna(subset=['score'], inplace=True)
# dropna() - функция для удаление строк с пропусками в колонке "Score"

# Корреляционный анализ
print('Как мы знаем, датасет имеет 13 числовых колонок, но "истинных" количественных только 3:'
      'age, absences, score\n'
      'Произведём корреляционный анализ для этих количественных признаков:\n')

sns.pairplot(puple[['age', 'absences', 'score']], kind='reg')
plt.show()

print(puple[['age', 'absences', 'score']].corr())
print('\nКак мы видим, зависимость между показтелем "Score" и "Absences" очень низкая,'
      'в связи с чем, показатель "absences" можно исключить из будущей модели.')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(puple.corr().round(2), annot=True, mask=np.triu(puple.corr()), ax=ax)
# np.triu() - функция для отображения значений под главной диагональю
plt.show()

print('\nПо данной тепловой карте видно:\n'
      '1. Показатели "studytime" и "studytime, granular" полностью обратно коррелируют\n'
      'Таким образом, показатель "studytime, granular" можно удалить, т.к он не несёт никакой информации.\n'
      '2. Показатели "Fedu" и "Medu" имеют довольно большой показателй корреляции.\n'
      'Таким образом, на их основе можно сгенерировать новый показатель для более точного описания.\n'
      'Заполнить пропуски этих столбцов на основе их линейной зависимости.')

# Создадим отдельный DF для значений коррелиции относительно показателя Score
print('\nСоздадим отдельный DF для значений коррелиции относительно показателя "Score":')
score_corr = pd.DataFrame(puple.corr().score.values,
                          index=puple.corr().score.index,
                          columns=['correlation'])
score_corr = score_corr.drop('score')
score_corr['abs_correlation'] = abs(score_corr.correlation)
print(score_corr.sort_values(by='abs_correlation', ascending=False))
print('\nПо данной таблице можно сделать вывод, что наиболее отрицательно на успеваемость влияют показатели:\n'
      '"failure" - неудачи во внеурочное время, "age" - возраст, "goout" - активное общение с друзьями.\n'
      'Положительно влияют: оброзование обоих родителей и самообучение.\n'
      'В качестви чистки основго DF, удалим столбцы значение корреляции которых со "Score" меньше 0.1\n')

puple.drop('studytime, granular', axis=1, inplace=True)
puple.drop(score_corr[score_corr.abs_correlation < 0.1].index, axis=1, inplace=True)
puple['Parents_edu'] = puple.Medu + puple.Fedu

# Построим графики box-plot для оценки номинативных признаков
for col in list(set(puple.columns) - {'age', 'absences', 'score'}):
    get_boxplot(puple, col)

print('\nПо данным графикам box-plot, можно сделать вывод, что на показатель Score имют влияние следующие показатели:\n'
      'school, schoolsup, Parents_edu, Medu, Fedu, Mjob, higher, studytime, age_group, address, failures, goout\n'
      'Также можно заметить, что показатель Fedu = teacher сильно выделяется среди остальных\n'
      'Можно сделать отдельный столбец - F_teacher, который будет отвечать за данный показатель')

puple['F_teacher'] = puple.Fjob.apply(lambda x: 1 if x == 'teacher' else 0)

# Произведём анализ номинативных признаков с помощью теста Стьюдента
print('\nПроизведём анализ номинативных признаков с помощью теста Стьюдента:\n')

for col in list(set(puple.columns) - {'age', 'absences', 'score'}):
    get_stat_dif(puple, col)
print('\nПо данному анализу значимыми параметрами для построения дальнейшей модели стали:\n'
      'Parents_edu, Medu, Mjob, F_teacher, higher, age_group, address, failures, goout, sex, romantic, paid')

print('\nТак как графики box-plot и анализ Стьюдента - взаимодополняющие методы, то необходимо брать параметры, которые'
      ' получилось обоими методами')

print('\nИтоговый датасет для построения модели:')
model_columns = ['school', 'schoolsup', 'Medu', 'Fedu', 'Parents_edu', 'Mjob', 'F_teacher', 'higher', 'studytime',
                 'age_group', 'address', 'failures', 'goout', 'sex', 'romantic', 'paid']
model_puple = puple[model_columns]
print(model_puple.head())

# Заполнение пропусков
print('\nСначала зполним значения показателей "Fedu" и "Medu" на основе их линейной зависимости:')

# Если значение 0, заменяем на значение другого показателя в этой строке, иначе не трогаем
puple.Fedu = np.where(puple.Fedu.isna(), puple.Medu, puple.Fedu)
puple.Medu = np.where(puple.Medu.isna(), puple.Fedu, puple.Medu)

print('\nДалее заполним количественные показатели на медианное значение, а номинативные на моду.')
for col in puple:
    puple[col] = passes_replacement(puple, col)

print('\nИтоговый вывод:\n\n'
      'В результате проведения EDA анализа, было выяснено:\n\n'
      '1. Данные достаточно чистые: количество пропусков в каждом столбце не превышает 15 %.\n'
      'Были обнаружены ошибки в столбцах "Fedu" и "famrel" и заменены на предпложения,'
      'полученные исходят из здарового смысла.\n'
      'Переменная "absences" содержала пару аномальных значений (> 100), которые были заменены на моду.\n\n'
      '2. Было решено с помощью колонки "age" создать новый признак - "age_group", который включает в себя 2 параметра\n'
      '"old" (> 17) и "young" (< 18), объединяющие студентов в разные возрастные группы.\n'
      'Из целевой переменной "Score" были удалены пропуски.\n\n'
      '3. В результате корреляционного анализа:\n'
      'Была обнаружена полная обратная зависимость между показателями "studytime" и "studytime, granular",\n'
      'поэтому один из них был удалён, так как не нёс никакой информации.\n'
      'Тажке была замечена достаточно сильная линейная зависимость между показетялми "Fedu" и "Medu",\n'
      'благодаря которым был составлен новый показетль - "Parents_edu", а также заполнены пропуски в этих показтелях.\n'
      'Были исключены параметры с показателем корреляции относительно целевой переменной < 0.1.\n\n'
      '4. В результате анализа номинативных переменных с помощью графиков box-plot и теста Стьюдента были выделены следующие признаки:\n'
      'school, schoolsup, Medu, Fedu, Parents_edu, Mjob, F_teacher, higher, studytime, age_group, address, failures, goout, sex, romantic, paid.\n\n'
      'Итоговый ответ:\n'
      'Так как в итговой модели оказались показатели, которые были сгенерированы из начальных критериев: Parents_edu, F_teache, age_group,\n'
      'то итоговым ответом будет: school, schoolsup, Medu, Fedu, Mjob, higher, studytime, address, failures, goout, sex, romantic, paid, age')
