import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Настройка вывода таблиц окна run
# desired_width=320
# pd.set_option('display.width', desired_width)
# pd.set_option('display.max_columns',30)
display_settings = {
    'max_columns': 20,
    'expand_frame_repr': True,  # Развернуть на несколько страниц
    'max_rows': 61,
    'precision': 2,
    'show_dimensions': True
}

for op, value in display_settings.items():
    pd.set_option("display.{}".format(op), value)


x_data = pd.read_csv(r'C:\Users\pc\PycharmProjects\Северсталь_задание\X_data.csv',sep=';')

y_submit = pd.read_csv(r'C:\Users\pc\PycharmProjects\Северсталь_задание\Y_submit.csv',sep=';')
y_train = pd.read_csv(r'C:\Users\pc\PycharmProjects\Северсталь_задание\Y_train.csv',sep=';')

# Проверяю на None значения
for i in x_data.columns:
    for i in x_data[i]:
        if pd.isna(i):
            print(i,'x_data None')

for i in y_train.columns:
    for i in y_train[i]:
        if pd.isna(i):
            print(i,'y_train None')

for i in y_submit.columns:
    for i in y_submit[i]:
        if pd.isna(i):
            print(i,'y_submit None')
print('Проверка на None закончена')


# так как размеры массивов разные надо выбрать максимально возможную их общие части для обучения модели

# нахожу название колонны со времени забора оценки качества продукции в y_train оно же совпадает с первым значением
time_comlumn_name_y_train = list(y_train)[0]

time_comlumn_name_y_submit = list(y_submit)[0]

# нахожу название колонны для предсказаний на x_train
predict_value_comlumn_name_y_train = list(y_train)[1]

predict_value_comlumn_name_y_submit = list(y_submit)[1]


# значения для предсказаний
predict_y_train_data = y_train[predict_value_comlumn_name_y_train]

predict_y_submit_data = y_submit[predict_value_comlumn_name_y_submit]

# перевожу в целочисленный тип данных название колонны
predict_value_comlumn_name_y_train = int(predict_value_comlumn_name_y_train)

predict_value_comlumn_name_y_submit = int(predict_value_comlumn_name_y_submit)


# нахожу название первого столбца в x_data со значениями времени сбора оценки качества продукции
time_column_x_data = list(x_data)[0]


# создаю series объект со столбцом со схожими данными о времени забора оценки качества продукции, что и в y_train
x_data_time_column_series = x_data[time_column_x_data]

# Получаю значение индекса первого совпадения времени y_train с x_train значением
first_index_match_y_test_in_x_test = list(x_data_time_column_series).index(time_comlumn_name_y_train)

first_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(time_comlumn_name_y_submit)


# так как y_train меньше в размере сэмплов, чем x_train то необходимо ограничить x_train по нему, чтобы выборки
#совпалдали

# создаю объект series со значениями времени сбора оценки качества продукции в y_train
time_column_y_train_series = y_train[time_comlumn_name_y_train]

time_column_y_submit_series = y_submit[time_comlumn_name_y_submit]


# индексы всех значений в y_train
all_indices_y_train = pd.Index(time_column_y_train_series)

all_indices_y_submit = pd.Index(time_column_y_submit_series)

# последние значение в y_train
last_value_time_column_y_train = all_indices_y_train[-1]

last_value_time_column_y_submit = all_indices_y_submit[-1]

# получаю последние значение совпадения y_train с x_train
last_index_match_y_test_in_x_test = list(x_data_time_column_series).index(last_value_time_column_y_train)

last_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(last_value_time_column_y_submit)

# получаю список названий колонн в x_train без первой колонны
list_column_names_x_train_without_first = list(x_data)[1:]

# получаю slice из значений с периодом времени тем же что и в y_train
x_train = x_data[first_index_match_y_test_in_x_test:last_index_match_y_test_in_x_test:]

x_test = x_data[first_index_match_y_submit_in_x_test:last_index_match_y_submit_in_x_test:]

# убираю колонну со временем взятия пробы
x_train_finish = x_train[list_column_names_x_train_without_first]

x_test_finish = x_test[list_column_names_x_train_without_first]

# создаю объект DataFrame
x_train_finish = pd.DataFrame(x_train_finish)

x_test_finish = pd.DataFrame(x_test_finish)

# y_train без первого значения
y_train_without_first_value = pd.DataFrame(data=predict_y_train_data)

y_test_without_first_value = pd.DataFrame(data=predict_y_submit_data)

# сдвигаю на одно значение
y_train_without_first_value_shifted = y_train_without_first_value.shift(1)

y_test_without_first_value_shifted = y_test_without_first_value.shift(1)

# добавляю недостоющее первое значение
y_train_without_first_value_shifted.loc[0] = [predict_value_comlumn_name_y_train]

y_test_without_first_value_shifted.loc[0] = [predict_value_comlumn_name_y_submit]


y_train_finish = y_train_without_first_value

y_test_finish = y_test_without_first_value

# переименовываю название колонны
y_train_finish.columns = ['predict_value']

y_test_finish.columns = ['predict_value']


step = 60
df_train = pd.DataFrame(columns=list_column_names_x_train_without_first)

for i in range(-1,x_train_finish.shape[0]+1,step):
        if (i+step) < x_train_finish.shape[0]:
            i=i+ 1
            j = i+(step-1)
            df2_train = x_train_finish[i:j]
            df3_train = df2_train.mean(axis=0)
            df_train = df_train.append(df3_train,ignore_index=True)

x_train_finish = df_train


df_test = pd.DataFrame(columns=list_column_names_x_train_without_first)

for i in range(-1,x_test_finish.shape[0]+1,step):
        if (i+step) < x_test_finish.shape[0]:
            i=i+ 1
            j = i+(step-1)
            df2_test = x_test_finish[i:j]
            df3_test = df2_test.mean(axis=0)
            df_test = df_test.append(df3_test,ignore_index=True)

x_test_finish = df_test
x_train, x_test, y_train, y_test = train_test_split(x_train_finish,y_train_finish,test_size=0.01,random_state=42)
lr = LinearRegression(n_jobs=-1)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test_finish)
mae = mean_absolute_error
print(mae(y_true=y_test_finish,y_pred=y_pred))