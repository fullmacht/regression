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
# print(x_data)
# print(x_data.shape)
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
pred_val_comlumn_name_y_train = list(y_train)[1]

time_comlumn_name_y_submit = list(y_submit)[0]

# нахожу название колонны для предсказаний на x_train


predict_value_comlumn_name_y_submit = list(y_submit)[1]
# predict_value_comlumn_name_y_test = list(y_train)[1]
# print(predict_value_comlumn_name_y_test)

# нахожу название колонны времени забора данных для записи результата модели и добавления первым значением
time_value_comlumn_name_y_submit = list(y_submit)[0]
# print(time_value_comlumn_name_y_submit)
# print(y_submit.shape,'y_submit')

# значения для предсказаний
predict_y_train_data = y_train[pred_val_comlumn_name_y_train]

predict_y_submit_data = y_submit[predict_value_comlumn_name_y_submit]

#значения времени забора данных
time_y_submit_data = y_submit[time_value_comlumn_name_y_submit]

# перевожу в целочисленный тип данных название колонны
predict_value_comlumn_name_y_train = int(pred_val_comlumn_name_y_train)

predict_value_comlumn_name_y_submit = int(predict_value_comlumn_name_y_submit)

time_value_comlumn_name_y_submit = str(time_value_comlumn_name_y_submit)



# нахожу название первого столбца в x_data со значениями времени сбора оценки качества продукции
time_column_x_data = list(x_data)[0]


# создаю series объект со столбцом со схожими данными о времени забора оценки качества продукции, что и в y_train
x_data_time_column_series = x_data[time_column_x_data]

# Получаю значение индекса первого совпадения времени y_train с x_train значением
first_index_match_y_test_in_x_test = list(x_data_time_column_series).index(time_comlumn_name_y_train)

first_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(time_comlumn_name_y_submit)
# print(time_comlumn_name_y_submit)


# так как y_train меньше в размере сэмплов, чем x_train то необходимо ограничить x_train по нему, чтобы выборки
#совпалдали

# создаю объект series со значениями времени сбора оценки качества продукции в y_train
time_column_y_train_series = y_train[time_comlumn_name_y_train]



pred_val_column_y_train_series = y_train[pred_val_comlumn_name_y_train]

time_column_y_submit_series = y_submit[time_comlumn_name_y_submit]

# time_column_y_test_series = y_test[predict_value_comlumn_name_y_test]


# индексы всех значений в y_train
# all_indices_y_test = pd.Index(time_column_y_test_series)
all_indices_y_train = pd.Index(time_column_y_train_series)
all_indices_pred_val_y_train = pd.Index(pred_val_column_y_train_series)
all_indices_y_submit = pd.Index(time_column_y_submit_series)

# последние значение в y_train
# last_value_time_column_y_test = all_indices_y_test[-1]
last_value_time_column_y_train = all_indices_y_train[-1]
last_pred_value_column_y_train = all_indices_pred_val_y_train[-1]
last_value_time_column_y_submit = all_indices_y_submit[-1]
time_comlumn_name_y_submit_str = str(time_comlumn_name_y_submit)
last_value_time_column_y_submit_df = pd.DataFrame(data=[last_value_time_column_y_submit],columns=[time_comlumn_name_y_submit_str])

# pre_last_value_time_column_y_submit = all_indices_y_submit[-2]
# print(last_value_time_column_y_submit)

# получаю последние значение совпадения y_train с x_train
last_index_match_y_test_in_x_test = list(x_data_time_column_series).index(last_value_time_column_y_train)

last_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(last_value_time_column_y_submit)
# pre_last_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(pre_last_value_time_column_y_submit)
# print(last_index_match_y_submit_in_x_test)
last_values_len_y_submit_in_x_data= x_data[last_index_match_y_submit_in_x_test::].shape[0]
# print(x_data[last_index_match_y_submit_in_x_test:last_index_match_y_submit_in_x_test+last_values_len_y_submit_in_x_data:])
# print(x_data[last_index_match_y_submit_in_x_test:last_index_match_y_submit_in_x_test+last_values_len_y_submit_in_x_data:].shape)
# получаю список названий колонн в x_train без первой колонны
list_column_names_x_train_without_first = list(x_data)[1:]

# получаю slice из значений с периодом времени тем же что и в y_train
x_train = x_data[first_index_match_y_test_in_x_test:last_index_match_y_test_in_x_test:]
# print('x_train')
# print(x_train.shape)
# print(x_train)
# print(x_train.shape)
# print(x_data[2103785::])


x_test = x_data[first_index_match_y_submit_in_x_test:last_index_match_y_submit_in_x_test:]
# last_missing_values = x_data[pre_last_index_match_y_submit_in_x_test:last_index_match_y_submit_in_x_test]
# print('x_test')
# print(x_test.shape)
# print(x_test)
# убираю колонну со временем взятия пробы
x_data_without_first_column = x_data[list_column_names_x_train_without_first]
x_train_finish = x_train[list_column_names_x_train_without_first]

x_test_finish = x_test[list_column_names_x_train_without_first]

# создаю объект DataFrame
x_train_finish = pd.DataFrame(x_train_finish)

x_test_finish = pd.DataFrame(x_test_finish)

# y_train без первого значения
y_train_without_first_value = pd.DataFrame(data=predict_y_train_data)

# y_test_without_first_value = pd.DataFrame(data=predict_y_submit_data)

y_submit_without_first_time_value = pd.DataFrame(data=time_y_submit_data)

# сдвигаю на одно значение
y_train_without_first_value_shifted = y_train_without_first_value.shift(1)

# y_test_without_first_value_shifted = y_test_without_first_value.shift(1)
y_submit_without_first_time_value_shifted = y_submit_without_first_time_value.shift(1)

# добавляю недостоющее первое значение
y_train_without_first_value_shifted.loc[0] = [predict_value_comlumn_name_y_train]

# y_test_without_first_value_shifted.loc[0] = [predict_value_comlumn_name_y_submit]

y_submit_without_first_time_value_shifted.loc[0] = [time_value_comlumn_name_y_submit]

print(y_submit_without_first_time_value)
y_submit_finish_time_values = y_submit_without_first_time_value_shifted
y_submit_finish_time_values = y_submit_without_first_time_value_shifted.append(last_value_time_column_y_submit_df,ignore_index=True)
print(y_submit_finish_time_values)

y_train_finish = y_train_without_first_value

# y_test_finish = y_test_without_first_value


# print((last_value_time_column_y_submit))
# print('before')
# print(y_submit_finish_time_values.shape,'y_submit_finish_time_values')
# print(y_test_finish.shape,'y_test_finish')
# print(y_train_finish.shape,'y_train_finish')
# print(x_train_finish.shape,'x_train_finish')

y_train_last_value = pd.Series(last_pred_value_column_y_train)
# y_train_last_value = list(last_pred_value_column_y_train)
# y_test_last_value = last_value_time_column_y_test
# y_submit_last_value = pd.DataFrame(data=[last_value_time_column_y_submit], columns = [time_comlumn_name_y_submit])
#добавляем последние значение в датасеты



# y_train_finish = y_train_finish[str(predict_value_comlumn_name_y_train)].append(y_train_last_value,ignore_index=True)
# y_train_finish = pd.DataFrame(data=y_train_finish,columns=[str(predict_value_comlumn_name_y_train)])
# y_train_finish = y_test_finish.append(y_test_last_value,ignore_index=True)
# y_submit_finish = y_submit_finish_time_values.append(y_submit_last_value,ignore_index=True)
# print('after')
# print(y_train_finish.shape)
# print(y_submit_finish.shape)
# print(last_value_time_column_y_submit,'last_val_y_submit')
# print(last_value_time_column_y_train,'last_val_y_train')
# print('start')
# print(y_train_finish,'_y_train')
# print(y_train_finish,'_y_train')
# print(y_submit_finish,'_y_submit')
# print(y_submit_finish,'_y_submit')

# print(x_data[])
# print(last_missing_values.shape)
# переименовываю название колонны
# print(y_submit_finish_time_values.shape)
# y_train_finish.columns = ['predict_value']

# y_submit_finish_time_values.columns = ['time_value']

# y_test_finish.columns = ['predict_value']
last = last_index_match_y_submit_in_x_test+last_values_len_y_submit_in_x_data
x_test_last_val = x_data_without_first_column[last_index_match_y_submit_in_x_test:last:]
# print(x_test_last_val)
# print(x_test_last_val.shape)
# print(x_test_last_val.shape[0])
# количество минут в часе
step = 60
df_train = pd.DataFrame(columns=list_column_names_x_train_without_first)

for i in range(0,x_train_finish.shape[0],step):
    # if (i+step-1) <= x_train_finish.shape[0]:
        i=i

        j = i+(step-1)
        # if (j) <= x_train_finish.shape[0]:
        df2_train = x_train_finish[i:j]
        df3_train = df2_train.mean(axis=0)
        df_train = df_train.append(df3_train,ignore_index=True)

x_train_finish = df_train

df_test = pd.DataFrame(columns=list_column_names_x_train_without_first)

for i in range(0,x_test_finish.shape[0],step):
    # if (i+step-1) <= x_test_finish.shape[0]:
        i=i
        j = i+(step-1)
        # if (j) <= x_train_finish.shape[0]:
        df2_test = x_test_finish[i:j]
        df3_test = df2_test.mean(axis=0)
        df_test = df_test.append(df3_test,ignore_index=True)


# вычисляем средние по последним 55 значения
# step = -(x_test_finish.shape[0]-last_index_match_y_submit_in_x_test)
# print(x_test_finish)
# print(step)
step = last_values_len_y_submit_in_x_data
# print(x_test_finish[last_index_match_y_submit_in_x_test::].shape)
# stop = x_test_finish[last_index_match_y_submit_in_x_test::].shape[0]+1
# last = last_index_match_y_submit_in_x_test+last_values_len_y_submit_in_x_data
# x_test_last_val = x_train_finish[last_index_match_y_submit_in_x_test:last:]
# print(x_test_last_val.shape)
# print(x_test_last_val.shape[0])
# print(x_test[last_index_match_y_submit_in_x_test::].shape[0])
for i in range(0,x_test_last_val.shape[0],step):
    # if (i+step-1) <= x_test_finish.shape[0]:
        i=i
        j = i+(step-1)
        # if (j) <= x_train_finish.shape[0]:
        df5_test = x_test_last_val[i:j]
        df6_test = df5_test.mean(axis=0)
        df_test = df_test.append(df6_test, ignore_index=True)

x_test_finish = df_test


# x_test_finish =
# print(x_train_finish)
# print(x_test_finish)
# print(y_train_finish)
# print(x_train_finish.shape)
# print(y_train_finish.shape)
# print(x_test_finish.shape)
x_train, x_test, y_train, y_test = train_test_split(x_train_finish,y_train_finish,test_size=0.01,random_state=42)
lr = LinearRegression(n_jobs=-1)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test_finish)
mae = mean_absolute_error
# print(mae(y_true=y_test_finish,y_pred=y_pred))
print(y_pred.shape,'y_pred')

y_submit_new_time_column_name = str(list(y_submit_finish_time_values)[0])
# print(y_submit_new_time_column_name)
y_submit_time_values = pd.to_datetime(y_submit_finish_time_values[y_submit_new_time_column_name])
df = y_submit_time_values.dt.strftime('%d.%m.%Y %H:%M')
# print(df.shape,)
with open('submit.txt', 'w') as dst:
    dst.write('')
    for i, p in zip(y_pred, df):
        dst.write('%s,%s\n' % (p, float(i)))