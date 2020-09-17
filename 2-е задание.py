import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



# Настройка вывода таблиц окна run
desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',30)

# чтение данных
x_data = pd.read_csv(r'C:\Users\pc\PycharmProjects\Северсталь_задание\X_data.csv',sep=';')
y_submit = pd.read_csv(r'C:\Users\pc\PycharmProjects\Северсталь_задание\Y_submit.csv',sep=';')
y_train = pd.read_csv(r'C:\Users\pc\PycharmProjects\Северсталь_задание\Y_train.csv',sep=';')


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#матрица корреляций
corr = x_data.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()


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

# нахожу название колонн они же совпадает с первым значением
time_comlumn_name_y_train = list(y_train)[0]
time_comlumn_name_y_submit = list(y_submit)[0]

pred_val_comlumn_name_y_train = list(y_train)[1]
predict_value_comlumn_name_y_submit = list(y_submit)[1]


# значения для предсказаний
predict_y_train_data = y_train[pred_val_comlumn_name_y_train]


# значения времени забора данных
time_y_submit_data = y_submit[time_comlumn_name_y_submit]


# перевожу в нужный тип данных название колонн
predict_value_comlumn_name_y_train = int(pred_val_comlumn_name_y_train)
first_predict_value_comlumn_name_y_submit = int(predict_value_comlumn_name_y_submit)
time_comlumn_name_y_submit = str(time_comlumn_name_y_submit)


# нахожу название первого столбца в x_data со значениями времени сбора оценки качества продукции
time_column_x_data = list(x_data)[0]


# создаю series объект с тем же столбцом что и в y_train с  временем забора оценки качества продукции со всеми данными из x_data
x_data_time_column_series = x_data[time_column_x_data]


# Получаю значение первого индекса совпадения по времени
first_index_match_y_test_in_x_test = list(x_data_time_column_series).index(time_comlumn_name_y_train)
first_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(time_comlumn_name_y_submit)


# так как в y_train y_submit меньше в размере сэмплов, чем x_train то необходимо ограничить x_train по ним, чтобы выборки
#совпалдали

# создаю объект series со значениями времени сбора оценки качества продукции
time_column_y_train_series = y_train[time_comlumn_name_y_train]
pred_val_column_y_train_series = y_train[pred_val_comlumn_name_y_train]
pred_val_column_y_test_series = y_submit[predict_value_comlumn_name_y_submit]
time_column_y_submit_series = y_submit[time_comlumn_name_y_submit]


# индексы всех значений
all_indices_y_test = pd.Index(pred_val_column_y_test_series)
all_indices_y_train = pd.Index(time_column_y_train_series)
all_indices_pred_val_y_train = pd.Index(pred_val_column_y_train_series)
all_indices_y_submit = pd.Index(time_column_y_submit_series)


# последние значение
last_prep_value_y_test = all_indices_y_test[-1]
last_pred_value_column_y_train = all_indices_pred_val_y_train[-1]

last_value_time_column_y_train = all_indices_y_train[-1]
last_value_time_column_y_submit = all_indices_y_submit[-1]


#
predict_value_comlumn_name_y_train_str = str(predict_value_comlumn_name_y_train)


# создаём датафреймы
last_prep_value_y_test_df = pd.DataFrame(data=[last_prep_value_y_test],columns=[predict_value_comlumn_name_y_submit])
last_value_time_column_y_submit_df = pd.DataFrame(data=[last_value_time_column_y_submit],columns=[time_comlumn_name_y_submit])
last_pred_value_column_y_train_df = pd.DataFrame(data=[last_pred_value_column_y_train],columns=[predict_value_comlumn_name_y_train_str])


# получаю последние значение совпадения
last_index_match_y_test_in_x_test = list(x_data_time_column_series).index(last_value_time_column_y_train)
last_index_match_y_submit_in_x_test = list(x_data_time_column_series).index(last_value_time_column_y_submit)

# длина недостоющих данных
last_values_len_y_submit_in_x_data= x_data[last_index_match_y_submit_in_x_test::].shape[0]


# получаю список названий колонн в x_train без первой колонны
list_column_names_x_train_without_first = list(x_data)[1:]


# получаю slice из значений с периодом времени тем же что и в y_train,y_submit
x_train = x_data[first_index_match_y_test_in_x_test:last_index_match_y_test_in_x_test:]
x_test = x_data[first_index_match_y_submit_in_x_test:last_index_match_y_submit_in_x_test:]


# убираю колонну со временем взятия пробы
x_data_without_first_column = x_data[list_column_names_x_train_without_first]
x_train_finish = x_train[list_column_names_x_train_without_first]
x_test_finish = x_test[list_column_names_x_train_without_first]


# создаю объект DataFrame
x_train_finish = pd.DataFrame(x_train_finish)
x_test_finish = pd.DataFrame(x_test_finish)


# без первого значения
y_train_without_first_value = pd.DataFrame(data=predict_y_train_data)
y_test_without_first_value = pd.DataFrame(data=pred_val_column_y_test_series)
y_submit_without_first_time_value = pd.DataFrame(data=time_y_submit_data)


# сдвигаю на одно значение вниз
y_test_shifted = y_test_without_first_value.shift(1)
y_train_without_first_value_shifted = y_train_without_first_value.shift(1)
y_submit_without_first_time_value_shifted = y_submit_without_first_time_value.shift(1)


# добавляю недостоющее первое значение
y_train_without_first_value_shifted.loc[0] = [predict_value_comlumn_name_y_train]
y_test_shifted.loc[0] = [first_predict_value_comlumn_name_y_submit]
y_submit_without_first_time_value_shifted.loc[0] = [time_comlumn_name_y_submit]


# добавляю последние значение
y_test_full = y_test_shifted.append(last_prep_value_y_test_df,ignore_index=True)
y_submit_finish_time_values = y_submit_without_first_time_value_shifted
y_submit_finish_time_values = y_submit_finish_time_values.append(last_value_time_column_y_submit_df,ignore_index=True)


#
Y = y_train_without_first_value_shifted

Y_test = y_test_full


# сумма значений длины последних строк
len_last_values = last_index_match_y_submit_in_x_test+last_values_len_y_submit_in_x_data
x_test_last_val = x_data_without_first_column[last_index_match_y_submit_in_x_test:len_last_values:]
df_train = pd.DataFrame(columns=list_column_names_x_train_without_first)
step = 60

# берём среднее у каждых 60 строк и записываем
for i in range(0,x_train_finish.shape[0],step):
        j = i+(step-1)
        df2_train = x_train_finish[i:j]
        df3_train = df2_train.mean(axis=0)
        df_train = df_train.append(df3_train,ignore_index=True)

X = df_train


df_test = pd.DataFrame(columns=list_column_names_x_train_without_first)

# берём среднее у каждых 60 строк и записываем
for i in range(0,x_test_finish.shape[0],step):
        j = i+(step-1)
        df2_test = x_test_finish[i:j]
        df3_test = df2_test.mean(axis=0)
        df_test = df_test.append(df3_test,ignore_index=True)


step = last_values_len_y_submit_in_x_data
# Проходимся по остаткам
for i in range(0,x_test_last_val.shape[0],step):
        j = i+(step)
        df5_test = x_test_last_val[i:j]
        df6_test = df5_test.mean(axis=0)
        df_test = df_test.append(df6_test, ignore_index=True)

X_test = df_test

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.01,random_state=42)
lr = LinearRegression(n_jobs=-1)
lr.fit(x_train,y_train)
y_pred = lr.predict(X_test)
mae = mean_absolute_error
mae = mae(Y_test,y_pred)
print('MAE = {}'.format(mae))


y_submit_new_time_column_name = str(list(y_submit_finish_time_values)[0])
y_submit_time_values = pd.to_datetime(y_submit_finish_time_values[y_submit_new_time_column_name])
df = y_submit_time_values.dt.strftime('%d.%m.%Y %H:%M')
with open('submit.txt', 'w') as dst:
    dst.write('')
    for i, p in zip(y_pred, df):
        dst.write('%s,%s\n' % (p, float(i)))


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
lr = LinearRegression(n_jobs=-1)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
mae = mean_absolute_error
mae = mae(y_test,y_pred)
print('MAE = {}'.format(mae))





x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
scaler.fit(y_train)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)
y_train = y_train.ravel()

from sklearn.model_selection import GridSearchCV
alfa = [0.0001, 0.001, 0.01, 0.1]
grid_param = {  'penalty' : ['l1','l2','elasticnet'],
                'alpha' : alfa,
                'learning_rate':  ['constant'],
                'eta0' : [0.1,0.01,0.001],
}
gs =GridSearchCV(SGDRegressor(),param_grid=grid_param,n_jobs=-1,cv=5,)
gs.fit(x_train,y_train)
y_pred = gs.predict(x_test)
bp = gs.best_params_
print(bp)


best_paran_ssc = {'alpha': 0.1, 'eta0': 0.001, 'learning_rate': 'constant', 'penalty': 'elasticnet'}



x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

reg = make_pipeline(StandardScaler(),SGDRegressor(alpha= 0.1, eta0= 0.001, learning_rate= 'constant', penalty= 'l1',max_iter= 100000))
reg.fit(x_train, y_train.values.ravel())

y_pred = reg.predict(x_test)



mae = mean_absolute_error
mae = mae(y_test,y_pred)
print('MAE = {}'.format(mae))



