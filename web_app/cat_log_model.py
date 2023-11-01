# Импортируем необходимые для работы библиотеки
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pickle

# Подгружаем данные, логарифмируем числовые признаки
data = pd.read_csv('data_cleaned.csv')
for column in ['sqft', 'school_rating', 'beds_area', 'baths_area', 'lotsize_sqft']:
    data[column] = np.log(data[column])
data[column] = np.log(data['school_min_distance']+1)

# Cоставим список категориальных признаков
cat_list = ['status', 'zipcode', 'state', 'property_add_type']

# Разбиваем данные на train и test
X = data.drop(['target'], axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Инициируем модель
cb_log_model = CatBoostRegressor(iterations=300, 
                                 learning_rate=0.25, 
                                 depth=11, 
                                 random_seed=42, 
                                 silent=True)

# Обучаем модель
cb_log_model.fit(X_train, np.log(y_train), cat_features=cat_list)

# Сериализуем модель и сохраняем в файл
pickle.dump(cb_log_model, open("cat_log_model.pkl", "wb"))