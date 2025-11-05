# Wine Quality Regression Models Experiment
Этот проект демонстрирует обучение двух моделей регрессии (LinearRegression и RandomForestRegressor) на датасете Wine Quality с использованием GridSearchCV для настройки гиперпараметров. Эксперименты логируются в MLflow для отслеживания, а результаты сравниваются по метрикам MSE и R². Лучшая модель (RandomForestRegressor) интегрирована в Airflow DAG для автоматизации.
---

## Описание моделей и гиперпараметров

![Визуализация эксперементов в mlflow](https://ltdfoto.ru/images/2025/11/05/TREKING-EKSPERIMENTOV.png)

#### LinearRegression
- Описание: Линейная регрессия - простая модель, которая предполагает линейную зависимость между признаками и целевой переменной (качеством вина). Используется StandardScaler для нормализации данных.
![Параметры LinearRegression в mlflow](https://ltdfoto.ru/images/2025/11/05/imagefdaa3560d68332ef.png)
- Гиперпараметры (GridSearchCV):
fit_intercept: [True, False] - Включать ли смещение (intercept) в модель.
- Лучшие параметры (из эксперимента): `{'lr__fit_intercept': True}.`
- Метрики на тесте: MSE = 0.3900, R² = 0.4032.
![Метрики LinearRegression в mlflow](https://ltdfoto.ru/images/2025/11/05/image94fc2ee926053faf.png)

#### RandomForestRegressor
- Описание: Модель случайного леса способна захватывать нелинейные зависимости. Используется StandardScaler для нормализации.
![Параметры RandomForest в mlflow](https://ltdfoto.ru/images/2025/11/05/image5d09db5e7ba6a590.png)
- Гиперпараметры (GridSearchCV):
    - `n_estimators`: [50, 100, 200] - Количество деревьев в лесу.
    -  `max_depth`: [None, 10, 20] - Максимальная глубина деревьев (None без ограничений).
    - `min_samples_split`: [2, 5, 10] - Минимальное количество образцов для разделения узла.
- Лучшие параметры (из эксперимента): `{'rf__max_depth': None, 'rf__min_samples_split': 2, 'rf__n_estimators': 200}`.
- Метрики на тесте: MSE = 0.3063, R² = 0.5312.
![Метрики RandomForest в mlflow](https://ltdfoto.ru/images/2025/11/05/imagec2e491ae557d4d92.png)
## Сравнение результатов

Результаты эксперимента (на основе 5-fold кросс-валидации и тестового набора, random_state=42 для воспроизводимости):

|Model|Best Params|Best CV MSE|Test MSE|Test R²|
|-----|-----------|-----------|--------|-------|
LinearRegression|{'lr__fit_intercept': True}|0.4401|0.3900|0.4032|
RandomForestRegressor|{'rf__max_depth': None, 'rf__min_samples_split': 2, 'rf__n_estimators': 200}|0.3677|0.3063|0.5312|

- MSE (Mean Squared Error): Мера ошибки предсказаний; ниже - лучше.
- R² (Coefficient of Determination): Доля дисперсии, объясненная моделью; выше - лучше.
- Обоснование выбора лучшей модели: RandomForestRegressor показывает лучшие результаты (ниже MSE и выше R²), так как он лучше справляется с нелинейными зависимостями в данных Wine Quality. LinearRegression подходит для простых случаев, но здесь уступает. Лучшая модель логируется в MLflow и интегрирована в DAG.

#### Воспроизводимость: 

Все random_state установлены на 42.
Автологирование MLflow: Захватывает метрики, параметры и модели автоматически.
![Логирование random_state в mlflow](https://ltdfoto.ru/images/2025/11/05/image711f983595d47986.png)