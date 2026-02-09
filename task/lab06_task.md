# Lab 06 — Базовая нейронная сеть и доработка
# Lab 06 — Neural Network Baseline and Tuning

---

##  Описание задания

### Цель работы
Создать базовую нейронную сеть для регрессии, реализовать обучение, а затем провести доработку модели через подбор гиперпараметров для улучшения качества.

### Задачи
1. Провести анализ данных (EDA).
2. Предобработать данные (масштабирование, нормализация, кодирование категориальных признаков).
3. Построить baseline-модель нейронной сети:
   - выбрать количество скрытых слоёв и нейронов,
   - выбрать функции активации для скрытых и выходного слоёв,
   - реализовать обучение модели и выбрать метрику оценки.
4. Провести подбор гиперпараметров (grid search) для:
   - функции активации,
   - Dropout (наличие и значение),
   - Batch Normalization (наличие),
   - размера батча.
   Архитектура слоёв оставляется как в baseline.
5. Визуализировать зависимость метрики RMSE от комбинаций параметров.
6. Сравнить результаты разных комбинаций и выбрать лучшую модель.
7. Сделать выводы по эффективности подходов.

### Требования к результату
- EDA с комментариями.
- Графики зависимости RMSE от гиперпараметров.
- Обоснование выбора лучшей модели.
- Выводы по результатам работы.

---

##  Task description

### Goal
Build a baseline neural network for regression, implement training, and tune hyperparameters to improve model performance.

### Tasks
1. Perform exploratory data analysis (EDA).
2. Preprocess data (scaling, normalization, encode categorical features).
3. Build a baseline neural network:
   - select number of hidden layers and neurons,
   - choose activation functions for hidden and output layers,
   - implement model training and select an evaluation metric.
4. Perform hyperparameter tuning (grid search) for:
   - activation function,
   - Dropout (presence and value),
   - Batch Normalization (presence),
   - batch size.
   Keep layer architecture same as baseline.
5. Visualize RMSE dependence on hyperparameter combinations.
6. Compare different combinations and select the best model.
7. Draw conclusions on the effectiveness of different approaches.

### Expected results
- EDA with explanations.
- RMSE vs. hyperparameters plots.
- Justification of the best model selection.
- Clear conclusions based on results.
