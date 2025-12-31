\# Lab 06 — Базовая нейронная сеть и доработка

\# Lab 06 — Neural Network Baseline and Tuning



---



\## 🇷🇺 Описание задания



\### Цель работы

Создать базовую нейронную сеть для регрессии, реализовать обучение, а затем провести доработку модели через подбор гиперпараметров для улучшения качества.



\### Задачи

1\. Провести анализ данных (EDA).

2\. Предобработать данные (масштабирование, нормализация, кодирование категориальных признаков).

3\. Построить baseline-модель нейронной сети:

&nbsp;  - выбрать количество скрытых слоёв и нейронов,

&nbsp;  - выбрать функции активации для скрытых и выходного слоёв,

&nbsp;  - реализовать обучение модели и выбрать метрику оценки.

4\. Провести подбор гиперпараметров (grid search) для:

&nbsp;  - функции активации,

&nbsp;  - Dropout (наличие и значение),

&nbsp;  - Batch Normalization (наличие),

&nbsp;  - размера батча.

&nbsp;  Архитектура слоёв оставляется как в baseline.

5\. Визуализировать зависимость метрики RMSE от комбинаций параметров.

6\. Сравнить результаты разных комбинаций и выбрать лучшую модель.

7\. Сделать выводы по эффективности подходов.



\### Требования к результату

\- EDA с комментариями.

\- Графики зависимости RMSE от гиперпараметров.

\- Обоснование выбора лучшей модели.

\- Выводы по результатам работы.



---



\## 🇬🇧 Task description



\### Goal

Build a baseline neural network for regression, implement training, and tune hyperparameters to improve model performance.



\### Tasks

1\. Perform exploratory data analysis (EDA).

2\. Preprocess data (scaling, normalization, encode categorical features).

3\. Build a baseline neural network:

&nbsp;  - select number of hidden layers and neurons,

&nbsp;  - choose activation functions for hidden and output layers,

&nbsp;  - implement model training and select an evaluation metric.

4\. Perform hyperparameter tuning (grid search) for:

&nbsp;  - activation function,

&nbsp;  - Dropout (presence and value),

&nbsp;  - Batch Normalization (presence),

&nbsp;  - batch size.

&nbsp;  Keep layer architecture same as baseline.

5\. Visualize RMSE dependence on hyperparameter combinations.

6\. Compare different combinations and select the best model.

7\. Draw conclusions on the effectiveness of different approaches.



\### Expected results

\- EDA with explanations.

\- RMSE vs. hyperparameters plots.

\- Justification of the best model selection.

\- Clear conclusions based on results.



