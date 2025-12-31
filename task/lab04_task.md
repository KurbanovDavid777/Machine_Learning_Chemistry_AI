# Lab 04 — Деревья решений и ансамбли
# Lab 04 — Decision Trees and Ensembles

---

## 🇷🇺 Описание задания

### Цель работы
Изучить методы построения моделей классификации на основе деревьев решений и ансамблей, включая бэггинг, стекинг, случайный лес и XGBoost, а также проанализировать важность признаков.

### Задачи
1. Провести анализ данных (EDA).
2. Предобработать данные (масштабирование, нормализация, кодирование категориальных признаков).
3. Построить дерево решений:
   - визуализировать дерево,
   - исследовать влияние глубины дерева на качество модели,
   - определить оптимальную глубину.
4. Настроить параметры дерева с помощью GridSearchCV.
5. Обучить ансамбли:
   - BaggingClassifier с выбранной базовой моделью,
   - StackingClassifier с выбранными классическими моделями.
6. Обучить RandomForestClassifier и настроить гиперпараметры с помощью GridSearchCV.
7. Обучить XGBoost (только с числовыми признаками) и вычислить F-score для оценки важности признаков.
8. Сделать выводы по проделанной работе.

### Требования к результату
- EDA с комментариями.
- Визуализация дерева решений и кластеров.
- Таблицы и графики важности признаков.
- Выводы по оптимальным параметрам и работе моделей.

---

## 🇬🇧 Task description

### Goal
Study classification methods based on decision trees and ensembles, including bagging, stacking, Random Forest, and XGBoost, and analyze feature importance.

### Tasks
1. Perform exploratory data analysis (EDA).
2. Preprocess data (scaling, normalization, encode categorical features).
3. Build a decision tree:
   - visualize the tree,
   - investigate the effect of tree depth on model quality,
   - determine the optimal depth.
4. Tune tree parameters with GridSearchCV.
5. Train ensembles:
   - BaggingClassifier with chosen base model,
   - StackingClassifier with selected classical models.
6. Train RandomForestClassifier and tune hyperparameters with GridSearchCV.
7. Train XGBoost (numerical features only) and compute F-score for feature importance.
8. Draw conclusions based on the results.

### Expected results
- EDA with explanations.
- Decision tree visualization.
- Tables and plots of feature importance.
- Conclusions on optimal parameters and model performance.
