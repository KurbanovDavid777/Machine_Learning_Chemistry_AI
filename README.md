# Machine Learning Labs / Лабораторные работы по Машинному Обучению

Этот репозиторий содержит **11 лабораторных работ по машинному обучению**, выполненных в рамках учебного курса.  
Все лабораторные содержат решения в Jupyter Notebook, исходные данные и описание заданий на русском и английском.

This repository contains **11 machine learning labs** completed as part of a course.  
All labs include Jupyter Notebook solutions, source data, and task descriptions in Russian and English.

---

## Структура проекта / Project Structure
```
ML/
│
├─ notebooks/ # Jupyter ноутбуки с решениями / Jupyter notebooks with solutions
|
│ ├─ Lab_1.ipynb
│ ├─ Lab_2.ipynb
│ ├─ Lab_3.ipynb
│ ├─ Lab_4.ipynb
│ ├─ Lab_5.ipynb
│ ├─ Lab_6.ipynb
│ ├─ Lab_7.ipynb
│ ├─ Lab_8.ipynb
│ ├─ Lab_9.ipynb
│ ├─ Lab_10.ipynb
│ └─ Lab_11.ipynb
│
├─ data/ # Данные для лабораторных / Data for labs
│ ├─ data_lab_1.csv
│ ├─ data_lab_2.csv
│ ├─ data_lab_3.csv
│ ├─ data_lab_4.csv
│ ├─ data_lab_5.csv
│ └─ data_lab_6.csv
│
├─ task/ # Задания на русском и английском / Tasks in Russian and English
│ ├─ lab01_task.md
│ ├─ lab02_task.md
│ ├─ lab03_task.md
│ ├─ lab04_task.md
│ ├─ lab05_task.md
│ ├─ lab06_task.md
│ ├─ lab07_task.md
│ ├─ lab08_task.md
│ ├─ lab09_task.md
│ ├─ lab010_task.md
│ └─ lab11_task.md
│
└─ README.md # Этот файл / This file
```


---

## Лабораторные работы / Labs

| №  | Название / Title | Краткое описание / Short Description |
|----|-----------------|--------------------------------------|
| 1  | Предобработка данных / Data preprocessing | Обработка данных, анализ признаков, построение модели для предсказания цены квартиры. / Data preprocessing, feature analysis, ML model for predicting apartment price. |
| 2  | Классификация ухода клиента / Customer churn classification | Разработка и оценка классификатора ухода клиента с использованием F1 и ROC AUC. / Build and evaluate a customer churn classifier using F1 and ROC AUC. |
| 3  | Кластеризация стран / Country clustering | Кластеризация стран по социально-экономическим показателям, определение приоритетной помощи. / Cluster countries based on socio-economic indicators to identify priority assistance. |
| 4  | Деревья решений и ансамбли / Decision trees & ensembles | Обучение деревьев решений, Bagging, Stacking, RandomForest, XGBoost. / Train decision trees, Bagging, Stacking, RandomForest, XGBoost. |
| 5  | Временные ряды / Time series | Анализ сезонности и трендов, прогнозирование, метрики MAE/RMSE. / Seasonality and trend analysis, forecasting, metrics MAE/RMSE. |
| 6  | Нейронные сети / Neural networks | Построение сети для регрессии, подбор гиперпараметров. / Build a neural network for regression, hyperparameter tuning. |
| 7  | Нейронные сети с нуля / Neural networks from scratch | Реализация сети без готовых слоёв и оптимизаторов. / Implement network without built-in layers and optimizers. |
| 8  | Fine-tuning CNN / Fine-tuning CNN | Fine-tuning предобученной сети для классификации изображений EuroSAT. / Fine-tune pretrained CNN for EuroSAT image classification. |
| 9  | YOLO детекция / YOLO detection | Реализация YOLO с предобученным backbone для детекции опухолей мозга. / Implement YOLO with pretrained backbone for brain tumor detection. |
| 10 | Transformer для текста / Transformer for text | Реализация Transformer для задач классификации текста. / Implement Transformer for text classification tasks. |
| 11 | ViT/Swin и contrastive learning / ViT/Swin & contrastive learning | ViT/Swin для изображений, контрастное обучение с текстовой моделью. / ViT/Swin for images, contrastive learning with text model. |

---

## Как запускать / How to run

1. Установите [Python 3.10+](https://www.python.org/) и [Jupyter Notebook](https://jupyter.org/).  
2. Создайте виртуальное окружение и активируйте его:

```powershell
python -m venv venv
.\venv\Scripts\activate
