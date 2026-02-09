# Lab 09 — Детекция опухолей мозга с YOLO
# Lab 09 — Brain Tumor Detection with YOLO

---

##  Описание задания

### Цель работы
Реализовать YOLO-модель для детекции опухолей мозга, используя предобученный backbone для извлечения признаков, а также реализовать комбинированную функцию потерь и NMS.

### Задачи
1. Взять датасет Ultralytics/Brain-tumor.
2. Выбрать предобученный backbone для классификации изображений.
3. Реализовать YOLO-модель с выбранным backbone.
4. Реализовать комбинированную функцию потерь (аналогично YOLO).
5. Провести обучение модели.
6. Реализовать NMS (Non-Maximum Suppression) алгоритм.
7. Сгенерировать предсказания, применить NMS и визуализировать результаты.
8. Проверить адекватность результатов (точность не требуется).

### Требования к результату
- EDA датасета и визуализация примеров.
- Описание выбранного backbone.
- Визуализация предсказанных bounding box’ов.
- Выводы о работе модели.

---

##  Task description

### Goal
Implement a YOLO model for brain tumor detection using a pre-trained image classification backbone, with combined loss function and NMS implementation.

### Tasks
1. Use the Ultralytics/Brain-tumor dataset.
2. Select a pre-trained backbone for image classification.
3. Implement a YOLO model using the selected backbone.
4. Implement a combined loss function (YOLO-style).
5. Train the model.
6. Implement Non-Maximum Suppression (NMS) algorithm.
7. Generate predictions, apply NMS, and visualize results.
8. Verify that the results are reasonable (accuracy not required).

### Expected results
- EDA of the dataset and example visualizations.
- Description of the chosen backbone.
- Visualizations of predicted bounding boxes.
- Clear conclusions on model performance.
