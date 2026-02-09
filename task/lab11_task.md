# Lab 11 — ViT/Swin трансформеры и контрастное обучение
# Lab 11 — ViT/Swin Transformers and Contrastive Learning

---

##  Описание задания

### Цель работы
Реализовать пайплайн обучения ViT или Swin трансформера для классификации изображений и объединить результаты с текстовой моделью из Lab 10 с использованием контрастного обучения для zero-shot классификации.

### Задачи
1. Выбрать любой датасет классификации изображений (желательно похожий на Lab 10) и сделать базовое описание.
   - Если изображений много, можно ограничить число классов (>5 классов должно остаться).
   - Можно уменьшать размеры изображений для упрощения обучения.
2. Реализовать все компоненты ViT или Swin трансформера.
3. Реализовать пайплайн обучения модели классификации изображений.
4. Дообучить текстовую модель из Lab 10 и модель из этой лабораторной с использованием контрастного обучения.
5. Продемонстрировать возможность **zero-shot классификации** и сравнения эмбеддингов двух моделей.
6. Ограничения такие же, как в Lab 10 (без готовых блоков для трансформеров и эмбеддингов).

### Требования к результату
- EDA и описание датасета.
- Реализация всех компонентов трансформера.
- Метрики качества модели классификации изображений.
- Демонстрация контрастного обучения и zero-shot классификации.
- Выводы по результатам работы.

---

##  Task description

### Goal
Implement a training pipeline for ViT or Swin transformer for image classification and combine it with the text model from Lab 10 using contrastive learning for zero-shot classification.

### Tasks
1. Select any image classification dataset (preferably similar to Lab 10) and provide a basic description.
   - If there are many images, limit the number of classes (>5 classes must remain).
   - Resize images if necessary to simplify training.
2. Implement all components of ViT or Swin transformer.
3. Implement a training pipeline for image classification.
4. Fine-tune the text model from Lab 10 and the model from this lab using contrastive learning.
5. Demonstrate **zero-shot classification** and embedding comparison between the two models.
6. Same restrictions as Lab 10 (no pre-built transformer blocks or embedders).

### Expected results
- EDA and dataset description.
- Full implementation of transformer components.
- Image classification model metrics.
- Demonstration of contrastive learning and zero-shot classification.
- Clear conclusions based on results.
