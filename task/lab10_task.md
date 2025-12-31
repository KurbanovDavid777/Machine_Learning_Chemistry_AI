\# Lab 10 — Transformer для текстовой классификации

\# Lab 10 — Transformer for Text Classification



---



\## 🇷🇺 Описание задания



\### Цель работы

Реализовать компоненты Transformer с нуля для классификации текста и построить пайплайн обучения модели.



\### Задачи

1\. Выбрать любой датасет классификации текста и сделать его базовое описание.

2\. Реализовать следующие компоненты Transformer:

&nbsp;  - MultiHeadAttention,

&nbsp;  - PositionalEncoding (cosine),

&nbsp;  - TransformerEncoderLayer,

&nbsp;  - TransformerEncoder,

&nbsp;  - TransformerClassifier.

3\. Для неклассификационных задач реализовать аналогичный Transformer<Task> модуль.

4\. Реализовать пайплайн обучения и обучить модель для текстовой классификации.

5\. Продемонстрировать результаты модели.

6\. Разрешено использовать:

&nbsp;  - готовый токенайзер,

&nbsp;  - базовые слои PyTorch,

&nbsp;  - optimizer и функции метрик.

7\. Нельзя использовать:

&nbsp;  - готовый текстовый эмбеддер,

&nbsp;  - готовые SDPA или блоки Transformer.

8\. Рекомендации:

&nbsp;  - выбрать датасет, который можно будет расширить в следующей лабораторной для мультимодальной задачи (например, классификация настроения по фото лица).



\### Требования к результату

\- Реализация всех компонентов Transformer.

\- Обоснование выбора архитектуры.

\- Метрики качества модели.

\- Выводы по результатам работы.



---



\## 🇬🇧 Task description



\### Goal

Implement Transformer components from scratch for text classification and build a training pipeline.



\### Tasks

1\. Select any text classification dataset and provide a basic description.

2\. Implement the following Transformer components:

&nbsp;  - MultiHeadAttention,

&nbsp;  - PositionalEncoding (cosine),

&nbsp;  - TransformerEncoderLayer,

&nbsp;  - TransformerEncoder,

&nbsp;  - TransformerClassifier.

3\. For non-classification tasks, implement a corresponding Transformer<Task> module.

4\. Implement a training pipeline and train the text classification model.

5\. Demonstrate the model results.

6\. Allowed:

&nbsp;  - pre-built tokenizer,

&nbsp;  - basic PyTorch layers,

&nbsp;  - optimizer and metric functions.

7\. Not allowed:

&nbsp;  - pre-built text embedder,

&nbsp;  - ready SDPA or Transformer blocks.

8\. Recommendation:

&nbsp;  - choose a dataset that can be extended in the next lab for a multimodal task (e.g., facial expression classification).



\### Expected results

\- Full implementation of Transformer components.

\- Justification of architecture choices.

\- Model quality metrics.

\- Clear conclusions based on the results.



