# Machine Learning and Data Science Portfolio

Este repositorio contiene el historial de proyectos y el desarrollo técnico enfocado hacia el rol de AI Engineer y Machine Learning Engineer. El objetivo principal es consolidar los fundamentos prácticos del aprendizaje automático clásico y profundo, conectando la experimentación en entornos de análisis de datos con scripts preparados para producción.

---

## Metodología de Desarrollo y Buenas Prácticas

Con el fin de evitar el sesgo de asimilación pasiva al seguir tutoriales de GeeksforGeeks, el flujo de trabajo para cada proyecto se divide estrictamente en las siguientes fases:

1. **Fase de Prospección (Jupyter Notebook):** Archivos estructurados como `*_exploration.ipynb` o `*_experiments.ipynb`. Se utilizan para el Análisis Exploratorio de Datos (EDA), transformaciones iniciales, visualización matemática y pruebas rápidas de algoritmos.
2. **Fase de Producción (Scripts Python):** El código validado en el notebook se refactoriza en scripts modulares (`train.py`). Se aplican principios de diseño de software como modularidad, encapsulamiento en funciones y reproducibilidad.
3. **Eficiencia y Serialización:** Se prioriza el uso de operaciones vectoriales nativas en NumPy (como la normalización absoluta mediante divisiones flotantes) sobre abstracciones innecesarias cuando las restricciones del problema lo permiten. Los modelos optimizados se serializan mediante `joblib` o `pickle` para su posterior consumo en APIs o microservicios.

---

## Estructura del Repositorio

```text
machine-learning-portfolio/
│
├── .gitignore               # Exclusión de datasets binarios, checkpoints de Jupyter y modelos (.pkl)
├── README.md                # Índice principal y control de progreso
├── requirements.txt         # Gestión de dependencias del entorno (Scikit-Learn, Pandas, NumPy, OpenCV)
│
├── 01_beginners/            # Proyectos de Fundamentos (ML Clásico y Procesamiento de Señal)
│   ├── 01_text_image/       # Clasificación estocástica, k-NN y procesamiento base
│   ├── 02_finance/          # Regresión lineal, polinómica y series temporales
│   └── 03_retail/           # Clustering, segmentación y forecasting clásico
│
└── 02_advanced/             # Proyectos de Especialización (Deep Learning, NLP y Computer Vision)
    ├── 01_computer_vision/  # Arquitecturas convolucionales (CNN Custom, ResNet, Transfer Learning)
    └── 02_nlp_speech/       # Modelos secuenciales y Transformers (RNN, LSTM, Fine-tuning de BERT)
```

## Mapa de Ruta y Control de Progreso

### Nivel: Principiante (Beginner Projects)

#### 1. Procesamiento de Texto e Imágenes
- [X] Detecting Spam Emails
- [X] SMS Spam Detection
- [X] Classification of Text Documents
- [X] Classify Handwritten Digits (k-NN / Logistic Regression)
- [X] OCR of Handwritten digits
- [X] Recognizing HandWritten Digits
- [X] Identifying handwritten digits using Logistic Regression
- [X] Cartooning an Image
- [ ] Count number of Object
- [X] Count number of Faces
- [ ] Text Detection and Extraction
- [X] CIFAR-10 Image Classification
- [ ] Black and white image colorization
- [ ] Handwritten Digit Recognition using Neural Network

#### 2. Redes Sociales y Análisis de Sentimiento
- [ ] Twitter Sentiment Analysis
- [ ] Facebook Sentiment Analysis

#### 3. Finanzas y Economía
- [ ] Credit Card Fraud Detection
- [ ] Dogecoin Price Prediction
- [ ] Zillow Home Value (Zestimate) Prediction
- [ ] Bitcoin Price Prediction
- [ ] Online Payment Fraud Detection
- [ ] Stock Price Prediction
- [ ] Stock Price Prediction Project using TensorFlow
- [ ] Microsoft Stock Price Prediction
- [ ] Predicting Stock Price Direction using Support Vector Machines
- [ ] Share Price Forecasting Using Facebook Prophet

#### 4. Comercio y Gestión de Clientes (Retail)
- [ ] Sales Forecast Prediction
- [ ] Customer Churn Analysis Prediction
- [ ] Inventory Demand Forecasting
- [ ] Customer Segmentation
- [ ] Analyzing selling price of used cars
- [ ] Box Office Revenue Prediction
- [ ] Flipkart Reviews Sentiment Analysis
- [ ] Click-Through Rate Prediction
- [ ] Loan Approval Prediction using Multiple Machine Learning Models
- [ ] Loan Eligibility prediction using SVM
- [ ] House Price Prediction
- [ ] Boston Housing Prediction
- [ ] Employee Management System

#### 5. Salud y Medicina (Healthcare)
- [ ] Disease Prediction
- [ ] Heart Disease Prediction Using Logistic Regression
- [ ] Prediction of Wine type
- [ ] Parkinson’s Disease Prediction
- [ ] Breast Cancer Wisconsin Diagnosis using Logistic Regression
- [ ] Cancer cell classification
- [ ] Breast Cancer Wisconsin Diagnosis using KNN and Cross-Validation
- [ ] Autism Prediction
- [ ] Medical Insurance Price Prediction
- [ ] Skin Cancer Detection
- [ ] Heart Disease Prediction using ANN
- [ ] Predicting Air Quality Index
- [ ] Predicting Air Quality with Neural Networks
- [ ] Titanic Survival Prediction

#### 6. Alimentación y Deportes
- [ ] Wine Quality Prediction
- [ ] IPL Score Prediction Using Deep Learning
- [ ] Calories Burnt Prediction using Machine Learning

#### 7. Transporte y Medio Ambiente
- [ ] Vehicle Count Prediction From Sensor Data
- [ ] Ola Bike Ride Request Forecast
- [ ] Rainfall Prediction

#### 8. Otros Modelos Base
- [ ] Spaceship Titanic Project
- [ ] Waiter’s Tip Prediction
- [ ] Fake News Detection
- [ ] Fake News Detection Model
- [ ] Predict Fuel Efficiency

---

### Nivel: Avanzado (Advanced Projects)

#### 1. Procesamiento de Imagen y Vídeo Avanzado
- [ ] Multiclass image classification
- [ ] Image Caption Generator
- [ ] FaceMask Detection
- [ ] Dog Breed Classification
- [ ] Flower Recognition
- [ ] Cat & Dog Classification using CNN
- [ ] Traffic Signs Recognition
- [ ] Residual Networks (ResNet)
- [ ] Lung Cancer Detection using CNN
- [ ] Lung Cancer Detection Using Transfer Learning
- [ ] Pneumonia Detection using Deep Learning
- [ ] Detecting Covid-19 with Chest X-ray
- [ ] Detecting COVID-19 From Chest X-Ray Images using CNN
- [ ] Image Segmentation

#### 2. Sistemas de Recomendación
- [ ] Ted Talks Recommendation System
- [ ] Movie Recommender System
- [ ] Movie recommendation based on emotion
- [ ] Music Recommendation System

#### 3. Procesamiento de Voz y Lenguaje Natural (NLP)
- [ ] Speech Recognition
- [ ] Voice Assistant
- [ ] Next Sentence Prediction
- [ ] Hate Speech Detection
- [ ] Fine-tuning BERT model for Sentiment Analysis
- [ ] Sentiment Classification Using BERT
- [ ] Sentiment Analysis with RNN
- [ ] Autocorrect Feature
- [ ] Analysis of Restaurant reviews
- [ ] Restaurant Review Analysis Using NLP and SQLite

#### 4. Seguridad y Monitorización Secuencial
- [ ] Intrusion Detection System
- [ ] License Plate Recognition
- [ ] Detect and Recognize Car License Plate
- [ ] Age Detection
- [ ] Face and Hand Landmarks Detection
- [ ] Human Activity Recognition
- [ ] Sequential Model with Abalone Dataset

---

## Stack Tecnológico Principal

* **Core:** Python 3.x, NumPy, Pandas, SciPy.
* **Modelado Clásico:** Scikit-Learn.
* **Deep Learning & NLP:** TensorFlow, Keras, HuggingFace Core.
* **Visión por Computador:** OpenCV (cv2).
* **Análisis Gráfico:** Matplotlib.
