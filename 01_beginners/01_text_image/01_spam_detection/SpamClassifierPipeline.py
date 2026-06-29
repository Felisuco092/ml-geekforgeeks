import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SpamClassifierPipeline:
    def __init__(self, max_len=100):
        self.model = None
        self.tokenizer = None
        self.max_len = max_len
    
    def preparar_datos(self, textos):
        if self.tokenizer is None:
            raise ValueError("Primero entrena el modelo con datos para el tokenizer")
        
        secuencias = self.tokenizer.texts_to_sequences(textos)
        padding_vector = pad_sequences(secuencias, maxlen=self.max_len, padding='post', truncating='post')

        return padding_vector
    
    def entrenar(self, train_textos, train_etiquetas, val_textos, val_etiquetas, epochs=10, batch_size=32, ruta_guardado="modelo_spam"):
        """Para entrenar al modelo pasando los textos, los labels"""
        os.makedirs(ruta_guardado, exist_ok=True)
        
        # 1. Inicializar y ajustar el Tokenizer con el texto de entrenamiento
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(train_textos)
        vocab_size = len(self.tokenizer.word_index) + 1
        
        # 2. Procesar los datasets
        X_train, y_train = self.preparar_datos(train_textos), train_etiquetas
        X_val, y_val = self.preparar_datos(val_textos), val_etiquetas
        
        # 3. Construir la arquitectura de la red
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=self.max_len),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation="sigmoid") # Logits crudos
        ])
        
        # 4. Configurar optimizador, pérdida y métricas de desbalanceo
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        # 5. Callbacks estratégicos
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5)
        ]
        
        
        # 6. Ejecutar el entrenamiento
        historial = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # 7. Guardar artefactos en disco
        # El modelo nativo de Keras (.keras guarda arquitectura, pesos y estado del optimizador)
        self.model.save(os.path.join(ruta_guardado, "modelo_lstm.keras"))
        
        # Guardar el Tokenizer en formato JSON (imprescindible para el futuro)
        tokenizer_json = self.tokenizer.to_json()
        with open(os.path.join(ruta_guardado, "tokenizer.json"), "w", encoding="utf-8") as f:
            f.write(tokenizer_json)
            
        # Guardar las métricas finales en un archivo de texto de registro
        self._guardar_reporte_metricas(historial, ruta_guardado)
        print(f"--- Entrenamiento completado con éxito. Artefactos guardados en '{ruta_guardado}/' ---")
    
    def _guardar_reporte_metricas(self, historial, ruta_guardado="modelo_spam"):
        """Genera un archivo .txt con la evolución de las métricas de la última época."""
        ultima_epoca = len(historial.history['loss']) - 1
        with open(os.path.join(ruta_guardado, "reporte_metricas.txt"), "w", encoding="utf-8") as f:
            f.write("=== REPORTE FINAL DE ENTRENAMIENTO ===\n")
            f.write(f"Épocas totales ejecutadas: {ultima_epoca + 1}\n\n")
            f.write("Métricas de validación finales logradas:\n")
            for metrica, valores in historial.history.items():
                f.write(f"- {metrica}: {valores[ultima_epoca]:.4f}\n")
    
    def cargar_pipeline(self, ruta_guardado="modelo_spam"):
        """Cargamos en el objeto los modelos a partir de archivos keras y json del tokenizer"""
        # Primero cargamos el modelo
        self.model = tf.keras.models.load_model(os.path.join(ruta_guardado, "modelo_lstm.keras"))

        #Cargar json
        with open(os.path.join(ruta_guardado, "tokenizer.json"), "r", encoding="utf-8") as f:
            text_tokenizer = f.read()
            self.tokenizer = tokenizer_from_json(text_tokenizer)
    
    def predecir(self, lista_textos, umbral=0.5):
        """Predice sobre textos nuevos en caliente."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("El pipeline no está entrenado ni cargado.")
            
        X_pad = self.preparar_datos(lista_textos)
        sigmoid = self.model.predict(X_pad)
        
        # Si sigmoid > 0.5 es Spam (1), de lo contrario Ham (0)
        predicciones_binarias = (sigmoid > umbral).astype(int)
        
        resultados = []
        for texto, logit, pred in zip(lista_textos, sigmoid, predicciones_binarias):
            resultados.append({
                "texto": texto,
                "sigmoid": float(logit[0]),
                "clase": "spam" if pred[0] == 1 else "ham"
            })
        return resultados