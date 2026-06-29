### ==============================
# Esto está hecho con IA para probar el SpamClassifierPipeline
### ==============================

import pandas as pd
import os
from sklearn.model_selection import train_test_split
# Importamos tu clase desde el archivo donde la guardaste
from SpamClassifierPipeline import SpamClassifierPipeline 

# ==========================================
# 1. CARGA Y BALANCEO DE DATOS
# ==========================================
print("Cargando y balanceando dataset...")
df = pd.read_csv('spam_ham_dataset.csv')

ham_data = df[df['label_num'] == 0]
spam_data = df[df['label_num'] == 1]
ham_balanced = ham_data.sample(n=len(spam_data), random_state=42)
combined_data = pd.concat([ham_balanced, spam_data]).reset_index(drop=True)

# Separamos en entrenamiento y validación (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    combined_data['text'], 
    combined_data['label_num'], 
    test_size=0.2, 
    random_state=42
)

# ==========================================
# 2. ENTRENAMIENTO Y GUARDADO AUTOMÁTICO
# ==========================================
# Instanciamos el pipeline configurando el tamaño de secuencia
pipeline = SpamClassifierPipeline(max_len=100)

if(os.path.exists("modelo_spam")):
    pipeline.cargar_pipeline()
else:
    # Entrenamos (esto crea la carpeta 'modelo_spam' con el .keras, el .json y el .txt)
    pipeline.entrenar(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)

    # ==========================================
    # 3. REINICIAR Y CARGAR DESDE DISCO
    # ==========================================
    # Simulamos un script totalmente nuevo borrando el objeto anterior
    del pipeline

    # Creamos una tubería vacía de producción y la resucitamos desde la carpeta guardada
    pipeline = SpamClassifierPipeline(max_len=100)
    pipeline.cargar_pipeline("modelo_spam")

# ==========================================
# 4. PREDICCIÓN EN CALIENTE (INFERENCIA)
# ==========================================
correos_nuevos = [
    "Hey Félix, remember we have to review the Kubernetes deployment manifests tomorrow.",
    "WINNER!! You have been selected for a free cash prize. Click here to claim your money now.",
    "Hola Juan, aqui tienes el link para nuestra reunion a las 8:00"
]

# Predecimos directamente usando el umbral estándar por defecto (0.5)
resultados = pipeline.predecir(correos_nuevos)

print("\n=== RESULTADOS DE LA PREDICCIÓN (0 es Ham y 1 es Spam)===")

for resultado in resultados:

    print(f"\n- Correo: '{resultado['texto'][:60]}...'")
    print(f"  Sigmoid: {resultado['sigmoid']:.4f}")
    print(f"  Resultado Final:  [{resultado['clase']}]")