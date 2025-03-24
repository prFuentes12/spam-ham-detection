# %%  Librerías e instalación inicial
import kagglehub
import os
import pandas as pd
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# %%  Descargar dataset (solo se ejecuta una vez)
path = kagglehub.dataset_download("wanderfj/enron-spam")
print(" Path to dataset files:", path)

# %%  Cargar y procesar archivos en un DataFrame
data = []

for folder in ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]:
    for label in ["ham", "spam"]:
        folder_path = os.path.join(path, folder, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    data.append({
                        "contenido": content,
                        "etiqueta": 1 if label == "spam" else 0,
                        "fuente": folder,
                        "tipo": label
                    })
            except Exception as e:
                print(f" Error leyendo {file_path}: {e}")

df = pd.DataFrame(data)
print(df.head())

# %%  Limpiar duplicados y revisar nulos

# Ver duplicadas
print(f" Filas duplicadas antes de limpiar: {df.duplicated().sum()}")

# Eliminar duplicadas
df = df.drop_duplicates()
print(f" Duplicadas eliminadas. Total de filas ahora: {len(df)}")

# Verificar valores nulos por columna
print("\n Valores nulos por columna:")
print(df.isnull().sum())


# # %%  Preprocesamiento de texto
# stop_words = set(stopwords.words('english'))
# tokenizer = TreebankWordTokenizer()

# def preprocess_text(text):
#     text = text.lower()
#     tokens = tokenizer.tokenize(text)
#     tokens = [t for t in tokens if t.isalpha()]
#     tokens = [t for t in tokens if t not in stop_words]
#     return " ".join(tokens)

# # Aplicar a la columna de contenido
# df["contenido_procesado"] = df["contenido"].apply(preprocess_text)
# print(df[["contenido", "contenido_procesado"]].head())

# %%  ¡Listo para vectorizar, entrenar modelos, etc!
