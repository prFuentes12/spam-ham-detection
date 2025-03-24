# %%  Librerías e instalación inicial
import kagglehub
import os
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve 
from sklearn.model_selection import StratifiedKFold





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


# %%  Preprocesamiento de texto
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()

def preprocess_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

# Aplicar a la columna de contenido
df["contenido_procesado"] = df["contenido"].apply(preprocess_text)
print(df)

# %%  Vectorizar y probar distintas configuraciones, comprobar la mejor
# vectorizer = TfidfVectorizer()


# configuraciones = [
#     {'max_features': 1000, 'ngram_range': (1, 1)},
#     {'max_features': 3000, 'ngram_range': (1, 1)},
#     {'max_features': 5000, 'ngram_range': (1, 1)},
#     {'max_features': 5000, 'ngram_range': (1, 2)},
#     {'max_features': 8000, 'ngram_range': (1, 2)},
#     {'max_features': 10000, 'ngram_range': (1, 2)},
# ]

# for i, cfg in enumerate(configuraciones, 1):
#     print(f"\n Configuración {i}: max_features={cfg['max_features']}, ngram_range={cfg['ngram_range']}")

#     vectorizer = TfidfVectorizer(max_features=cfg['max_features'], ngram_range=cfg['ngram_range'])
#     Xi = vectorizer.fit_transform(df["contenido_procesado"])
#     y = df["etiqueta"]

#     model = MultinomialNB()
#     scores = cross_val_score(model, Xi, y, cv=5, scoring='accuracy') 

#     print(f" Accuracy promedio (5-Fold CV): {np.mean(scores):.4f}")


# Usamos la mejor configuración para transformar el texto
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["contenido_procesado"])
y = df["etiqueta"]

# Separar en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Entrenar el modelo
model = LinearSVC()
model.fit(X_train, y_train)

# Predecir en test
y_pred = model.predict(X_test)

# Calcular métricas
print(" Evaluación del Modelo SVM:")
print(f"🔹 Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"🔹 Precision: {precision_score(y_test, y_pred):.4f}")
print(f"🔹 Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"🔹 F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print(" Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))


# Evaluar en train
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# Evaluar en test
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Accuracy en entrenamiento: {train_acc:.4f}")
print(f"Accuracy en test: {test_acc:.4f}")




#Rendimiento del modelo mediante curva

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=skf, scoring='accuracy')

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Entrenamiento')
plt.plot(train_sizes, test_mean, label='Validación')
plt.xlabel('Tamaño del conjunto de entrenamiento')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Curva de aprendizaje')
plt.show()