# Librerías
import pandas as pd
import joblib as jb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Cargar datos
df_zombie = pd.read_csv("data/zombie_data_binary.csv", low_memory=False) 

# Features y target
X = df_zombie.drop(columns=["fecha_zombificacion", "grupo_supervivencia", "sobreviviente"])
Y = df_zombie["sobreviviente"]

# Train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Entrenar el modelo
modelo = DecisionTreeClassifier(max_depth=15, random_state=42, class_weight='balanced')
modelo.fit(X_train, Y_train)

# Evaluar
Y_pred = modelo.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Reporte de clasificación:\n", classification_report(Y_test, Y_pred))

# Matriz de confusión
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

# Guardar modelo
jb.dump(modelo, "model/modelo_zombie.pkl")
