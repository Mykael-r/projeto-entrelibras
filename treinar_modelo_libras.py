import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dados_libras.csv")

X = df.drop("letra", axis=1)
y = df["letra"]

# Divide entre treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Treino do modelo
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_treino, y_treino)

# Avaliação
y_pred = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")

# Matriz de confusão
cm = confusion_matrix(y_teste, y_pred, labels=sorted(y.unique()))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

joblib.dump(modelo, "modelo_libras.pkl")
print("Modelo salvo como modelo_libras.pkl")