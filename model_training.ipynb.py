# ================================
# 1) IMPORT LIBRARIES
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ================================
# 2) DATASET LOCATION
# ================================

FILEPATH = r"E:\Bkup 23.08.2021\profile bkup\Downloads\archive\Crop_recommendation.csv"

TARGET = "label"


# ================================
# 3) LOAD DATASET
# ================================

print("Loading dataset...")

df = pd.read_csv(FILEPATH)

print("\nDataset Shape:", df.shape)
print("Columns:", df.columns)


# ================================
# 4) CREATE FERTILITY LEVEL
# ================================

df["nutrient_avg"] = (df["N"] + df["P"] + df["K"]) / 3

def fertility_level(x):

    if x < 40:
        return "Low"

    elif x < 80:
        return "Medium"

    else:
        return "High"

df["Fertility_Level"] = df["nutrient_avg"].apply(fertility_level)

print("\nFertility Level Distribution:")
print(df["Fertility_Level"].value_counts())


# ================================
# 5) FEATURES AND TARGET
# ================================

X = df.drop([TARGET, "nutrient_avg", "Fertility_Level"], axis=1)
y = df[TARGET]


# ================================
# 6) TRAIN TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ================================
# 7) MODELS
# ================================

models = {

    "Random Forest": RandomForestClassifier(n_estimators=200),

    "Decision Tree": DecisionTreeClassifier(),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC(),

    "Naive Bayes": GaussianNB()

}


# ================================
# 8) TRAIN AND EVALUATE MODELS
# ================================

accuracy_results = {}

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    accuracy_results[name] = acc

    print(f"\n{name}")
    print("---------------------------")
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# ================================
# 9) MODEL COMPARISON
# ================================

print("\n==============================")
print("MODEL ACCURACY COMPARISON")
print("==============================")

for model, acc in accuracy_results.items():
    print(f"{model}: {acc:.4f}")


# ================================
# 10) BEST MODEL SELECTION
# ================================

best_model_name = max(accuracy_results, key=accuracy_results.get)

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", accuracy_results[best_model_name])


# ================================
# 11) ACCURACY COMPARISON GRAPH
# ================================

plt.figure(figsize=(8,5))

plt.bar(accuracy_results.keys(), accuracy_results.values())

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=30)

plt.show()


# ================================
# 12) CORRELATION HEATMAP
# ================================

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True)

plt.title("Feature Correlation Heatmap")

plt.show()


# ================================
# 13) USER INPUT PREDICTION
# ================================

def get_user_input():

    print("\nEnter Crop Soil Details")

    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temperature = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    ph = float(input("pH: "))
    rainfall = float(input("Rainfall: "))

    user_data = pd.DataFrame({
        'N':[N],
        'P':[P],
        'K':[K],
        'temperature':[temperature],
        'humidity':[humidity],
        'ph':[ph],
        'rainfall':[rainfall]
    })

    return user_data, N, P, K


# ================================
# 14) PREDICTION
# ================================

user_sample, N, P, K = get_user_input()

prediction = best_model.predict(user_sample)

# Fertility Level Calculation
nutrient_avg = (N + P + K) / 3

if nutrient_avg < 40:
    fertility = "Low"

elif nutrient_avg < 80:
    fertility = "Medium"

else:
    fertility = "High"


print("\n================================")
print("Recommended Crop:", prediction[0])
print("Soil Fertility Level:", fertility)
print("================================")