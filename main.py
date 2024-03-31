from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from classes import ADAMLogisticRegression
from classes import IWLSLogisticRegression
from classes import SGDLogisticRegression


# Preparing data
models = {
    "ADAM Logistic Regression": ADAMLogisticRegression(
        learning_rate=0.001, iterations=1000, beta1=0.9,
        beta2=0.999, epsilon=1e-8, include_interactions=False
    ),

    "SGD Logistic Regression": SGDLogisticRegression(
        learning_rate=0.01, iterations=1000, include_interactions=False
    ),

    "IWLS Logistic Regression": IWLSLogisticRegression(
        iterations=25, include_interactions=False
    )
}

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()

n_runs = 10
accuracies = {name: [] for name in models.keys()}


# Run experiments
for _ in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        accuracies[name].append(accuracy)


# Show results
for name, scores in accuracies.items():
    print(f"{name} Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.2f}")


data_to_plot = [
    (model, acc)
    for model, acc_list in accuracies.items()
    for acc in acc_list
    ]

df = pd.DataFrame(data_to_plot, columns=["Model", "Accuracy"])

plt.figure(figsize=(10, 6))
sns.boxplot(x="Model", y="Accuracy", data=df)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.show()

plt.savefig("model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
