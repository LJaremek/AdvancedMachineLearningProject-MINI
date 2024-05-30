from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

TEST_SIZE = 1000
TOP_RECORDS = int(TEST_SIZE*0.2)


def calculate_money(
        columns_indices: list[int],
        model=None,
        x_data: np.array = None,
        y_data: np.array = None,
        n: int = 5
        ) -> float:

    if x_data is None:
        x_train_path = "../data/x_train.txt"
        x_data = np.loadtxt(x_train_path, delimiter=" ")

    if y_data is None:
        y_train_path = "../data/y_train.txt"
        y_data = np.loadtxt(y_train_path, delimiter=" ")

    if model is None:
        model = RandomForestClassifier(n_estimators=100)

    x_data = x_data[:, columns_indices]

    money = []
    for _ in range(n):

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2
            )

        model.fit(x_train, y_train)

        # Type TOP_RECORDS the best records
        y_proba = model.predict_proba(x_test)[:, 1]
        top_indices = np.argsort(y_proba)[-TOP_RECORDS:]
        y_pred = np.zeros_like(y_test)
        y_pred[top_indices] = 1

        num_correct = np.sum((y_test == 1) & (y_pred == 1))
        profit = (
            (num_correct * 10) * (1_000 / TOP_RECORDS) -
            len(columns_indices) * 200
        )

        money.append(profit)

    return sum(money)/n
