from sklearn.ensemble import RandomForestClassifier
import numpy as np

TEST_SIZE = 1000
TOP_RECORDS = int(TEST_SIZE*0.2)


def calculate_money(
        columns_indices: list[int],
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

    x_data = x_data[:, columns_indices]

    money = []
    for _ in range(n):
        random_indices = np.random.choice(
            len(x_data), TEST_SIZE, replace=False
            )

        mask = np.zeros(len(x_data), dtype=bool)
        mask[random_indices] = True

        x_train = x_data[~mask]
        x_test = x_data[mask]

        y_train = y_data[~mask]
        y_test = y_data[mask]

        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)

        # Type TOP_RECORDS the best records
        y_proba = model.predict_proba(x_test)[:, 1]
        top_indices = np.argsort(y_proba)[-TOP_RECORDS:]
        y_pred = np.zeros_like(y_test)
        y_pred[top_indices] = 1

        num_correct = np.sum((y_test == 1) & (y_pred == 1))
        profit = num_correct * 20 - len(columns_indices) * 200
        scaled_profit = profit * (1_000 / TOP_RECORDS)

        money.append(scaled_profit)

    return sum(money)/n
