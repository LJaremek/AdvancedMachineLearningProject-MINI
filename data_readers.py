import pandas as pd


def ionosphere() -> tuple:
    df = pd.read_csv("data/ionosphere/ionosphere.data", header=None)

    df.iloc[:, -1] = df.iloc[:, -1].map({'g': 1, 'b': 0})

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return x, y


def water_quality() -> tuple:
    df = pd.read_csv("data/WaterQuality/water_potability.csv")

    x = df.drop("Potability", axis=1)
    y = df["Potability"]

    return x, y
