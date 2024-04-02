from scipy.io import arff
import pandas as pd


def ionosphere() -> tuple:
    df = pd.read_csv("data/ionosphere/ionosphere.data", header=None)

    df.iloc[:, -1] = df.iloc[:, -1].map({"g": 1, "b": 0})

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return x, y


def water_quality() -> tuple:
    df = pd.read_csv("data/WaterQuality/water_potability.csv")

    x = df.drop("Potability", axis=1)
    y = df["Potability"]

    return x, y


def wind() -> tuple:
    data = arff.loadarff("./data/wind/wind.arff")
    wind_df = pd.DataFrame(data[0])
    wind_df.replace({"binaryClass": {b"P": 1, b"N": 1}}, inplace=True)
    wind_df.drop(columns=["year", "month", "day"], inplace=True)

    X = wind_df[wind_df.columns.drop("binaryClass")].to_numpy()
    y = wind_df["binaryClass"].to_numpy()

    return X, y


def japanese_vowels() -> tuple:
    data = arff.loadarff("./data/JapaneseVowels/kdd_JapaneseVowels.arff")
    vowels_df = pd.DataFrame(data[0])
    vowels_df.replace({"binaryClass": {b"P": 1, b"N": 1}}, inplace=True)

    X = vowels_df[vowels_df.columns.drop("binaryClass")].to_numpy()
    y = vowels_df["binaryClass"].to_numpy()

    return X, y


def female_bladder() -> tuple:
    data = arff.loadarff("./data/arsenic-female-bladder/arsenic-female-bladder.arff")
    bladder_df = pd.DataFrame(data[0])
    bladder_df.drop(columns=["group"], inplace=True)
    bladder_df.replace({"binaryClass": {b"P": 1, b"N": 1}}, inplace=True)

    X = bladder_df[bladder_df.columns.drop("binaryClass")].to_numpy()
    y = bladder_df["binaryClass"].to_numpy()

    return X, y
