from scipy.io import arff
import pandas as pd
import numpy as np

def ionosphere() -> tuple:
    ionosphere_data = pd.read_csv("data/ionosphere/ionosphere_clean.data", header=None,index_col=None).to_numpy()
    X_ionosphere = ionosphere_data[:,1:-1].astype(float)
    X_ionosphere = np.delete(X_ionosphere, np.s_[0], axis=1)
    y_ionosphere = ionosphere_data[:,0].astype(int)

<<<<<<< HEAD
    return X_ionosphere, y_ionosphere
=======
    df.iloc[:, -1] = df.iloc[:, -1].map({"g": 1, "b": 0})

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return x, y
>>>>>>> 595238deee6730fdb1a6246d25fbaf9afb587576


def water_quality() -> tuple:
    water_data = pd.read_csv("data/WaterQuality/water_potability_clean.csv").to_numpy()
    water_data = water_data[~np.isnan(water_data).any(axis=1)]
    X_water = water_data[:,:-1].astype(float)
    y_water = water_data[:,-1].astype(int)

    return X_water, y_water

<<<<<<< HEAD
def heart_attack() -> tuple:
    heart_data = pd.read_csv("data/HeartAttackAnalysis/heart.csv").to_numpy()
    X_heart = heart_data[:, :-1].astype(int)
    y_heart = heart_data[:, -1].astype(int)

    return X_heart, y_heart
=======
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
>>>>>>> 595238deee6730fdb1a6246d25fbaf9afb587576
