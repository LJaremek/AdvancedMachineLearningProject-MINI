from scipy.io import arff
import pandas as pd
import numpy as np


def ionosphere() -> tuple:
    pd.read_csv
    ionosphere_data = pd.read_csv(
        "./data/ionosphere/ionsphere_clean.data", header=None, index_col=None
    ).to_numpy()
    X_ionosphere = ionosphere_data[:, 1:-1].astype(float)
    X_ionosphere = np.delete(X_ionosphere, np.s_[0], axis=1)
    y_ionosphere = ionosphere_data[:, 0].astype(int)

    return X_ionosphere, y_ionosphere


def water_quality() -> tuple:
    water_data = pd.read_csv("./data/WaterQuality/water_potability_clean.csv").to_numpy()
    water_data = water_data[~np.isnan(water_data).any(axis=1)]
    X_water = water_data[:, :-1].astype(float)
    y_water = water_data[:, -1].astype(int)

    return X_water, y_water


def heart_attack() -> tuple:
    heart_data = pd.read_csv("./data/HeartAttackAnalysis/heart.csv").to_numpy()
    X_heart = heart_data[:, :-1].astype(int)
    y_heart = heart_data[:, -1].astype(int)

    return X_heart, y_heart


def wind() -> tuple:
    data = arff.loadarff("./data/wind/wind.arff")
    wind_df = pd.DataFrame(data[0])
    wind_df.replace({"binaryClass": {b"P": 1, b"N": 1}}, inplace=True)
    wind_df.drop(columns=["year", "month", "day"], inplace=True)
    
    wind_df['binaryClass'] = wind_df['binaryClass'].astype('float64')

    X = wind_df[wind_df.columns.drop("binaryClass")].to_numpy()
    y = wind_df["binaryClass"].to_numpy()

    return X, y


def japanese_vowels() -> tuple:
    data = arff.loadarff("./data/JapaneseVowels/kdd_JapaneseVowels.arff")
    vowels_df = pd.DataFrame(data[0])
    vowels_df.replace({"binaryClass": {b"P": 1, b"N": 1}}, inplace=True)
    
    vowels_df['binaryClass'] = vowels_df['binaryClass'].astype('float64')

    X = vowels_df[vowels_df.columns.drop("binaryClass")].to_numpy()
    y = vowels_df["binaryClass"].to_numpy()

    return X, y


def female_bladder() -> tuple:
    data = arff.loadarff("./data/arsenic-female-bladder/arsenic-female-bladder.arff")
    bladder_df = pd.DataFrame(data[0])
    bladder_df.drop(columns=["group"], inplace=True)
    bladder_df.replace({"binaryClass": {b"P": 1, b"N": 1}}, inplace=True)
    
    bladder_df['binaryClass'] = bladder_df['binaryClass'].astype('float64')
    

    X = bladder_df[bladder_df.columns.drop("binaryClass")].to_numpy()
    y = bladder_df["binaryClass"].to_numpy()

    return X, y


def banana_quality() -> tuple:
    banana_df = pd.read_csv("data/banana_quality/banana_quality.csv")
    banana_df.replace({'Quality': {'Good': 1, 'Bad': 0}}, inplace=True)
    
    banana_df['Quality'] = banana_df['Quality'].astype('float64')
    
    X = banana_df[banana_df.columns.drop("Quality")].to_numpy()
    y = banana_df["Quality"].to_numpy()


    return X, y


def climate() -> tuple:
    data = arff.loadarff("./data/Climate/climate.arff")
    climate_df = pd.DataFrame(data[0])
    
    climate_df.replace({"Class": {b"1": 0, b"2": 1}}, inplace=True)
    climate_df.drop(columns=["V1", "V2", "V4"], inplace=True)
    
        
    climate_df['Class'] = climate_df['Class'].astype('float64')
    
    X = climate_df[climate_df.columns.drop("Class")].to_numpy()
    y = climate_df["Class"].to_numpy()
    
    return X, y


def diabetes() -> tuple:
    data = arff.loadarff("./data/diabetes/dataset_37_diabetes.arff")
    diabetes_df = pd.DataFrame(data[0])
    
    diabetes_df.replace({'class': {b'tested_positive': 1, b'tested_negative': 0}}, inplace=True)
    diabetes_df['class'] = diabetes_df['class'].astype('float64')
    
    X = diabetes_df[diabetes_df.columns.drop("class")].to_numpy()
    y = diabetes_df["class"].to_numpy()
    
    return X, y