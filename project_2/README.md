# Wyjaśnienie folderów
### 1. data
Pliki z danymi

### 2. models
Tam będą zapisane pickle wytrenowanych modeli

### 3. notebooks
Notebooki do analizy danych, trenowania modeli, testowania rozwiązań


# Omówienie notatników
### 1. dataset_analyse.ipynb
Przy pomocy lasu losowego wybieramy 10 najistotniejszych kolumn.
Póki co, bierzemy te 10 kolumn z 500 które otrzymaliśmy oraz ich permutacji. Najlepsze kolumny są zapisywane do pliku `the_best_features.txt`.
TODO: Znaleźć X najlepszych kolumn z 500, które posiadamy
TODO: puścić to na algorytmie Boruta (mój lapek za słaby na te tysiące kolumn)

### 2. train_model.ipynb
Spróbujmy wytrnować jakis model liniowy na 10 najlepszych kolumnach.
Mi się udało osiągnąć najwyżej 72%, to mało.
TODO: Przetestować jakiś U-Net
TODO: Przetestować na innych kolumnach niż te 10 z lasu losowego

### 3. xgboost.ipynb
Spróbowałem wytrenować xgboost na 10 najlepszych kolumnach z tysięcy, wyniki podobne jak model liniowy (okolice 70%). To nie jest dużo :/
TODO: przetestować inne parametry XGBoost'a
TODO: Sprawdzić go na innym pakiecie kolumn
