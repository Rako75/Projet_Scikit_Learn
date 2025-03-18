import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Charger les données
fichier = "loan_approval_dataset.csv"
df_loan = pd.read_csv(fichier)

# Prétraitement des données
df_loan.drop('loan_id', axis=1, inplace=True)
df_loan.dropna(inplace=True)
df_loan[' education'] = df_loan[' education'].map({' Not Graduate': 0, ' Graduate': 1})
df_loan[' self_employed'] = df_loan[' self_employed'].map({' No': 0, ' Yes': 1})
df_loan[' loan_status'] = df_loan[' loan_status'].map({' Rejected': 0, ' Approved': 1})
# Séparation des features et de la target
X = df_loan.drop(columns=[' loan_amount'])
y = df_loan[' loan_amount']

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialisation des modèles
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf')
}

# Entraînement et évaluation des modèles
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    results[name] = {
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train R2": train_r2,
        "Test R2": test_r2
    }

# Comparaison des performances des modèles
df_results = pd.DataFrame(results).T
best_model_name = df_results.sort_values(by="Test MSE").index[0]
best_model_instance = models[best_model_name]

# Interface Streamlit
st.title("Prédiction du Montant du Prêt")

# Affichage des performances des modèles
st.subheader("Comparaison des modèles")
st.dataframe(df_results.sort_values(by="Test MSE"))

st.subheader(f"Meilleur modèle: {best_model_name}")
st.write(f"MSE: {df_results.loc[best_model_name, 'Test MSE']:.4f}, MAE: {df_results.loc[best_model_name, 'Test MAE']:.4f}, R2: {df_results.loc[best_model_name, 'Test R2']:.4f}")

# Interface utilisateur pour faire des prédictions
st.subheader("Faire une prédiction")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

# Prétraiter les entrées utilisateur
input_df = pd.DataFrame([user_input])
input_df = input_df[X.columns]  # Réorganiser les colonnes de input_df pour correspondre à X.columns
input_df_scaled = scaler.transform(input_df)

# Prédire le montant du prêt avec le meilleur modèle
if st.button("Prédire le montant du prêt"):
    prediction = best_model_instance.predict(input_df_scaled)
    st.success(f"Montant estimé du prêt: {prediction[0]:.2f}")
