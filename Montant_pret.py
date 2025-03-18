import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

# Prétraitement
df_loan.drop('loan_id', axis=1, inplace=True)
df_loan.dropna(inplace=True)
df_loan['education'] = df_loan[' education'].map({' Not Graduate': 0, ' Graduate': 1})
df_loan['self_employed'] = df_loan[' self_employed'].map({' No': 0, ' Yes': 1})
df_loan['loan_status'] = df_loan[' loan_status'].map({' Rejected': 0, ' Approved': 1})

# Séparation des features et de la target
X = df_loan.drop(columns=[' loan_amount'])
y = df_loan[' loan_amount']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèles
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

# Entraînement et évaluation
df_results = pd.DataFrame(columns=["Model", "MSE", "MAE", "R2"])

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    df_results = df_results.append({"Model": name, "MSE": mse, "MAE": mae, "R2": r2}, ignore_index=True)

best_model = df_results.sort_values(by="MSE").iloc[0]

# Interface Streamlit
st.title("Prédiction du Montant du Prêt")

# Afficher les performances des modèles
st.subheader("Comparaison des modèles")
st.dataframe(df_results.sort_values(by="MSE"))

st.subheader(f"Meilleur modèle: {best_model['Model']}")
st.write(f"MSE: {best_model['MSE']:.4f}, MAE: {best_model['MAE']:.4f}, R2: {best_model['R2']:.4f}")

# Prédiction
st.subheader("Faire une prédiction")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

if st.button("Prédire le montant du prêt"):
    model = models[best_model['Model']]
    user_data = np.array([list(user_input.values())]).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    st.success(f"Montant estimé du prêt: {prediction[0]:.2f}")
