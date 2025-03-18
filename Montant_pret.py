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

# Fonction pour charger les données et les afficher
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Application Streamlit
st.title("Prédiction du Montant du Prêt")

# Charger les données
fichier = "loan_approval_dataset.csv"
df_loan = load_data(fichier)

# Suppression de la colonne 'Loan_id'
df_loan.drop('loan_id', axis=1, inplace=True)

# Transformation des variables catégorielles en numériques
df_loan[' education'] = df_loan[' education'].map({' Not Graduate': 0, ' Graduate': 1})
df_loan[' self_employed'] = df_loan[' self_employed'].map({' No': 0, ' Yes': 1})
df_loan[' loan_status'] = df_loan[' loan_status'].map({' Rejected': 0, ' Approved': 1})

# Séparation des données en variables indépendantes (X) et variable cible (y)
X_reg = df_loan.drop(columns=[' loan_amount'])
y_reg = df_loan[' loan_amount']

# Standardisation des données
scaler = StandardScaler()
X_reg_scaled = pd.DataFrame(scaler.fit_transform(X_reg), columns=X_reg.columns)

# Division en ensembles d'entraînement et de test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Initialisation des modèles
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf')
}

# Entraînement et évaluation des modèles
results = {}

for name, model in models.items():
    model.fit(X_train_reg, y_train_reg)
    train_pred = model.predict(X_train_reg)
    test_pred = model.predict(X_test_reg)

    train_rmse = np.sqrt(mean_squared_error(y_train_reg, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, test_pred))
    train_mse = mean_squared_error(y_train_reg, train_pred)
    test_mse = mean_squared_error(y_test_reg, test_pred)
    train_mae = mean_absolute_error(y_train_reg, train_pred)
    test_mae = mean_absolute_error(y_test_reg, test_pred)
    train_r2 = r2_score(y_train_reg, train_pred)
    test_r2 = r2_score(y_test_reg, test_pred)

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

# Convertir les résultats en DataFrame
df_results = pd.DataFrame(results).T

# Sélectionner le meilleur modèle en fonction des critères
best_model = df_results.sort_values(by=['Test MSE', 'Test MAE', 'Test R2'], ascending=[True, True, False]).head(1)

# Afficher le meilleur modèle
st.header("Meilleur Modèle de Prédiction")
st.write(best_model)

# Visualiser les performances des modèles
metrics = ["Test RMSE", "Test MSE", "Test MAE", "Test R2"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    sns.barplot(x=df_results.index, y=df_results[metric], ax=axes[i])
    axes[i].set_title(metric)
    axes[i].tick_params(axis='x', rotation=45)

# Passez la figure à st.pyplot() pour éviter l'avertissement
st.pyplot(fig)

# Interface pour que l'utilisateur saisisse les valeurs
st.header("Prédisez le Montant du Prêt")

# Interface de saisie des variables pour la prédiction
education = st.selectbox('Niveau d\'éducation', ['Not Graduate', 'Graduate'])
self_employed = st.selectbox('Statut d\'emploi', ['No', 'Yes'])
loan_status = st.selectbox('Statut du prêt', ['Rejected', 'Approved'])

# Vous pouvez ajouter d'autres champs de saisie pour d'autres variables
applicant_income = st.number_input('Revenu de l\'applicant', min_value=0, step=1000)
coapplicant_income = st.number_input('Revenu du coapplicant', min_value=0, step=1000)
loan_amount = st.number_input('Montant du prêt', min_value=0, step=1000)
loan_amount_term = st.selectbox('Durée du prêt', [12, 24, 36, 48, 60])
credit_history = st.selectbox('Historique de crédit', [0, 1])

# Créer un dictionnaire avec les valeurs saisies
input_data = {
    'education': [0 if education == 'Not Graduate' else 1],
    'self_employed': [0 if self_employed == 'No' else 1],
    'loan_status': [0 if loan_status == 'Rejected' else 1],
    'applicant_income': [applicant_income],
    'coapplicant_income': [coapplicant_income],
    'loan_amount': [loan_amount],
    'loan_amount_term': [loan_amount_term],
    'credit_history': [credit_history]
}

# Convertir le dictionnaire en DataFrame
input_df = pd.DataFrame(input_data)

# Assurez-vous que l'ordre des colonnes dans `input_df` correspond à l'ordre d'origine des données d'entraînement
input_df = input_df[X_reg.columns]  # Reorganiser les colonnes pour correspondre à l'ordre de X_reg

# Appliquer la même transformation que pour les données d'entraînement
input_df_scaled = scaler.transform(input_df)

# Prédire le montant du prêt avec le meilleur modèle
best_model_name = best_model.index[0]
best_model_instance = models[best_model_name]
prediction = best_model_instance.predict(input_df_scaled)

# Afficher la prédiction
st.subheader(f"Montant du prêt prédit avec le modèle {best_model_name}")
st.write(f"Le montant prédit du prêt est : {prediction[0]:.2f}")
