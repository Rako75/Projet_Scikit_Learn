import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 📌 **Chargement des données**
fichier = "loan_approval_dataset.csv"
df_loan = pd.read_csv(fichier)

# Suppression de la colonne identifiant et des valeurs manquantes
df_loan.drop(columns=['loan_id'], inplace=True)
df_loan.dropna(inplace=True)

# Renommage des colonnes en français
df_loan.rename(columns={
    ' education': 'Niveau d éducation',
    ' self_employed': 'Travailleur indépendant',
    ' loan_status': 'Statut du prêt',
    ' no_of_dependents': 'Nombre de personnes à charge',
    ' income_annum': 'Revenu annuel',
    ' cibil_score': 'Score CIBIL',
    ' residential_assets_value': 'Valeur des biens résidentiels',
    ' commercial_assets_value ': 'Valeur des biens commerciaux',
    ' luxury_assets_value': 'Valeur des biens de luxe',
    ' bank_asset_value': 'Valeur des actifs bancaires',
    ' loan_amount': 'Montant du prêt'
}, inplace=True)

# Mapping des variables catégorielles
df_loan['Niveau d éducation'] = df_loan['Niveau d éducation'].map({' Not Graduate': 0, ' Graduate': 1})
df_loan['Travailleur indépendant'] = df_loan['Travailleur indépendant'].map({' No': 0, ' Yes': 1})
df_loan['Statut du prêt'] = df_loan['Statut du prêt'].map({' Rejected': 0, ' Approved': 1})

# 📊 **Séparation des données pour la régression**
X_reg = df_loan.drop(columns=['Montant du prêt'])
y_reg = df_loan['Montant du prêt']

# Normalisation
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Division en ensembles d'entraînement et de test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# 📌 **Modèles de régression**
regression_models = {
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

# 📊 **Entraînement des modèles de régression**
regression_results = {}

for name, model in regression_models.items():
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)

    regression_results[name] = {
        "MSE": mean_squared_error(y_test_reg, y_pred),
        "MAE": mean_absolute_error(y_test_reg, y_pred),
        "R²": r2_score(y_test_reg, y_pred)
    }

# Sélection du meilleur modèle de régression
df_reg_results = pd.DataFrame(regression_results).T
best_reg_model = df_reg_results["MSE"].idxmin()
best_reg_instance = regression_models[best_reg_model]

# 📌 **Séparation des données pour la classification**
X_class = df_loan.drop(columns=['Statut du prêt'])
y_class = df_loan['Statut du prêt']

# Normalisation
X_class_scaled = scaler.fit_transform(X_class)

# Division en ensembles d'entraînement et de test
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)

# 📌 **Modèles de classification**
classification_models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# 📊 **Entraînement des modèles de classification**
classification_results = {}

for name, model in classification_models.items():
    model.fit(X_train_class, y_train_class)
    y_pred = model.predict(X_test_class)

    classification_results[name] = {
        "Accuracy": accuracy_score(y_test_class, y_pred),
        "Precision": precision_score(y_test_class, y_pred, average='weighted'),
        "Recall": recall_score(y_test_class, y_pred, average='weighted'),
        "F1-score": f1_score(y_test_class, y_pred, average='weighted')
    }

# Sélection du meilleur modèle de classification
df_class_results = pd.DataFrame(classification_results).T
best_class_model = df_class_results["Accuracy"].idxmax()
best_class_instance = classification_models[best_class_model]

# 📊 **Interface utilisateur Streamlit**
st.title("📊 Prédiction et Classification des prêts")

st.sidebar.subheader("🔢 Entrée des données utilisateur")
user_input = {}

for col in X_class.columns:
    if col == 'Niveau d éducation':
        education_option = st.sidebar.selectbox("Niveau d'éducation", ["Non diplômé", "Diplômé"])
        user_input[col] = 1 if education_option == "Diplômé" else 0
    elif col == 'Travailleur indépendant':
        self_employed_option = st.sidebar.selectbox("Travailleur indépendant", ["Non", "Oui"])
        user_input[col] = 1 if self_employed_option == "Oui" else 0
    elif col == 'Nombre de personnes à charge':
        user_input[col] = st.sidebar.number_input("Nombre de personnes à charge", min_value=0, max_value=5, step=1, value=3)
    else:
        user_input[col] = st.sidebar.number_input(f"{col}", float(df_loan[col].min()), float(df_loan[col].max()), float(df_loan[col].mean()))

# Transformation des données utilisateur
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# **Prédiction du montant du prêt**
predicted_loan_amount = best_reg_instance.predict(input_scaled)[0]
st.sidebar.write(f"💰 **Montant du prêt prédit** : {predicted_loan_amount:,.2f} ")

# **Prédiction du statut du prêt**
if st.sidebar.button("🔮 Prédire le Statut du Prêt"):
    predicted_status = best_class_instance.predict(input_scaled)[0]
    st.sidebar.write(f"🏦 **Statut du prêt prédit** : {'Approuvé' if predicted_status == 1 else 'Refusé'}")
