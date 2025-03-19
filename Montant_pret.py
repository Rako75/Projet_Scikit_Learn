import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Charger les données
fichier = "loan_approval_dataset.csv"
df_loan = pd.read_csv(fichier)

# Prétraitement des données
df_loan.drop('loan_id', axis=1, inplace=True)
df_loan.dropna(inplace=True)
df_loan[' education'] = df_loan[' education'].map({' Not Graduate': 0, ' Graduate': 1})
df_loan[' self_employed'] = df_loan[' self_employed'].map({' No': 0, ' Yes': 1})
df_loan[' loan_status'] = df_loan[' loan_status'].map({' Rejected': 0, ' Approved': 1})

# Renommer les colonnes en français
df_loan.rename(columns={
    ' education': "Niveau d'éducation",
    ' self_employed': "Travailleur indépendant",
    ' applicant_income': "Revenu du demandeur",
    ' coapplicant_income': "Revenu du co-demandeur",
    ' income_annum': "Revenu annuel",
    ' cibil_score': "Score CIBIL",
    ' residential_assets_value': "Valeur des biens résidentiels",
    ' commercial_assets_value': "Valeur des biens commerciaux",
    ' luxury_assets_value': "Valeur des biens de luxe",
    ' bank_asset_value': "Valeur des actifs bancaires",
    ' loan_amount': "Montant du prêt",
    ' loan_term': "Durée du prêt (mois)",
    ' credit_history': "Historique de crédit",
    ' property_area': "Zone de la propriété",
    ' no_of_dependents': "Nombre de personnes à charge"
}, inplace=True)

# Séparation des features et de la target pour la **régression**
X_reg = df_loan.drop(columns=['Montant du prêt'])
y_reg = df_loan['Montant du prêt']

# Standardisation des données
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

    mse = mean_squared_error(y_test_reg, y_pred)
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    regression_results[name] = {
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    }

# 📈 **Sélection du meilleur modèle de régression**
df_reg_results = pd.DataFrame(regression_results).T
best_reg_model = df_reg_results["MSE"].idxmin()
best_reg_instance = regression_models[best_reg_model]

# Définition des features et de la cible pour la **classification**
X_class = df_loan.drop(columns=[' loan_status'])
y_class = df_loan[' loan_status']

# Standardisation des données
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

    acc = accuracy_score(y_test_class, y_pred)
    precision = precision_score(y_test_class, y_pred, average='weighted')
    recall = recall_score(y_test_class, y_pred, average='weighted')
    f1 = f1_score(y_test_class, y_pred, average='weighted')

    classification_results[name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

# 📈 **Sélection du meilleur modèle de classification**
df_class_results = pd.DataFrame(classification_results).T
best_class_model = df_class_results["Accuracy"].idxmax()
best_class_instance = classification_models[best_class_model]

# 📊 **Interface Streamlit**
st.title("📊 Prédiction et Classification des prêts")
st.write("""
Bienvenue dans notre application de prédiction et de classification des demandes de prêt !  
Cette application utilise plusieurs modèles de Machine Learning pour :  
- Prédire le montant du prêt qui peut être accordé.  
- Déterminer si une demande de prêt sera approuvée ou refusée.  

📌 **Comment utiliser cette application ?**  
👉 Remplissez les informations requises dans la barre latérale.  
👉 Obtenez instantanément une prédiction sur le montant du prêt et son statut d'approbation.
""")

# 📌 **Affichage des performances des modèles**
st.subheader("Comparaison des modèles de régression")
st.dataframe(df_reg_results)
st.write(f"🏆 **Meilleur modèle de régression** : {best_reg_model}")

st.subheader("Comparaison des modèles de classification")
st.dataframe(df_class_results)
st.write(f"🏆 **Meilleur modèle de classification** : {best_class_model}")

# ✅ **Prédiction en temps réel**
st.sidebar.header("📊 Prédiction du Montant du prêt et statut")

# Entrée utilisateur pour la régression et classification
user_input = {}
for col in X_class.columns:
    if col == "Niveau d'éducation':
        education_option = st.sidebar.selectbox(" Niveau d'éducation", [" Not Graduate", " Graduate"])
        user_input[col] = 1 if education_option == " Graduate" else 0
    elif col == 'Travailleur indépendant':
        self_employed_option = st.sidebar.selectbox(" Travailleur indépendant", [" No", " Yes"])
        user_input[col] = 1 if self_employed_option == " Yes" else 0
    elif col == 'Nombre de personnes à charge':
        user_input[col] = st.sidebar.number_input(f"{col}", min_value=0, max_value=5, step=1, value=3)
    else:
        user_input[col] = st.sidebar.number_input(f"{col}", float(df_loan[col].min()), float(df_loan[col].max()), float(df_loan[col].mean()))

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# **Prédiction avec le meilleur modèle de régression**
predicted_loan_amount = best_reg_instance.predict(input_scaled)[0]
st.sidebar.write(f"💰 **Montant du prêt prédit** : {predicted_loan_amount:,.2f} ")

# **Prédiction avec le meilleur modèle de classification**
if st.sidebar.button("🔮 Prédire le Statut du Prêt"):
    predicted_status = best_class_instance.predict(input_scaled)[0]
    st.sidebar.write(f"🏦 **Statut du prêt prédit** : {'Approuvé' if predicted_status == 1 else 'Refusé'}")
