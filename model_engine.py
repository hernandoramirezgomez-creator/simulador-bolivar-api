import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def clean_numeric_series(s):
    if s.dtype == object:
        s_clean = s.astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s_clean.replace(r'^\s*$', np.nan, regex=True), errors='coerce')
    return pd.to_numeric(s, errors='coerce')

class RidgeSimulatorEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.vars_included = {}
        self.data_base = None
        self.matriz_variables = None

    def load_from_json(self, base_data, matriz_data):
        self.data_base = pd.DataFrame(base_data)
        self.matriz_variables = pd.DataFrame(matriz_data)
        self.data_base['PERIODO'] = clean_numeric_series(self.data_base['PERIODO'])
        self.data_base['CREC PRIMAS'] = clean_numeric_series(self.data_base['CREC PRIMAS'])

    def train_all_models(self):
        self.models.clear()
        ramos = self.data_base['RAMO'].dropna().unique()
        metrics_summary = {}

        for ramo in ramos:
            ramo_str = str(ramo).strip()
            if not ramo_str: continue

            ramo_data = self.data_base[self.data_base['RAMO'] == ramo_str].copy()
            ramo_matriz = self.matriz_variables[(self.matriz_variables['RAMO'] == ramo_str) & (self.matriz_variables['INCLUIR'] == 'SI')]
            variables_incluir = ramo_matriz['DES_VARIABLE'].tolist()

            if not variables_incluir: continue

            X = ramo_data[variables_incluir].copy()
            for col in X.columns:
                X[col] = clean_numeric_series(X[col])
            
            y = clean_numeric_series(ramo_data['CREC PRIMAS'].copy())

            X = X[y.notna()]
            y = y.dropna()

            if len(X) < 2: continue

            # --- TU LÓGICA EXACTA DE COLAB ---
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            n_splits = min(5, len(X_train))
            if n_splits < 2: continue

            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=cv)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                ridge_model.fit(X_train_scaled, y_train)

                y_pred_train = ridge_model.predict(X_train_scaled)
                r2_tr = r2_score(y_train, y_pred_train)

            # Guardar en memoria RAM
            self.models[ramo_str] = ridge_model
            self.scalers[ramo_str] = scaler
            self.vars_included[ramo_str] = variables_incluir

            metrics_summary[ramo_str] = {"r2_train": float(r2_tr)}

        return metrics_summary

    def predict_ramo(self, ramo, macro_inputs):
        ramo = str(ramo).strip()
        if ramo not in self.models: return 0.0
        
        model = self.models[ramo]
        scaler = self.scalers[ramo]
        vars_inc = self.vars_included[ramo]

        input_row = [float(macro_inputs.get(v, 0.0)) for v in vars_inc]
        X_df = pd.DataFrame([input_row], columns=vars_inc)
        X_scaled = scaler.transform(X_df)
        return float(model.predict(X_scaled)[0])