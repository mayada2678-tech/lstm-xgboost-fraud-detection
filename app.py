# -*- coding: utf-8 -*-
"""
Vollständige Streamlit App mit LSTM Autoencoder & XGBoost Hybrid-Modell
inklusive Keras-3-Workaround für die Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import keras

# ==============================================================================
# WORKAROUND: MONKEY-PATCH FÜR KERAS INKOMPATIBILITÄTEN (CLOUD FIX)
# Zwingt alte Keras-Versionen in der Cloud, neue Keras-3-Parameter zu ignorieren
# ==============================================================================

# 1. Patch für GlorotUniform ('input_axes' Fehler)
original_glorot_init = keras.initializers.GlorotUniform.__init__
def patched_glorot_init(self, seed=None, **kwargs):
    original_glorot_init(self, seed=seed)
keras.initializers.GlorotUniform.__init__ = patched_glorot_init
tf.keras.initializers.GlorotUniform.__init__ = patched_glorot_init

# 2. Patch für den Dense-Layer ('quantization_config' Fehler)
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, units, **kwargs):
    kwargs.pop('quantization_config', None) # Ignoriere diesen Parameter!
    original_dense_init(self, units=units, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init
tf.keras.layers.Dense.__init__ = patched_dense_init

# 3. Patch für den LSTM-Layer (Vorsichtshalber)
original_lstm_init = keras.layers.LSTM.__init__
def patched_lstm_init(self, units, **kwargs):
    kwargs.pop('quantization_config', None)
    original_lstm_init(self, units=units, **kwargs)
keras.layers.LSTM.__init__ = patched_lstm_init
tf.keras.layers.LSTM.__init__ = patched_lstm_init
# ==============================================================================


# ==============================================================================
# 1. SEITEN-KONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Fraud Detection - Hybrid AI",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #262730; }
    .status-fraud { background-color: #ff4b4b1a; border: 1px solid #ff4b4b; padding: 15px; border-radius: 8px; font-weight: bold; color: #ff4b4b; text-align: center; font-size: 1.2rem; }
    .status-ok { background-color: #00c8531a; border: 1px solid #00c853; padding: 15px; border-radius: 8px; font-weight: bold; color: #00c853; text-align: center; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODELLE LADEN (Cachen, damit es nur einmal geladen wird)
# ==============================================================================
@st.cache_resource
def load_all_models():
    try:
        # 1. Scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        # 2. XGBoost
        with open("xgboost_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
            
        # 3. Keras Modelle (hier greift unser Workaround von oben!)
        autoencoder = load_model("autoencoder.h5", compile=False)
        encoder_model = load_model("encoder.h5", compile=False)
        
        return scaler, xgb_model, autoencoder, encoder_model
    except Exception as e:
        st.error(f"Fehler beim Laden der Modelle. Sind alle Dateien im Ordner? Details: {e}")
        return None, None, None, None

scaler, xgb_model, autoencoder, encoder_model = load_all_models()


# ==============================================================================
# 3. BENUTZEROBERFLÄCHE (EINGABE)
# ==============================================================================
st.title("🛡️ Kreditkarten-Betrugserkennung")
st.write("KI-System basierend auf LSTM-Autoencoder und XGBoost.")

if scaler is not None:
    st.sidebar.header("Transaktionsdaten eingeben")
    
    # Beispielhafte Eingabefelder (dynamisch)
    amount = st.sidebar.number_input("Transaktionsbetrag (€)", min_value=0.0, value=250.0, step=10.0)
    delta_time = st.sidebar.number_input("Sekunden seit letzter TX", min_value=0.0, value=45.0, step=1.0)
    tx_count_1h = st.sidebar.slider("Anzahl TX (letzte 1h)", 1, 50, 2)
    
    st.sidebar.divider()
    st.sidebar.info("Hinweis: Für die restlichen V-Features werden für diesen Demo-Test Null- oder Durchschnittswerte angenommen.")
    
    submit_btn = st.sidebar.button("Transaktion prüfen", type="primary", use_container_width=True)


# ==============================================================================
# 4. VORHERSAGE-PIPELINE (Aktivierung per Button)
# ==============================================================================
    if submit_btn:
        st.subheader("Analyse-Ergebnis")
        
        # Leeres Array für alle erwarteten Features des Scalers erstellen
        expected_features = scaler.n_features_in_
        X_input = np.zeros((1, expected_features))
        
        # Die eingegebenen Werte zuweisen (Beispiel-Index-Mapping)
        X_input[0, 0] = np.log1p(amount)
        X_input[0, 1] = delta_time
        X_input[0, 2] = tx_count_1h
        
        # 1. Daten skalieren
        X_scaled = scaler.transform(X_input)
        
        # 2. Für LSTM umformen (3D: Samples, Timesteps, Features)
        X_3d = X_scaled.reshape(1, 1, expected_features)
        
        # 3. LSTM Bottleneck Features & Rekonstruktionsfehler (MSE) berechnen
        bottleneck_feats = encoder_model.predict(X_3d, verbose=0)
        reconstruction = autoencoder.predict(X_3d, verbose=0)
        mse = np.mean(np.power(X_3d - reconstruction, 2), axis=(1, 2)).reshape(-1, 1)
        
        # 4. Hybrid Feature Vektor bauen (Skalierte Daten + Bottleneck + MSE)
        X_hybrid = np.hstack((X_scaled, bottleneck_feats, mse))
        
        # 5. XGBoost Vorhersage
        fraud_prob = xgb_model.predict_proba(X_hybrid)[0, 1] * 100
        
        # Schwellenwert
        XGB_THRESHOLD = 80.0 
        
        # 6. Ergebnisse im Dashboard anzeigen
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Betrugswahrscheinlichkeit (XGBoost)", value=f"{fraud_prob:.2f} %")
            if fraud_prob >= XGB_THRESHOLD:
                st.markdown("<div class='status-fraud'>🚨 BLOCKIERT (BETRUG)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-ok'>✅ GENEHMIGT</div>", unsafe_allow_html=True)
                
        with col2:
            st.write("**Technische Details der Hybrid-Pipeline:**")
            st.code(f"""
1. Original Features: {expected_features} Input-Variablen
2. LSTM Autoencoder MSE: {mse[0][0]:.6f} (Rekonstruktionsfehler)
3. Extrahiert: {bottleneck_feats.shape[1]} Latent Bottleneck Features
4. XGBoost Input Features: {X_hybrid.shape[1]} (Kompletter Hybrid-Vektor)
            """, language="text")
