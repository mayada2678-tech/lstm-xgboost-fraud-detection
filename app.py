# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:07:37 2026

@author: mayad
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from xgboost import XGBClassifier

# ==============================================================================
# 1. SEITEN-KONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="LSTM Autoencoder with XGBoost",
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
# 2. MODELLE LADEN (Caching für Performance)
# ==============================================================================
@st.cache_resource
def load_models():
    # 1. Scaler (PowerTransformer)
    with open("scaler_pt.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    # 2. LSTM Autoencoder
   import keras

# Workaround für den Keras-Versionskonflikt (input_axes Fehler)
class SafeGlorotUniform(keras.initializers.GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed)

with keras.saving.custom_object_scope({'GlorotUniform': SafeGlorotUniform}):
    autoencoder = tf.keras.models.load_model("fraud_lstm_autoencoder_pt.h5", compile=False)
    
    # 3. Encoder Bottleneck extrahieren (die 32 Latent Features)
    encoder_model = Model(
        inputs=autoencoder.input, 
        outputs=autoencoder.get_layer("encoder_bottleneck").output
    )
    
    # 4. XGBoost Modell
    xgb_model = XGBClassifier()
    xgb_model.load_model("xgb_fraud_model.json")
    
    return scaler, autoencoder, encoder_model, xgb_model

try:
    scaler, autoencoder, encoder_model, xgb_model = load_models()
except Exception as e:
    st.error(f"⚠️ Fehler beim Laden der Modelle. Sind alle Dateien im Ordner? Details: {e}")
    st.stop()

# ==============================================================================
# 3. UI-AUFBAU: EINGABEBEREICH
# ==============================================================================
st.title("🛡️ LSTM Autoencoder with XGBoost")
st.write("Hybrides KI-Modell zur Echtzeit-Erkennung von Kreditkartenbetrug.")

with st.sidebar:
    st.header("Transaktionsdaten")
    amount = st.number_input("Betrag (€)", min_value=0.0, value=250.0, step=10.0)
    delta_time = st.number_input("Sekunden seit letzter TX", min_value=0.0, value=45.0, step=1.0)
    tx_count_1h = st.slider("Anzahl TX (letzte 1h)", 1, 50, 2)
    
    st.divider()
    st.write("*(Hinweis: Für die restlichen V-Features werden für diesen Demo-Test Durchschnittswerte angenommen)*")
    submit_btn = st.button("Transaktion prüfen", type="primary", use_container_width=True)

# ==============================================================================
# 4. VORHERSAGE-PIPELINE (Wenn Button geklickt)
# ==============================================================================
if submit_btn:
    st.subheader("Analyse-Ergebnis")
    
    # 1. Dummy Array mit der korrekten Feature-Anzahl erstellen
    expected_features = scaler.n_features_in_
    X_input = np.zeros((1, expected_features))
    
    # Feature Engineering (wie in deinem Trainingsskript)
    X_input[0, 0] = np.log1p(amount) # Amount_log (angenommen dies ist Index 0)
    X_input[0, 1] = delta_time       # delta_time (angenommen Index 1)
    X_input[0, 2] = tx_count_1h      # tx_count_1h (angenommen Index 2)
    # Die restlichen Indizes bleiben 0 (als Baseline für die V-Features)
    
    # 2. Skalierung (Yeo-Johnson)
    X_scaled = scaler.transform(X_input)
    
    # 3. Reshape für LSTM (3D: Samples, Timesteps, Features)
    X_3d = X_scaled.reshape(1, 1, expected_features)
    
    # 4. LSTM Bottleneck Features & MSE berechnen
    bottleneck_feats = encoder_model.predict(X_3d, verbose=0)
    reconstruction = autoencoder.predict(X_3d, verbose=0)
    mse = np.mean(np.power(X_3d - reconstruction, 2), axis=(1, 2)).reshape(-1, 1)
    
    # 5. Hybrid Feature Vektor bauen (Skaliert + Bottleneck + MSE)
    X_hybrid = np.hstack((X_scaled, bottleneck_feats, mse))
    
    # 6. XGBoost Prediction
    fraud_prob = xgb_model.predict_proba(X_hybrid)[0, 1] * 100
    
    # Schwellenwert (Setze hier den optimalen Threshold aus deinem Training ein, z.B. 0.80 -> 80%)
    XGB_THRESHOLD = 80.0 
    
    # 7. Ergebnisse anzeigen
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
1. Original Features: {expected_features} (Log Amount, Delta Time, etc.)
2. Scaler: Yeo-Johnson PowerTransformer
3. LSTM Autoencoder MSE: {mse[0][0]:.6f}
4. Extrahiert: 32 Latent Bottleneck Features
5. XGBoost Input Features: {X_hybrid.shape[1]} (Kombiniert)
        """, language="text")
