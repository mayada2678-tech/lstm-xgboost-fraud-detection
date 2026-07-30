# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from xgboost import XGBClassifier
import keras

# ==============================================================================
# SEITEN-KONFIGURATION (Wie früher: Breit und übersichtlich)
# ==============================================================================
st.set_page_config(
    page_title="Betrugsdetektor: CSV-Prüfung",
    page_icon="💳",
    layout="wide"
)

# ==============================================================================
# WORKAROUND: MONKEY-PATCH FÜR KERAS INKOMPATIBILITÄTEN (CLOUD FIX)
# ==============================================================================
original_glorot_init = keras.initializers.GlorotUniform.__init__
def patched_glorot_init(self, seed=None, **kwargs):
    original_glorot_init(self, seed=seed)
keras.initializers.GlorotUniform.__init__ = patched_glorot_init
tf.keras.initializers.GlorotUniform.__init__ = patched_glorot_init

original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, units, **kwargs):
    kwargs.pop('quantization_config', None) 
    original_dense_init(self, units=units, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init
tf.keras.layers.Dense.__init__ = patched_dense_init

# ==============================================================================
# MODELLE LADEN 
# ==============================================================================
@st.cache_resource
def load_models():
    try:
        # 1. Scaler laden
        with open('scaler_pt.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # 2. Autoencoder laden (HIER IST DER FIX: compile=False)
        autoencoder = load_model('fraud_lstm_autoencoder_pt.h5', compile=False)
        
        # 3. Encoder dynamisch aus Autoencoder extrahieren!
        encoder_output = None
        for layer in autoencoder.layers:
            if isinstance(layer, tf.keras.layers.RepeatVector):
                break
            encoder_output = layer.output
            
        if encoder_output is not None:
            encoder_model = Model(inputs=autoencoder.input, outputs=encoder_output)
        else:
            # Fallback
            bottleneck_index = len(autoencoder.layers) // 2 - 1
            encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[bottleneck_index].output)
        
        # 4. XGBoost laden
        xgb_model = XGBClassifier()
        xgb_model.load_model('xgb_fraud_model.json')
        
        return scaler, autoencoder, encoder_model, xgb_model
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Modelle. Details: {e}")
        return None, None, None, None

scaler, autoencoder, encoder_model, xgb_model = load_models()

# ==============================================================================
# STREAMLIT UI & VORHERSAGE (DASHBOARD-ANSICHT)
# ==============================================================================
st.title("💳 Kreditkarten-Betrugserkennung: CSV-Prüfung")
st.write("Lade eine CSV-Datei mit Transaktionsdaten (Testdaten) hoch, um sie in Echtzeit auf Betrug zu analysieren.")
st.write("---")

if scaler and autoencoder and xgb_model:
    expected_features = scaler.n_features_in_
    
    # Datei-Upload statt Texteingabe
    uploaded_file = st.file_uploader(f"Transaktionsdaten hochladen (CSV mit {expected_features} Spalten)", type=["csv"])
    
    if uploaded_file is not None:
        # Daten einlesen
        df = pd.read_csv(uploaded_file)
        
        st.write("### Vorschau der hochgeladenen Daten:")
        st.dataframe(df.head())
        
        if st.button("🚨 Alle Transaktionen prüfen"):
            with st.spinner('Analysiere Transaktionen durch KI...'):
                try:
                    # Prüfen, ob die Anzahl der Features stimmt
                    if df.shape[1] != expected_features:
                        st.error(f"Fehler: Das Modell erwartet genau {expected_features} Features, aber die CSV hat {df.shape[1]}.")
                    else:
                        # 1. Skalieren
                        X_scaled = scaler.transform(df.values)
                        
                        # 2. Reshape für LSTM (Samples, 1, Features)
                        X_3d = X_scaled.reshape(X_scaled.shape[0], 1, expected_features)
                        
                        # 3. Encoder Bottleneck Features & MSE extrahieren
                        bottleneck_feats = encoder_model.predict(X_3d, verbose=0)
                        if len(bottleneck_feats.shape) == 3:
                            bottleneck_feats = bottleneck_feats.reshape(bottleneck_feats.shape[0], -1)
                            
                        reconstruction = autoencoder.predict(X_3d, verbose=0)
                        mse = np.mean(np.power(X_3d - reconstruction, 2), axis=(1, 2)).reshape(-1, 1)
                        
                        # 4. Hybrid Feature Vektor bauen
                        X_hybrid = np.hstack((X_scaled, bottleneck_feats, mse))
                        
                        # 5. XGBoost Vorhersage
                        fraud_probs = xgb_model.predict_proba(X_hybrid)[:, 1] * 100
                        XGB_THRESHOLD = 80.0 
                        
                        # 6. Ergebnisse in DataFrame einfügen
                        df_results = df.copy()
                        # Wir fügen die Ergebnisse als erste Spalten ein, damit man sie sofort sieht
                        df_results.insert(0, 'Status', np.where(fraud_probs >= XGB_THRESHOLD, '🚨 BETRUG', '✅ OK'))
                        df_results.insert(1, 'Betrugsrisiko (%)', fraud_probs.round(2))
                        df_results.insert(2, 'MSE (Abweichung)', mse.round(4))
                        
                        st.success("Analyse erfolgreich abgeschlossen!")
                        
                        # Zeige Ergebnisse in einer schönen Tabelle
                        st.write("### 📊 Analyse-Ergebnisse")
                        
                        # Tabelle stylen: Betrugszeilen rot markieren
                        def style_fraud(row):
                            if row['Status'] == '🚨 BETRUG':
                                return ['background-color: #ffcccc'] * len(row)
                            return [''] * len(row)
                            
                        st.dataframe(df_results.style.apply(style_fraud, axis=1), height=500)
                        
                except Exception as e:
                    st.error(f"Fehler bei der Batch-Vorhersage: {e}")
