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
# MODELLE LADEN (ANGEPASST AN DEINE GITHUB-DATEINAMEN)
# ==============================================================================
@st.cache_resource
def load_models():
    try:
        # 1. Scaler laden
        with open('scaler_pt.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        # 2. Autoencoder laden
        autoencoder = load_model('fraud_lstm_autoencoder_pt.h5', compile=False)
        
        # 3. Encoder dynamisch aus Autoencoder extrahieren!
        # Wir suchen den Layer direkt vor dem "RepeatVector" (das ist der Bottleneck)
        encoder_output = None
        for layer in autoencoder.layers:
            if isinstance(layer, tf.keras.layers.RepeatVector):
                break
            encoder_output = layer.output
            
        if encoder_output is not None:
            encoder_model = Model(inputs=autoencoder.input, outputs=encoder_output)
        else:
            # Fallback, falls die Architektur anders ist (z.B. die exakte Mitte nehmen)
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
# STREAMLIT UI & VORHERSAGE
# ==============================================================================
st.title("🛡️ Kreditkarten-Betrugserkennung")
st.write("KI-System basierend auf LSTM-Autoencoder und XGBoost.")

# CSS für schöne Status-Anzeigen
st.markdown("""
<style>
.status-ok { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
.status-fraud { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

if scaler and autoencoder and xgb_model:
    expected_features = scaler.n_features_in_
    
    st.write(f"Bitte gib die {expected_features} Features für die Transaktion ein:")
    
    user_input = st.text_input("Features (durch Komma getrennt)", 
                               value=",".join(["0.0"] * expected_features))
    
    if st.button("Transaktion prüfen"):
        try:
            # Eingabe verarbeiten
            input_values = [float(x.strip()) for x in user_input.split(',')]
            
            if len(input_values) != expected_features:
                st.warning(f"Bitte genau {expected_features} Werte eingeben!")
            else:
                # 1. Skalieren
                X_input = np.array(input_values).reshape(1, -1)
                X_scaled = scaler.transform(X_input)
                
                # 2. Umformen für LSTM (Samples, Timesteps, Features)
                X_3d = X_scaled.reshape(1, 1, expected_features)
                
                # 3. Features & MSE berechnen
                bottleneck_feats = encoder_model.predict(X_3d, verbose=0)
                reconstruction = autoencoder.predict(X_3d, verbose=0)
                mse = np.mean(np.power(X_3d - reconstruction, 2), axis=(1, 2)).reshape(-1, 1)
                
                # Bottleneck für XGBoost wieder flach machen (falls 3D)
                if len(bottleneck_feats.shape) == 3:
                    bottleneck_feats = bottleneck_feats.reshape(1, -1)
                
                # 4. Hybrid Feature Vektor bauen
                X_hybrid = np.hstack((X_scaled, bottleneck_feats, mse))
                
                # 5. XGBoost Vorhersage
                fraud_prob = xgb_model.predict_proba(X_hybrid)[0, 1] * 100
                XGB_THRESHOLD = 80.0 
                
                # 6. Anzeige
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Betrugsrisiko", value=f"{fraud_prob:.2f} %")
                    if fraud_prob >= XGB_THRESHOLD:
                        st.markdown("<div class='status-fraud'>🚨 BETRUG!</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='status-ok'>✅ GENEHMIGT</div>", unsafe_allow_html=True)
                        
                with col2:
                    st.write("**Details:**")
                    st.code(f"""
MSE (Fehler): {mse[0][0]:.6f}
Latente Features: {bottleneck_feats.shape[1]}
XGBoost Inputs: {X_hybrid.shape[1]}
                    """, language="text")
                    
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")
