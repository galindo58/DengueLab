from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data.xlsx')
app = Flask(__name__)

# La variable objetivo del archivo es la primera columna: grupos codificados 1 y 2.
# Se excluyen columnas de tratamientos porque pueden ocurrir después de la evaluación clínica.
TARGET = 'GROUPS: DENGUE FEVER OR COMPLICATED DENGUE'
EXCLUDED = ['IV FLUIDS', 'ANTIBIOTICS', 'ANTIPYRETICS', 'STEROIDS']

def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    target = next((c for c in df.columns if c.upper().startswith('GROUPS:')), df.columns[0])
    df = df.dropna(subset=[target]).copy()
    y = pd.to_numeric(df[target], errors='coerce')
    valid = y.notna()
    df, y = df.loc[valid], y.loc[valid].astype(int)
    feature_cols = [c for c in df.columns if c != target and c.upper() not in EXCLUDED]
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    # Variables sin ningún dato útil no pueden entrenarse.
    feature_cols = [c for c in feature_cols if X[c].notna().any()]
    X = X[feature_cols]
    return df, X, y, target

DF, X, y, TARGET_NAME = load_data()
classes = sorted(y.unique().tolist())
if len(classes) != 2:
    raise ValueError('El dataset debe tener exactamente dos clases para este proyecto.')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
models = {
    'Regresión logística': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ]),
    'Red neuronal artificial': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
                                alpha=0.001, max_iter=1800, early_stopping=True,
                                random_state=42))
    ])
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(classes[1])]
    cm = confusion_matrix(y_test, pred, labels=classes)
    try: auc = roc_auc_score((y_test == classes[1]).astype(int), proba)
    except Exception: auc = None
    results[name] = {
        'accuracy': round(accuracy_score(y_test, pred), 4),
        'precision': round(precision_score(y_test, pred, pos_label=classes[1], zero_division=0), 4),
        'recall': round(recall_score(y_test, pred, pos_label=classes[1], zero_division=0), 4),
        'f1': round(f1_score(y_test, pred, pos_label=classes[1], zero_division=0), 4),
        'auc': round(float(auc), 4) if auc is not None else None,
        'matrix': cm.tolist(),
        'predictions': pred.tolist()
    }

# Valores de referencia para completar el formulario de predicción.
def feature_metadata():
    out=[]
    for c in X.columns:
        vals = X[c].dropna()
        median = float(vals.median()) if len(vals) else 0.0
        unique = sorted(vals.unique().tolist())
        is_categorical = len(unique) <= 12 and all(float(v).is_integer() for v in unique)
        out.append({'name': c, 'default': median, 'categorical': is_categorical,
                    'options': [float(v) for v in unique] if is_categorical else []})
    return out
FEATURES = feature_metadata()

@app.route('/')
def home():
    return render_template('index.html', total=len(DF), feature_count=len(X.columns),
                           class_counts={str(k): int((y==k).sum()) for k in classes},
                           features=FEATURES, target=TARGET_NAME)

@app.route('/api/metrics')
def metrics():
    return jsonify({'results': results, 'classes': classes, 'test_size': len(y_test),
                    'train_size': len(y_train), 'total': len(DF), 'features': len(X.columns),
                    'feature_names': list(X.columns)})

@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True) or {}
    values = payload.get('values', {})
    row = {}
    for f in FEATURES:
        try: row[f['name']] = float(values.get(f['name'], f['default']))
        except (TypeError, ValueError): row[f['name']] = f['default']
    sample = pd.DataFrame([row], columns=X.columns)
    response = {}
    for name, model in models.items():
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        response[name] = {'class': prediction,
                          'probabilities': {str(c): round(float(p), 4) for c,p in zip(model.classes_, probabilities)}}
    return jsonify({'predictions': response, 'note': 'Resultado experimental para fines académicos; no es un diagnóstico médico.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
