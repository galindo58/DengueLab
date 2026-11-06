# DengueLab — analítica predictiva de dengue pediátrico

Aplicación web académica desarrollada con Flask y scikit-learn. Entrena y compara dos modelos reales usando el dataset `data.xlsx` incluido:

- Regresión logística.
- Red neuronal artificial multicapa (MLP).

## Funciones
- Métricas de evaluación sobre un conjunto de prueba independiente: exactitud, precisión, sensibilidad, F1 y ROC-AUC.
- Matrices de confusión.
- Formulario para comparar predicciones de ambos modelos.
- Resumen del conjunto de datos y de las variables.

## Publicar en Render
- Runtime: Python 3.
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

La aplicación lee `data.xlsx` desde la misma carpeta que `app.py`.

## Ejecutar localmente
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Abre `http://127.0.0.1:5000`.

## Método y advertencias
Se realiza una partición estratificada 75/25 con semilla fija. Los valores faltantes se imputan con la mediana y las variables se estandarizan dentro de cada pipeline. Se excluyen las columnas de tratamientos (`IV FLUIDS`, `ANTIBIOTICS`, `ANTIPYRETICS`, `STEROIDS`) para reducir el riesgo de fuga de información.

La correspondencia de clases 1 (dengue fever/no complicado) y 2 (dengue complicado) debe confirmarse con el diccionario/README original del dataset. La interpretación de los grupos no debe considerarse validación clínica.

**Uso exclusivamente académico y experimental.** No es un dispositivo médico, no diagnostica y no debe utilizarse para triage ni decisiones de tratamiento. Las métricas describen este dataset y esta partición, no el desempeño clínico en otros hospitales o poblaciones.

Fuente científica: Bhaskar M, Mahalingam S, M M H, Achappa B. *Predictive scoring system for risk of complications in pediatric dengue infection*. F1000Research. 2022;11:446. DOI: 10.12688/f1000research.111214.1.
