# Análisis de Sentimiento en Reseñas de Videojuegos

Proyecto de **NLP** que procesa el corpus *Amazon Video\_Games\_5* (≈ 230 k reseñas) para entrenar y comparar modelos clásicos de clasificación binaria (positivo ► ≥ 4 ★ / negativo ≤ 3 ★).

---

## Estructura del repositorio

| Carpeta / archivo          | Descripción                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| `notebooks/EDA.ipynb`      | Exploración de datos, limpieza inicial, visualizaciones                 |
| `notebooks/Modelado.ipynb` | Preprocesado, bag‑of‑words, entrenamiento y evaluación                  |
| `src/preprocess.py`        | Función `clean_text()` (tubería completa de normalización y lematizado) |
| `data/`                    | JSON original y subconjuntos procesados (`train.csv`, `test.csv`)       |
| `models/`                  | Modelos entrenados (`logreg.pkl`, `svc_pipeline.pkl`)                   |

---

## Flujo de trabajo resumido

1. **EDA & Limpieza**
   - Filtrado de outliers, inspección de longitud de texto, distribución de `overall`.
   - Análisis de `reviewerID` y `asin` para evitar fugas en el split.
2. **Preprocesado de texto** (`clean_text`)
   - Minúsculas, expansión de contracciones, eliminación de signos, normalización ASCII.
   - Lematizado con *spaCy* y exclusión de *stop‑words* (conservando "no / not").
3. **Vectorización**
   - `CountVectorizer` uni‑ y bi‑gramas, `min_df=5`, `max_df=0.80`.
4. **Entrenamiento & Test**
   - Modelos: Regresión Logística, SVM lineal, LinearSVC + tubería TF‑IDF.
   - Split **GroupShuffleSplit** (80 / 20) estratificado por `asin`.
5. **Búsqueda de hiper‑parámetros**
   - `GridSearchCV` (5 folds estratificados) para `C` y `max_features` en la tubería SVC.
6. **Métricas & Conclusiones**
   - Se reporta **F1‑macro** para mitigar desbalance 75 / 25 %.

---

## Resultados principales

| Modelo                 | Vectorizador  | F1‑macro (TEST) | Comentario breve                           |
| ---------------------- | ------------- | --------------- | ------------------------------------------ |
| LogReg                 | Bow (1‑2 g)   | **0.805**       | Probabilidades útiles para calibrar umbral |
| LinearSVC              | Bow (1‑2 g)   | 0.806           | Mejora recall clase minoritaria            |
| **TF‑IDF + LinearSVC** | TF‑IDF (60 k) | **0.808**       | Mejor CV, regularización *C = 0.5*         |

---

## Requisitos rápidos

```bash
python >= 3.9
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Uso mínimo

```bash
# Entrenar de cero\python notebooks/Modelado.ipynb  # (o ejecutar cells)
# Predecir nuevas reseñas\python src/predict.py "Amazing RPG with great story!"
```

---

## Próximos pasos

- Probar embeddings (*fastText*, *Sentence‑BERT*).
- Ajuste fino de un modelo ligero (*DistilBERT*) vía *Transformers*.
- API REST (*FastAPI*) para inferencia en producción.

---

## Autor

**Nauzet S.** — Proyecto académico para la asignatura *Procesamiento de Lenguaje Natural*.

