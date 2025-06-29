# Análisis de Sentimiento en Reseñas de Videojuegos

Proyecto de **NLP** que procesa el corpus *Amazon Video\_Games\_5* (≈ 230 k reseñas) para entrenar y comparar modelos clásicos de clasificación binaria (positivo ► ≥ 4 ★ / negativo ≤ 3 ★).

---

## Estructura del repositorio

| Archivo          | Descripción                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| `EDA.ipynb`                | Exploración de datos, limpieza inicial, visualizaciones                 |
| `PROCE.ipynb`              | Preprocesado, bag‑of‑words, entrenamiento y evaluación                  |
| `Video_Games_5.json`       | JSON original                                                           |

---

## Flujo de trabajo resumido

1. **EDA & Limpieza**
   - Filtrado de outliers, inspección de longitud de texto, distribución de `overall`.
   - Análisis de `reviewerID` y `asin` para evitar fugas en el split.
2. **Preprocesado de texto** (`clean_text`)
   - Minúsculas, expansión de contracciones, eliminación de signos, normalización ASCII.
   - Lematizado con *spaCy* y exclusión de *stop‑words* (conservando "no / not").
3. **Vectorización**
   - `CountVectorizer` uni‑ y bi‑gramas,
4. **Entrenamiento & Test**
   - Modelos: Regresión Logística, SVM lineal, LinearSVC + tubería TF‑IDF.
5. **Búsqueda de hiper‑parámetros**
   - `GridSearchCV`
6. **Métricas & Conclusiones**
   - Se reporta **F1‑macro** para mitigar desbalance 75 / 25 %.

---

## Resultados principales

| Modelo                 | Vectorizador  | F1‑macro (TEST) | Comentario breve                           |
| ---------------------- | ------------- | --------------- | ------------------------------------------ |
| LogReg                 | Bow (1‑2 g)   | **0.805**       | Probabilidades útiles para calibrar umbral |
| LinearSVC              | Bow (1‑2 g)   | **0.77**        | Con Posible mejora            |
| **TF‑IDF + LinearSVC** | TF‑IDF (60 k) | **0.808**       | Mejor CV, regularización *C = 0.5*         |

---

## Requisitos rápidos

```bash
python >= 3.9
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Autor
**Nauzet Fernandez Lorenzo** 

