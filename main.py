from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import io

# ── Inicializar app ────────────────────────────────────────────────────────────
app = FastAPI(title="API Predicción MNIST")

# ── CORS (permite peticiones desde el frontend) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cargar modelo UNA sola vez ─────────────────────────────────────────────────
try:
    modelo = tf.keras.models.load_model("model/modeloV3.keras")
    print("✅ Modelo cargado correctamente")
except Exception as e:
    raise RuntimeError(f"Error cargando el modelo: {e}")

# Clases 0-9
nombre_clases = [str(i) for i in range(10)]

# ── Ruta raíz ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Modelo MNIST listo y funcionando",
        "version": "1.0",
        "clases": nombre_clases
    }

# ── Ruta de predicción ─────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        image = image.resize((28, 28))
        image = np.array(image) / 255.0

        # Invertir colores (fondo negro, número blanco — igual que MNIST)
        image = 1.0 - image

        # Dar forma correcta para el modelo: (1, 28, 28, 1)
        image = np.expand_dims(image, axis=-1)  # (28,28,1)
        image = np.expand_dims(image, axis=0)   # (1,28,28,1)

        # Predecir
        pred = modelo.predict(image)
        clase_idx = int(np.argmax(pred))
        probabilidad = float(np.max(pred))

        print(f"Predicción: {clase_idx} | Probabilidad: {probabilidad:.4f}")

        return {
            "clase": str(clase_idx),
            "probabilidad": round(probabilidad, 4),
            "todas": {str(i): round(float(pred[0][i]), 4) for i in range(10)}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))