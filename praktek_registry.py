import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

model_uri = "models:/iris_classifier/Staging"

print("--- Memuat Model dari Model Registry (Stage: Staging) ---")

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model Staging Berhasil Dimuat tanpa Run ID!")
    
    input_data = pd.DataFrame([[6.7, 3.1, 5.6, 2.4]])
    predictions = model.predict(input_data)
    
    print("\n🎯 Hasil Prediksi Stage Staging:")
    print(f"Predictions: {predictions}")
    print("-" * 40)

except Exception as e:
    print(f"❌ Gagal memuat model registry: {e}")
    print("💡 Pastikan Anda sudah mematikan toggle 'New model registry UI', mengubah Stage menjadi 'Staging', dan menyimpan filenya.")