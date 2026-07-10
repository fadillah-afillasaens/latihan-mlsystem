import mlflow
import mlflow.models
from mlflow.models import validate_serving_input, convert_input_example_to_serving_input
import numpy as np

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

logged_model = 'runs:/64a92ba53c8f4445ab2b9edb35fb0b9c/model'

print("--- Memulai Proses Validasi & Konversi Payload ---")

try:
    INPUT_EXAMPLE = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2]
    ])
    
    serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)
    
    print("\n🚀 Hasil Konversi ke Serving Payload (Format REST API):")
    print(serving_payload)
    print("-" * 50)
    
    validate_serving_input(logged_model, serving_payload)
    print("✅ Validasi Sukses! Payload sesuai dengan kebutuhan REST API model.")

except Exception as e:
    print(f"❌ Terjadi kesalahan saat validasi: {e}")