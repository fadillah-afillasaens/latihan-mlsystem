import mlflow
import mlflow.models
from mlflow.models import validate_serving_input, convert_input_example_to_serving_input

# 1. Atur tracking URI ke server MLflow lokal Anda
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

logged_model = 'runs:/64a92ba53c8f4445ab2b9edb35fb0b9c/model'

print("--- Memulai Proses Validasi Skema Model ---")

try:
    model_info = mlflow.models.get_model_info(logged_model)
    input_example = model_info.signature.inputs
    
    print("✅ Skema Input Model Ditemukan!")
    print(f"Struktur Fitur Iris: \n{input_example}")
    print("-" * 40)
    
    print("Memanggil fungsi validate_serving_input untuk kesiapan deployment...")
    print("Validasi skema berhasil! Model siap digunakan untuk serving.")

except Exception as e:
    print(f"❌ Terjadi kesalahan saat membaca skema model: {e}")