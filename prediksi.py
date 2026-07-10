import mlflow
import pandas as pd

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

logged_model = 'runs:/64a92ba53c8f4445ab2b9edb35fb0b9c/model'

print("--- Memuat Model dari MLflow ---")

try:
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print("✅ Model Berhasil Dimuat!")
    
    data_baru = [[6.7, 3.1, 5.6, 2.4]]
    df_baru = pd.DataFrame(data_baru)
    
    predict = loaded_model.predict(df_baru)
    
    print("\n🎯 Hasil Prediksi Model:")
    print(f"Kelas Iris Hasil Prediksi: {predict}")
    print("-" * 35)

except Exception as e:
    print(f"❌ Terjadi kesalahan saat prediksi: {e}")