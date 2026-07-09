import numpy as np
import mlflow

print("Mempersiapkan data dan membuat objek dataset MLflow...")

array = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

dataset = mlflow.data.from_numpy(array, source="data.csv")

with mlflow.start_run():
    print("Mencatat informasi dataset ke MLflow...")
    
    mlflow.log_input(dataset, context="training")

print("Pencatatan selesai! Silakan cek folder mlruns Anda.")