import mlflow

print("Menyiapkan informasi metadata dataset...")

dataset_version = "v1.0"
dataset_path = "/data/dataset.csv"

with mlflow.start_run():
    print("Mencatat versi dan lokasi dataset ke folder params MLflow...")
    
    mlflow.log_param("dataset_version", dataset_version)
    mlflow.log_param("dataset_path", dataset_path)

print("Selesai! Informasi dataset berhasil dicatat sebagai parameter.")