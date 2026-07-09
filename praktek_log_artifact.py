import mlflow
import os

dataset_name = "Telco-Customer-Churn.csv"
with open(dataset_name, "w") as f:
    f.write("customerID,gender,Churn\n")
    f.write("1,Female,No\n")
    f.write("2,Male,Yes\n")

print(f"File {dataset_name} berhasil dibuat di komputer lokal.")

with mlflow.start_run():
    print("Mengunggah file CSV sepenuhnya ke dalam artefak MLflow...")
    
    mlflow.log_artifact(dataset_name, artifact_path="datasets")

print("Selesai! Seluruh file fisik dataset kini tersimpan aman di artefak MLflow.")

if os.path.exists(dataset_name):
    os.remove(dataset_name)