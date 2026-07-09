import hashlib
import mlflow
import os

def generate_dataset_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

dataset_path = "Telco-Customer-Churn.csv"
with open(dataset_path, "w") as f:
    f.write("customerID,gender,Churn\n1,Female,No\n2,Male,Yes\n")

print("Menghitung hash unik (SHA-256) dari dataset...")
dataset_hash = generate_dataset_hash(dataset_path)
print(f"Hash Terbentuk: {dataset_hash}")

with mlflow.start_run():
    print("Mencatat kode hash ke dalam parameter MLflow...")
    mlflow.log_param("dataset_hash", dataset_hash)

print("Selesai! Hash unik dataset berhasil dilacak dengan aman.")

if os.path.exists(dataset_path):
    os.remove(dataset_path)