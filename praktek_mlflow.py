import mlflow

accuracy_score = 0.95
model_tiruan = "Ini adalah objek model Random Forest"

print("Memulai pencatatan eksperimen ke MLflow...")

with mlflow.start_run():
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    mlflow.log_metric("accuracy", accuracy_score)
    
    mlflow.log_text(model_tiruan, "info_model.txt")

print("Eksperimen berhasil dicatat dengan aman!")