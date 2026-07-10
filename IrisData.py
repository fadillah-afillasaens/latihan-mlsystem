import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
mlflow.set_experiment("Latihan_Validasi_Iris")

data = load_iris()
logdata = mlflow.data.from_numpy(data.data, source="data.csv")
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
input_example = X_train[0:5]
with mlflow.start_run():
    n_estimators = random.randint(0,1000)
    max_depth = random.randint(1,50)
    mlflow.autolog()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)