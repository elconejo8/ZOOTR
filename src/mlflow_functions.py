import subprocess
import time
import webbrowser
import mlflow


def start_mlflow(experiment_name):
    subprocess.Popen(
        ['mlflow', 'ui', '--backend-store-uri', 'file:Tracking'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(3)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    webbrowser.open("http://localhost:5000")
    
