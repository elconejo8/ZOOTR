import mlflow
import subprocess
import webbrowser
import time

def start_mlflow(experiment_name, open_ui=True):

    mlflow.set_tracking_uri("file:./Tracking")
    #File based tracking
    mlflow.set_experiment(experiment_name)
    
    if open_ui:
        subprocess.Popen(
            ['mlflow', 'ui', '--backend-store-uri', 'file:Tracking'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        webbrowser.open("http://localhost:5000")
