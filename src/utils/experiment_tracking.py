import mlflow
import os
import logging
from typing import Dict, Any, List

class ExperimentTracker:
    """A wrapper for MLflow to standardize experiment tracking."""

    def __init__(self, experiment_name: str, tracking_uri: str = None):
        """
        Initializes the tracker and sets the experiment.
        
        Args:
            experiment_name (str): The name of the experiment.
            tracking_uri (str, optional): The URI for the MLflow tracking server. 
                                          Defaults to a local 'mlruns' directory.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.run = None
        logging.info(f"MLflow experiment '{self.experiment_name}' is active.")

    def start_run(self, run_name: str = None) -> None:
        """Starts a new MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)
        logging.info(f"Started MLflow run: {run_name or self.run.info.run_id}")

    def end_run(self) -> None:
        """Ends the active MLflow run."""
        if self.run:
            mlflow.end_run()
            logging.info("Ended MLflow run.")
            self.run = None

    def log_param(self, key: str, value: Any) -> None:
        """Logs a single parameter."""
        mlflow.log_param(key, value)
        logging.info(f"Logged param: {key}={value}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs a dictionary of parameters."""
        mlflow.log_params(params)
        logging.info(f"Logged params: {params}")

    def log_metric(self, key: str, value: float) -> None:
        """Logs a single metric."""
        mlflow.log_metric(key, value)
        logging.info(f"Logged metric: {key}={value}")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs a dictionary of metrics."""
        mlflow.log_metrics(metrics)
        logging.info(f"Logged metrics: {metrics}")

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Logs a local file or directory as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)
        logging.info(f"Logged artifact from {local_path}")

    def log_figure(self, figure, artifact_file: str) -> None:
        """Logs a matplotlib or plotly figure."""
        mlflow.log_figure(figure, artifact_file)
        logging.info(f"Logged figure to {artifact_file}")

    def log_dict(self, dictionary: Dict, artifact_file: str) -> None:
        """Logs a dictionary as a JSON or YAML file."""
        mlflow.log_dict(dictionary, artifact_file)
        logging.info(f"Logged dictionary to {artifact_file}")

    def set_tag(self, key: str, value: Any) -> None:
        """Sets a tag for the current run."""
        mlflow.set_tag(key, value)

    def __enter__(self):
        """Allows using the tracker with a 'with' statement."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures the run is ended when exiting a 'with' block."""
        self.end_run()


def get_tracker(experiment_name: str = "Default Experiment") -> ExperimentTracker:
    """
    Factory function to get an experiment tracker instance.
    
    Args:
        experiment_name (str): The name for the experiment.
        
    Returns:
        ExperimentTracker: An instance of the tracker.
    """
    # Set tracking URI to a local directory
    # Assumes the script is run from the project root or similar context
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    tracking_uri = "file:" + os.path.join(project_root, "mlruns")
    return ExperimentTracker(experiment_name, tracking_uri)

if __name__ == '__main__':
    # Example usage:
    logging.info("Running ExperimentTracker example...")

    # Using the factory function
    tracker = get_tracker("My Example Experiment")
    
    # Using 'with' statement for a run
    with tracker:
        tracker.log_param("learning_rate", 0.01)
        tracker.log_metric("accuracy", 0.95)
        
        # Create a dummy file to log as an artifact
        with open("dummy_artifact.txt", "w") as f:
            f.write("This is a test artifact.")
        
        tracker.log_artifact("dummy_artifact.txt")
        os.remove("dummy_artifact.txt")

    print("\nExample run completed. Check the 'mlruns' directory.")
