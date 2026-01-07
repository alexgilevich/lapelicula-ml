import os.path
from itertools import pairwise, chain

from databricks.bundles.jobs import (
    Job,
    ForEachTask,
    JobParameterDefinition,
    SparkPythonTask,
    RunIf,
    Task,
    TaskDependency,
    Library,
    PythonPyPiLibrary,
    CronSchedule,
    JobEnvironment,
    Environment,
    QueueSettings,
    JobRunAs
)

RAW_DATA_PATH, RUN_AS_SERVICE_PRINCIPAL_ID = os.environ.get("RAW_DATA_PATH", None), os.environ.get("RUN_AS_SERVICE_PRINCIPAL_ID", None)
assert RAW_DATA_PATH is not None

parameters = [("RAW_DATA_PATH", RAW_DATA_PATH), ("UC_DEFAULT_CATALOG_NAME", "${var.catalog}"), ("UC_DEFAULT_SCHEMA_NAME", "${var.schema}")]
task_parameters = [f"--{key}={val}" for key, val in parameters]
task_files = [
    "step01_extract_raw_data",
    "step02_preprocess_movies",
    "step03_generate_synthetic_users",
    "step04_build_user_features",
    "step05_oversample_ratings",
    "step06_enrich_movies",
    "step07_sync_to_dynamodb"
]
tasks = [
    Task(
        task_key = source_file_cur,
        depends_on = [TaskDependency(task_key = source_file_prev)] if source_file_prev else [],
        spark_python_task=SparkPythonTask(python_file = os.path.join("src/preprocessing/", source_file_cur + ".py"), parameters=task_parameters),
        environment_key="Default",
        disable_auto_optimization=True
    )
    for source_file_prev, source_file_cur in pairwise(chain([None], task_files))
]

preprocessing_job = Job(
    name="lapelicula_preprocessing_job",
    environments = [
        JobEnvironment(
            environment_key="Default",
            spec=Environment(
                environment_version="4", 
                dependencies=[line.strip() for line in open("resources/requirements_preprocessing.txt").readlines() if line.strip()]
            ),
        )
    ],
    tasks = tasks,
    parameters = [JobParameterDefinition(name = key, default = val) for key, val in parameters],
    max_concurrent_runs = 1,
    queue = QueueSettings(enabled = False)
)

if RUN_AS_SERVICE_PRINCIPAL_ID is not None:
    preprocessing_job.run_as = JobRunAs(service_principal_name = RUN_AS_SERVICE_PRINCIPAL_ID)