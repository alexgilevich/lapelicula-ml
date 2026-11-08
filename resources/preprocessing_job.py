import os.path
from itertools import chain, pairwise

from databricks.bundles.jobs import (
    Environment,
    Job,
    JobEnvironment,
    JobParameterDefinition,
    JobRunAs,
    Library,
    PythonWheelTask,
    QueueSettings,
    Task,
    TaskDependency,
)

RAW_DATA_PATH, RUN_AS_SERVICE_PRINCIPAL_ID = (
    os.environ.get("RAW_DATA_PATH", None),
    os.environ.get("RUN_AS_SERVICE_PRINCIPAL_ID", None),
)
assert RAW_DATA_PATH is not None

parameters = [
    ("RAW_DATA_PATH", RAW_DATA_PATH),
    ("UC_DEFAULT_CATALOG_NAME", "${var.catalog}"),
    ("UC_DEFAULT_SCHEMA_NAME", "${var.schema}"),
    ("MOVIE_API_URL", "${var.movie_api_url}"),
]
task_parameters = [f"--{key}={val}" for key, val in parameters]
task_files = [
    "step01_extract_raw_data",
    "step02_preprocess_movies",
    "step03_generate_synthetic_users",
    "step04_build_user_features",
    "step05_oversample_ratings",
    "step06_enrich_movies",
    "step07_generate_embeddings",
    "step08_detect_sentiment",
    "step09_save_gold_movies",
    "step10_sync_to_api",
]
tasks = [
    Task(
        task_key=source_file_cur,
        depends_on=[TaskDependency(task_key=source_file_prev)] if source_file_prev else [],
        python_wheel_task=PythonWheelTask(
            package_name="lapelicula",
            entry_point=source_file_cur,
            parameters=task_parameters,
        ),
        libraries=[Library(whl="dist/*.whl")],
        environment_key="Default",
        disable_auto_optimization=True,
    )
    for source_file_prev, source_file_cur in pairwise(chain([None], task_files))
]

preprocessing_job = Job(
    name="lapelicula_preprocessing_job",
    environments=[
        JobEnvironment(
            environment_key="Default",
            spec=Environment(
                environment_version="4",
                dependencies=[
                    line.strip() for line in open("resources/requirements_preprocessing.txt").readlines() if line.strip()
                ],
            ),
        )
    ],
    tasks=tasks,
    parameters=[JobParameterDefinition(name=key, default=val) for key, val in parameters],
    max_concurrent_runs=1,
    queue=QueueSettings(enabled=False),
)

if RUN_AS_SERVICE_PRINCIPAL_ID is not None:
    preprocessing_job.run_as = JobRunAs(service_principal_name=RUN_AS_SERVICE_PRINCIPAL_ID)
