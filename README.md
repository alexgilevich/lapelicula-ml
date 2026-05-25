<h1 align="center">
  <br>
  <a href="https://lapelicula.net/"><img width="200" height="200" alt="watching-a-movie" src="https://github.com/user-attachments/assets/827d93cb-6ea6-4723-9801-a71aaba12432" /></a>
  <br>
  Project La Pelicula
  <br>
</h1>


<h4 align="center">Movie recommender system on top of Python and .NET</h4>

<p align="center">
  <a href="https://github.com/alexgilevich/lapelicula-ui/actions/workflows/aws.yml">
    <img src="https://github.com/alexgilevich/lapelicula-ui/actions/workflows/aws.yml/badge.svg"
         alt="CI/CD">
  </a>
</p>

<p align="center">
  <a href="https://lapelicula.net/">Try it out</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#how-to-build-locally">How To Build Locally</a> •
  <a href="#credits">Credits</a> •
  <a href="#attribution">Attribution</a> •
  <a href="#license">License</a> •
  <a href="#support">Support me</a>
</p>

<p align="center">
  This is the ML repo. The ML code found here is included as a git submodule in the <a href="https://github.com/alexgilevich/lapelicula-ui" target="_blank">UI repo</a>. The API called from the .NET backend is the functions from the `recommnedation_system` module.
</p>

<p align="center">
<img src="https://github.com/user-attachments/assets/d93d183a-b1ff-4bfa-b47b-f94a296ffc91"
         alt="Screenshot">
</p>

## Architecture

The project consists of two repos: ML and UI. 

ML repo includes code for data preprocessing and model training on top of Apache Spark, MLFlow and TensorFlow. Once trained, the model binaries are saved to S3 as a MLFlow wrapper over TensorFlow Keras model. 

UI repo includes C# / Python backend and React.js frontend code. The backend loads the model from the configured S3 prefix via MLFlow. The code for model loading and inferencing is written entirely with Python. Python and C# code are executed in the same process without any extra Python APIs in between. .NET passes the movie matrix for inferencing to Python via Python Buffer protocol which ensures minimal in-memory data copying. 

The movies list is stored in a Amazon DynamoDB table prefilled by the data preprocessing job (see the ML repo).

Even though Apache Spark code operates with data using Unity Catalog and Delta Lake, UI repo does not depend on them and the only requirement for it to function is to load the trained model and list of all movies.

### Model Performance

*DISCLAIMER: I have done quite a lot of experiments to get to the desired quality level. Still, the model is trained only on the basis of movie genres, so it is more of a "genre recommender" for now. However, the current architecture (v2 on top of Apache Spark) is very flexible and allows to add more features much easier than it was before. So, adding more features (like movie description embeddings) is on top of my TODO list (see it below).*

What I have found to work the best at the moment on the MovieLens 100k data set is:

- Generating synthetic users: detecting centroids with [K-Medoids algorithm](https://github.com/kno10/python-kmedoids) and transforming the existing 610 users to roughly 40k (each with its own "center of interests")
- Oversampling of the low-represented movie genres

All metrics (NDCG, MSE, MAE, etc.) are logged and tracked in MLFlow Experiments.

### Features

* Neural Network with two tower architecture
* Trained with the (oversampled) [MovieLens 100k](https://grouplens.org/datasets/movielens/) tiny dataset
* Predictions on the whole movie data set (~9700 movies currently)
* Training and raw data preprocessing with Apache Spark / Databricks, Unity Catalog, Delta Lake, MLFlow and TensorFlow
* Content features: only movie genres for now (see todo)
* User features: average user rating for each genre
* Predictions on the whole movie data set (~9700 movies currently)
* MSE loss function with negative sampling
* Oversampling by low-represented genres
   
<img width="531" height="455" alt="La Pelicula Architecture Diagram" src="https://github.com/user-attachments/assets/fa0ed504-6629-446a-a8ff-9b337e44027b" />



## How To Build and Run Locally

Prerequisites: 

* You have Python 3.13 installed (or use pyenv)
* You have a TMDB API key which you can get [here](https://developer.themoviedb.org/docs/getting-started) (the data is cached to a Unity Catalog table).
* You have [MLFlow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/) installed and run locally
* You have [Unity Catalog](https://docs.unitycatalog.io/quickstart/) installed and run locally

Apache Spark code by default will run in client in-process mode using threads as workers. You only need to install the dependencies above.

From your command line:

```bash
# Clone this repository
$ git clone https://github.com/alexgilevich/lapelicula-ml

# Go into the repository
$ cd lapelicula-ml

# Set environment variables (example below)
$ export TMDB_API_KEY="{put your TMDB API key here}"
$ export AWS_ACCESS_KEY_ID={your AWS credentials to load the model from S3 and access DynamoDB movie table}
$ export AWS_SECRET_ACCESS_KEY={your AWS credentials to load the model from S3 and access DynamoDB movie table}
$ export AWS_REGION={your AWS credentials to load the model from S3 and access DynamoDB movie table}
$ export CONTAINER_TYPE=local-development
$ export UC_DEFAULT_CATALOG_NAME=lapelicula
$ export UC_DEFAULT_SCHEMA_NAME=default
$ export UC_DEFAULT_TABLE_LOCATION=~/.unitycatalog/lapelicula/default
$ export RAW_DATA_PATH=~/Sources/lapelicula-ml/data
$ export MLFLOW_MODEL_EXPERIMENT_NAME=lapelicula-movie-recommender-training
$ export MLFLOW_TRAINING_ARTIFACTS_LOCATION=/tmp/mlflow/
$ export MLFLOW_TRACKING_SERVER_URI=http://127.0.0.1:5000
$ export MLFLOW_REGISTRY_SERVER_URI=uc:http://127.0.0.1:8080
$ export MODEL_SAVE_S3_BUCKET=lapelicula
$ export MODEL_SAVE_S3_PREFIX=models/
$ export MLFLOW_MODEL_NAME=lapelicula-movie-recommender-model-new
$ export NUM_EPOCHS=30


# Install packages
$ uv sync
# or
$ pip install -r requirements.txt


# Run preprocessing
$ cd src/preprocessing
$ python3 step01_extract_raw_data.py
$ python3 step02_preprocess_movies.py
$ python3 step03_generate_synthetic_users.py
$ python3 step04_build_user_features.py
$ python3 step05_oversample_ratings.py
$ python3 step06_enrich_movies.py
$ python3 step07_sync_to_dynamodb.py

# Run training + inference on the test users
$ cd src/training
$ python3 step01_generate_training_data.py
$ python3 step02_split_training_data.py
$ python3 step03_train_model.py
# to load the model and test inference
$ python3 step04_test_inference.py

```


## How To Build and Run with Databricks

You can benefit from GPU serverless clusters by deploying to the US-based workspaces

```bash
$ databricks configure
# for development mode deployment
$ RAW_DATA_PATH=s3://bucket/prefix-to-raw-movielens-data databricks bundle deploy --target dev 
# for production mode deployment
$ RAW_DATA_PATH=s3://bucket/prefix-to-raw-movielens-data databricks bundle deploy --var "run_as={service principal or user identifier}" --target prod 
```

## TODO

- [ ] Use extra movie features not found in the MovieLens data set (such as generated embeddings of movie descriptions)
- [ ] Add support for candidate selection phase via ANN or other vector search algorithms with .NET backend
- [ ] Upgrade to a bigger MovieLens dataset
- [x] Incorporate in-batch genre-aware negative sampling and switch from mean squared error (MSE) to binary classification loss (or something else, e.g. contrastive loss)
- [x] Incorporate ranking metrics (such as precision@K, recall@K, NDCG)
- anything else? reach out to me to tell me what you think 

Feel free to help me with the list above by contributing to this repo :)

## Attribution

- [KMedoids – k-means algorithm alternative](https://github.com/kno10/python-kmedoids)
- [MovieLens dataset](https://grouplens.org/datasets/movielens/) – obtained approval from GroupLens to use their datasets


## Related

[Try it out yourself](https://lapelicula.net/)

## Credits

[Alexandr Gilevich](https://github.com/alexgilevich) – author and main contributor

## Support

If you like this project and think it has helped in any way, consider buying me a coffee!

<a href="https://www.buymeacoffee.com/alexgilevich" target="_blank" b-uzeyq7dyx3=""><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" b-uzeyq7dyx3=""></a>

## License

* ML repo is licensed under [MIT](https://github.com/alexgilevich/lapelicula-ml/LICENSE)
* UI and the code in the UI repo is licensed under [GNU General Public License v3.0](https://github.com/alexgilevich/lapelicula-ui/blob/main/LICENSE)


---

[lapelicula.net](https://lapelicula.net/)


