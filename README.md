<h1 align="center">
  <br>
  <a href="https://lapelicula.net/"><img width="200" height="200" alt="watching-a-movie" src="https://github.com/user-attachments/assets/827d93cb-6ea6-4723-9801-a71aaba12432" /></a>
  <br>
  Project La Pelicula
  <br>
</h1>


<h4 align="center">Movie recommender system on top of Tensorflow, Python and .NET</h4>

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

*DISCLAIMER: I have done quite a lot of experiments to get to the desired quality level. Still, a lot of ideas were left out and not implemented (yet). See the todo list for more details.*

What I have found to work the best at the moment on the MovieLens 100k data set is:

- Generating synthetic users: detecting centroids with K-medoids algorithm and transforming the existing 610 users to roughly 40k (each with its own "center of interests")
- Oversampling of the low-represented movie genres

I also tried many other ideas such as:
- Oversampling by rating bins (high and low bins had a higher percentage in my case)
- Generating extra features:
  1) adding hardcoded compound genres (such as Comedy-Romance, Kids-Fantasy, Animation-Comedy, etc. – see the full list in features.py)
  2) adding release year, average rating and other "simple" movie features as features
  3) adding extra user features such as how "tailored" their preferences are
  4) etc.
- Weighted average for ratings (where I weighted "bad" movies with average rating <= 2 with 0.5 coefficient)
- etc.

However, even if the model worked well on the test cut and existing test users, for the use case of La Pelicula (where the users select 3-5 genres on average) and new data in general, the model performed equally or worse. 
Still, one can still reproduce these experiments by enabling the appropriate feature flags in the code (see `data_pipeline.py`).


### Features

* Neural Network with two tower architecture
* Trained with the (oversampled) [MovieLens 100k](https://grouplens.org/datasets/movielens/) tiny dataset (check out more info in the [ML repo](https://github.com/alexgilevich/lapelicula-ml))
* Python code emdedded and run directly in the .NET process with [CSnakes](https://tonybaloney.github.io/CSnakes/)
* Local training and raw data preprocessing without external dependencies (with Pandas and TensorFlow)
* Content features: only movie genres for now (see todo)
* User features: average user rating for each genre
* Predictions on the whole movie data set (~9700 movies currently)
* MSE loss function
* Oversampling by low-represented genres
   
<img width="662" height="641" alt="recommender-Page-1 drawio (1)" src="https://github.com/user-attachments/assets/7eb96326-d3b7-46af-8015-ad336564fd7d" />



## How To Build Locally

You need Python 3 to run the pipeline.
You will also need a TMDB API key which you can get [here](https://developer.themoviedb.org/docs/getting-started) (only used for the first time, after that the data is written to the disk).
From your command line:

```bash
# Clone this repository
$ git clone https://github.com/alexgilevich/lapelicula-ml

# Go into the repository
$ cd lapelicula-ml

# Set API key
$ export TMDB_API_KEY="{put your TMDB API key here}"

# Install packages
$ pip install -r requirements.txt

# Run preprocessing + training + inference on the test users
$ python3 recommendation_system.py

```

Once the initial training is done, you'll be able to run the code with the trained model saved locally.


## TODO

- [ ] Reduce memory usage for the data preprocessing stage with Pandas
- [ ] Add support for candidate selection phase via ANN or other vector search algorithms with .NET backend
- [ ] Incorporate in-batch genre-aware negative sampling and switch from mean squared error (MSE) to binary classification loss (or something else, e.g. contrastive loss)
- [ ] Incorporate ranking metrics (such as precision@K, recall@K, NDCG)
- [ ] Finally, using extra movie features not found in the MovieLens data set (such as generated embeddings of movie descriptions)
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

MIT

---

[lapelicula.net](https://lapelicula.net/)


