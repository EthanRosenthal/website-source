# Bayesian Rock Climbing

The blog post is in `post.ipynb`.

See `pyproject.toml` for information on how to install `httpstan` in addition to the other packages required for running `post.ipynb`. I had to build `httpstan` from source and then just ran `poetry install`. I think this is because I have an old version of Ubuntu that I'm loathe to upgrade (version 16.04). Presumably, you can comment out the `httpstan` info and just run `poetry install` if you have a newer OS.

Also, you need to download the [this](https://www.kaggle.com/datasets/dcohen21/8anu-climbing-logbook) Kaggle dataset to `./database.sqlite`.
