---
date: "2022-02-01"
slug: "everything-gets-a-package"
title: "Everything Gets a Package: My Python Data Science Setup"
tags:
  - data science
---

I make Python packages for everything. Big projects obviously get a package, but so does every tiny analysis. Spinning up a quick jupyter notebook to check something out? Build a package first. Oh yeah, and every package gets its own virtual environment.

Let's back up a little bit so that I can tell you why I do this. After that, I'll show you how I do this. Notably, my workflow is set up to make it simple to stay consistent.

## Picking Up Where I Left Off

I spent a year developing code almost exclusively within [JupyterLab](https://jupyter.org/) running on a docker container in the cloud. I had flexibility in picking the type of machine that the container was running on, so this allowed me to develop and run code on datasets that were too big for my local machine. I was also free to spin up multiple instances and run code on each one.

A result of this development environment was that I optimized my workflow to be as reproducible as possible. This is because I shut down the instance each day, and many things would get "reset". In engineering terms, not all of the "state was persisted". In particular, I would have to reinstall python packages each day that were not present in the original docker image. Given this, as well as my desire to be able to launch parallel experiments in multiple containers, I was incentivized to make my workflow extremely portable.

I used GitHub as the place to store my code and notebooks. Data all lived in cloud storage. A python package for my code held all of the python dependencies. It turn out, being forced to make your code portable is the same thing as requiring reproducibility in building and running your code. It's nice when constraints incentivize best practices.

I no longer use this cloud notebook development environment, but I still use this workflow. Even if I am only creating a notebook for an analysis, and there are no actual python `.py` files, I will define a python package solely to store the python dependencies required for the notebook. This ensures that future Ethan can run the notebook. 

This might seem like overkill, but one thing I've learned about data science is that nothing is ever over. This is not the last time that you will run that notebook. There will be follow-up questions based on your "final" analysis. You will end up wanting to train your model again at some point in the future; or, more sadistic, you will leave the company and somebody else will have to train your model. Being able to pick up where you left off, no matter how long ago you stopped working on something, pays dividends.

The rest of this post is about the tools that I use to lower the friction of making packages for every thing.

## Version Control

The worst part of Python is its packaging system. One level up from the packaging system is the actual version of Python that you're using. This matters! Beyond Python 3 vs 2, certain packages are only available for certain minor versions of Python, and various features are version-dependent such as [f-strings](https://www.python.org/dev/peps/pep-0498/), [type hints](https://www.python.org/dev/peps/pep-0563/), and [positional-only parameters](https://www.python.org/dev/peps/pep-0570/).

I use [pyenv](https://github.com/pyenv/pyenv) to manage the exact version of Python that I use for each project. I pair `pyenv` with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) in order to use `pyenv` for managing both my python version and my virtual environments. When I start a new project, I first create a virtual environment with a specific version of Python:

```commandline
# Create a virtual environment called "my-new-project"
# using Python 3.8.8
pyenv virtualenv 3.8.8 my-new-project
# Activate the virtual environment
pyenv activate my-new-project
```

I typically name the virtual environment the name of the directory that I'm in, and so new projects get a new directory.

From here on out, the workflow is comparable to if I had created a regular virtual environment using `virtualenv`. The primary difference is that I have also specified the version of Python to use.

If you're on a Mac, I'd recommend using `brew` to install `pyenv`. On Linux, use the [pyenv-installer](https://github.com/pyenv/pyenv-installer).

## Poetry In Motion

Immediately after I've created my new virtual environment, I then create a Python package using [poetry](https://python-poetry.org/). I typically run `poetry init` and enter blank info for most of the package info. I do this so that I can install my dependencies using `poetry`. That means that instead of running `pip install numpy`, I just run `poetry add numpy`. 

As long as I install my dependencies with `poetry`, then they will all be "tracked" with `poetry`. This is much better than running `pip freeze > requirements.txt` whenever you remember to do it. If I ever want to reinstall the dependencies, then I just have to navigate to my project directory and run `poetry install`.

The nice thing about `poetry` is that it allows me to easily "graduate" my code from notebook to script to package. In the notebook phase, `poetry` allows me to recreate the virtual environment that ran the notebook. If I want to move my code out of the notebook and into a full-fledged package, that's a small lift if I've been using `poetry` the whole time.

Speaking of the package, I put my code inside of a `src/` directory because of [this blog post](https://hynek.me/articles/testing-packaging/). I name my package the same name as the directory which is the same name as the virtual environment. For whatever reason, I often uses dashes for the latter two and underscores for the package.

The default `poetry` behavior is that it will manage your virtual environments for you. I don't like this because it requires you to change commands that you would run. For example, if my package installs a command line script called `my-script`, I can't just run `my-script` from the command line. Instead, I have to run `poetry run my-script`. 

This couples commands that I run with the fact that I'm using `poetry` which can be a bit awkward when you want to dockerize your code. It also enforces a virtual environment management framework on everybody in a shared codebase. Your `Makefile` now needs to know about `poetry`. I turn off this default behavior by running `poetry config virtualenvs.create false` after I first install `poetry`.


## Notebooks 😱

Notebooks are good and notebooks are bad. "Famous" data science people have argued both sides. I use notebooks for some things but not all the things. What I care about for this blog post is that I can run the notebook tomorrow in the same environment that I ran it today. All this (roughly) requires is ensuring that I can recreate the notebook's virtual environment.

As I've mentioned above, I create a Python package with `poetry` even when I'm only coding in a notebook. Sometimes I will use the same package across multiple, related notebooks. With `poetry`, I can recreate the virtual environment. The wrinkle of notebooks is making sure you're using the correct virtual environment.

I have a `base` virtual environment that is automatically activated when I start a shell. _This is where JupyterLab is installed_. For each package, I install [ipykernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) as a _development_ dependency:

```commandline
poetry add --dev ipykernel
```

After installing, I run the following command:

```commandline
python -m ipykernel install --user --name $NAME_OF_PACKAGE_VIRTUAL_ENVIRONMENT
```

I don't know exactly what this command does, but it ensures that I can choose my package "kernel" from within JupyterLab. 

One advantage of only installing JupyterLab once in a `base` environment is that I only have to customize it once. This means I can reuse the [Jupyterlab Code Formatter](https://jupyterlab-code-formatter.readthedocs.io/en/latest/index.html) extension for formatting code with [black](https://github.com/psf/black) across all my projects.

## Pip My Ride

From the tiny subset of readers who have made it this far, there is an even tinier subset who is probably asking "Why not just use `conda`?". I prefer to use `virtualenv` and `poetry`/`pip` (or really [PyPI](https://pypi.org/)) for Python dependencies for two reasons:

1. Most of my production code ends up getting deployed on Docker containers. I want fully reproducible builds, and I have [never found a way](https://twitter.com/eprosenthal/status/1181983025174896640) to export an existing `conda` environment in a way that it can be reproducibly built when packages come from different channels.
2. I have now worked at two companies that managed internal PyPI repositories such that you can publish private packages internally and then later install them using `pip`. I have not worked anywhere that managed this for private conda packages.

Oh, and there's probably some other vocal minority asking why I didn't write about Docker this whole time if I wanted fully reproducible environments. I use Docker for deployment. It feels like overkill for local development. I probably could probably optimize things so that it's lightweight for local development, but I'm okay right now with my current setup. Although, I have been meaning to add [pipx](https://github.com/pypa/pipx) to my setup for a while, but I'll save that for another day.
