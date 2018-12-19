This repository is the source code for my personal website located at [ethanrosenthal.com](http://ethanrosenthal.com). The static website is generated with [Hugo](https://gohugo.io/) and deployed with [netlifly](https://www.netlify.com/).


## Notebooks

Use [this branch of my fork](https://github.com/EthanRosenthal/nb_hugo_exporter/tree/fix-images) of the [nb_hugo_exporter](https://github.com/jbandlow/nb_hugo_exporter) repo.

```bash
git clone git@github.com:EthanRosenthal/nb_hugo_exporter.git
git checkout fix-images
```

Install the package locally with pip:

```bash
pip install -e .
```

Convert a jupyter notebook to hugo-compatible markdown, e.g.

```bash
python -m nbconvert notebooks/2018-12-07-spacecutter-ordinal-regression.ipynb --to hugo --output-dir content/blog/spacecutter-ordinal-regression
```

Open up `content/blog/spacecutter-ordinal-regression/index.md` and edit the metadata at the top. Something like the following works:

---
date: "2018-12-06"
title: "spacecutter: Ordinal Regression Models in PyTorch"
slug: "spacecutter-ordinal-regression"
hasMath: true
notebook: true
---

`hasMath: true` is required for `katex` to work. `notebook: true` is required for the notebook to be rendered correctly.

Deal with the inevitable `katex`/`latex` issues that arise :(

## Deployment

Push to master.
