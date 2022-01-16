This repository is the source code for my personal website located at [ethanrosenthal.com](http://ethanrosenthal.com). The static website is generated with [Hugo](https://gohugo.io/) and deployed with [netlifly](https://www.netlify.com/).


## Converting Notebooks to Hugo Markdown

Install this repo's Python package with poetry:

```commandline
poetry install
```

This will install my fork of [nb_hugo_exporter](https://github.com/jbandlow/nb_hugo_exporter).

Add a `hugo` section to the notebook metadata with the following tags. You can access the metadata from jupyter lab by clicking the gears icon on the upper right, opening Advanced Tools, and then looking in the Notebook Metadata cell. Make sure to click the check mark to save your changes to the metadata.

```python
{
    "hugo": {
        "date": "2021-11-03",
        "draft": false,
        # Whether or not there are emojis in the notebook
        "enableEmoji": true,
        # Whether or not this post was written in a jupyter notebook
        "notebook": true,
        # Set to true in order to enable latex support. (I use katex under the hood)
        "hasMath": false,
        # The URL path for the blog post: /2021/11/03/alignimation
        "slug": "alignimation",
        # Optional tags to add to the blog post
        "tags": [
            "computer-vision",
            "deep-learning",
            "machine-learning"
        ],
        # The blog post title
        "title": "Alignimation: Differentiable, Semantic Image Registration with Kornia"
    }
  ...
}
```
Convert the jupyter notebook to hugo-compatible markdown, e.g.

```bash
python -m nbconvert notebooks/2018-12-07-spacecutter-ordinal-regression.ipynb --to hugo --output-dir content/blog/spacecutter-ordinal-regression
```

Deal with the inevitable `katex`/`latex` issues that arise :(

## Deployment

Commit to master.
