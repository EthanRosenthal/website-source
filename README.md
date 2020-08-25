This repository is the source code for my personal website located at [ethanrosenthal.com](http://ethanrosenthal.com). The static website is generated with [Hugo](https://gohugo.io/) and deployed with [netlifly](https://www.netlify.com/).


## Rendering Notebooks

Pip install my fork of [nb_hugo_exporter](https://github.com/jbandlow/nb_hugo_exporter):

```commandline
pip install git+git@github.com:EthanRosenthal/nb_hugo_exporter.git@fix-images
```

Edit the notebook metadata (the wrench symbol on the left in jupyter lab, under Advanced Tools) to include the following tags:

```json
{
  "date": "2018-12-06",
  "title": "spacecutter: Ordinal Regression Models in PyTorch",
  "slug": "spacecutter-ordinal-regression",
  "hasMath": true,
  "notebook": true
}
```

`hasMath: true` is required for `katex` to work. `notebook: true` is required for the notebook to be rendered correctly.

Convert the jupyter notebook to hugo-compatible markdown, e.g.

```bash
python -m nbconvert notebooks/2018-12-07-spacecutter-ordinal-regression.ipynb --to hugo --output-dir content/blog/spacecutter-ordinal-regression
```

Deal with the inevitable `katex`/`latex` issues that arise :(

## Deployment

Commit to master.
