---
date: "2024-10-23"
slug: "portable-quarto-reports"
title: "Portable Quarto Reports"
tags:
  - data science
---

# TPS Report

Like many Python data people, If I need to put together a proper analysis or report, I typically reach for Jupyter notebooks. I don't like to reach for it. I want my analysis to be quick enough that I can run a couple lines of code in an IPython console and call it a day. But that's never the case. And as we know, analysis begets analysis, and we're going to need to rerun our old numbers.

As I've [written before]({{< ref "/blog/2022-02-01-everything-is-a-package.md" >}}), I'm so particular that I even create an entire Python package for a single analysis. Still, I like to write the report in a notebook so that my prose, code, and plots are all in one, reproducible place. 

_Reproducible_ is actually a poor word to use. Nobody wants to _reproduce_ an analysis or job. They want to _modify_ that job and rerun it under different conditions. Maybe we should leave _reproducibility_ to the crises in social and cognitive sciences, and focus on _runnability_ for code.

Anyway, I digress. 

There are many problems with producing a Jupyter notebook as a report. Nobody else wants to read it. Technical people don't want to spin up a Jupyter server in order to run it, and non-technical people can't even render it. Sure, you could make them get a GitHub account and view it on there, but that's kind of annoying for everybody. Even if you can render the notebook, my code is ugly and distracting. 

# Browser Power

My solution isn't great, but I like to render the report as an HTML file. Everybody has a browser, so everybody can open it. If you render with the option to hide the code, then the report becomes readable.

I'm not the first to come up with this. Chris Said has a great [blog post](https://chris-said.io/2016/02/13/how-to-make-polished-jupyter-presentations-with-optional-code-visibility/) from 9 years ago about this which I copied into some [old notes](https://github.com/EthanRosenthal/distmem/tree/master/jupyter#hide-code) around that time. However, sometime in the last 9 years, the solutions in that blog post stopped working. During that same period of time, I got less data science-y and stopped performing so many analyses, so I kinda just stopped caring.

# Quarto to the Rescue

I recently got to be a data scientist again for a couple days and write a Jupyter report. I re-discovered that hiding code when rendering notebooks to HTML was broken. Thankfully, there have been a number of advances in the last 9 years. We now have [Quarto](https://quarto.org/). It does many more things than I can describe here. One thing I will describe is how it renders Jupyter notebooks to other formats, such as HTML. It also has built-in support for hiding code. To top it all off, the styling looks much nicer than a typically-rendered Jupyter notebook.

In order to render a Jupyter notebook to HTML, you need to install Quarto and then run the following from the command line:

```commandline
quarto render path/to/your_notebook.ipynb --to html
```

The HTML file will then be written to `path/to/your_notebook.html`. However, the default report will not hide the code. The report will also be spread across an HTML file and a separate folder of styling files. You're not going to zip all this up to send somebody. Instead, you can add the following to the top cell of your notebook (note: the cell must be a _Raw_ cell, rather than a Python or Markdown cell):

```
---
title: Your Report Title
format:
  html: 
    code-fold: true
    code-tools: true
    self-contained: true
---
```

As an example, here's a screenshot of a notebook that I made:

{{< figure src="notebook_screenshot.png" >}}

And here is the Quarto-rendered HTML:

{{< figure src="quarto_screenshot.png" >}}

Nice, right?

# Nothing's Perfect

I have two qualms with this method of delivering reports:

1. It's nicer to point somebody to an actual website than to send a file. While you can put the file in Google Drive and point somebody there, Drive won't natively render the HTML. Thus, if you want to update the report, you have to re-send the file.
1. There's no way for people to leave comments, like they would in a Google Doc. 

So, if you have better ideas, then let me know!
