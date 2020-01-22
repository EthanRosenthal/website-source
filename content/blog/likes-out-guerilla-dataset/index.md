---
date: "2016-10-09"
slug: "likes-out-guerilla-dataset"
notebook: true
title: "Likes Out! Guerilla Dataset!"
---
{{% jupyter_cell_start markdown %}}

<!-- PELICAN_BEGIN_SUMMARY -->

-- [Zack de la Rocha](https://www.youtube.com/watch?v=H0kJLW2EwMg)

*tl;dr -> I collected an implicit feedback dataset along with side-information about the items. This dataset contains around 62,000 users and 28,000 items. All the data lives [here](https://github.com/EthanRosenthal/rec-a-sketch/tree/master/data) inside of [this](https://github.com/EthanRosenthal/rec-a-sketch) repo. Enjoy!*

<!-- PELICAN_END_SUMMARY -->

In a previous [post]({{< ref "/blog/explicit-matrix-factorization-sgd-als" >}}), I wrote about how to use matrix factorization and explicit feedback data in order to build recommendation systems. This is data where a user has given a clear preference for an item such as a star rating for an Amazon product or a numerical rating for a movie like in the [MovieLens](http://grouplens.org/datasets/movielens/) data. A natural next step is to discuss recommendation systems for implicit feedback which is data where a user has shown a *preference* for an item like "number of minutes listened" for a song on Spotify or "number of times clicked" for a product on a website.

Implicit feedback-based techniques likely consitute the majority of modern recommender systems. When I set out to write a post on these techniques, I found it difficult to find suitable data. This makes sense - most companies are loathe to share users' click or usage data (and for good reasons). A cursory google search revealed a couple datasets that people use, but I kept finding issues with these datasets. For example, the million song database was shown to have [some](http://labrosa.ee.columbia.edu/millionsong/blog/12-1-2-matching-errors-taste-profile-and-msd) [issues](http://labrosa.ee.columbia.edu/millionsong/blog/12-2-12-fixing-matching-errors) with data quality, while many [other](http://link.springer.com/chapter/10.1007%2F978-3-642-33486-3_5) [people](http://dl.acm.org/citation.cfm?id=2799671) just repurposed the MovieLens or Netflix data as though it was implicit (which it is not).

This started to feel like one of those "fuck it, I'll do it myself" things. And so I did.

All code for collecting this data is located on my [github](https://github.com/EthanRosenthal/rec-a-sketch). The actual collected data lives in this repo, as well.

## [Sketchfab](https://sketchfab.com/)

Back when I was a graduate student, I thought for some time that maybe I would work in the hardware space (or at a museum, or the government, or a gazillion other things). I wanted to have public, digital proof of my ([shitty](https://sketchfab.com/models/3a10dc58988748c69ed4b501eafaea00)) CAD skills, and I stumbled upon [Sketchfab](https://sketchfab.com/), a website which allows you to share 3D renderings that anybody else with a browser can rotate, zoom, or watch animate. It's kind of like YouTube for 3D (and now VR!).


<div class="sketchfab-embed-wrapper"><iframe width="640" height="480" src="https://sketchfab.com/models/522e811044bc4e09bf84431e6c1cc109/embed" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" onmousewheel=""></iframe>

<p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;">
    <a href="https://sketchfab.com/models/522e811044bc4e09bf84431e6c1cc109?utm_medium=embed&utm_source=website&utm_campain=share-popup" target="_blank" style="font-weight: bold; color: #1CAAD9;">Liopleurodon Ferox Swim Cycle</a>
    by <a href="https://sketchfab.com/kyan0s?utm_medium=embed&utm_source=website&utm_campain=share-popup" target="_blank" style="font-weight: bold; color: #1CAAD9;">Kyan0s</a>
    on <a href="https://sketchfab.com?utm_medium=embed&utm_source=website&utm_campain=share-popup" target="_blank" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a>
</p>
</div>

Users can "like" 3D models which is an excellent implicit signal. It turns out you can actually see which user liked which model. This presumably allows one to reconstruct the classic recommendation system "ratings matrix" of *users* as rows and *3D models* as columns with *likes* as the elements in the sparse matrix. 

Okay, I can see the likes on the website, but how do I actually get the data?

## [Crawling](https://www.youtube.com/watch?v=Gd9OhYroLN0) with Selenium

When I was at [Insight Data Science](http://insightdatascience.com/), I built an ugly [script](https://github.com/EthanRosenthal/TutorWorthy/blob/master/pre_production/scripts/scraper.py) to scrape a tutoring website. This was relatively easy. The site was largely static, so I used BeautifulSoup to simply parse through the HTML.

Sketchfab is a more modern site with extensive javascript. One must wait for the javascript to render the HTML before parsing through it. A method of automating this is to use [Selenium](http://www.seleniumhq.org/). This software essentially lets you write code to drive an actual web browser.

To get up and running with Selenium, you must first download a driver to run your browser. I went [here](https://sites.google.com/a/chromium.org/chromedriver/downloads) to get a Chrome driver. The Python Selenium package can then be installed using anaconda on the conda-forge channel:

```bash
conda install --channel https://conda.anaconda.org/conda-forge selenium
```

Opening a browser window with Selenium is quite simple:

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
from selenium import webdriver

chromedriver = '/path/to/chromedriver'
BROWSER = webdriver.Chrome(chromedriver)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Now we must decide where to point the browser.

Sketchfab has over [1 Million](https://blog.sketchfab.com/1million/) 3D models and more than 600,000 users. However, not every user has liked a model, and not every model has been liked by a user. I decided to limit my search to models that had been liked by at least 5 users. To start my crawling, I went to the "all" [page](https://sketchfab.com/models?sort_by=-likeCount) for popular models (sorted by number of likes, descending) and started crawling from the top.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
BROWSER.get('https://sketchfab.com/models?sort_by=-likeCount&page=1')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Upon opening the main models page, you can open the chrome developer tools (ctrl-shift-i in linux) to reveal the HTML structure of the page. This looks like the following (click to view full-size):

[![main page](/images/sketchfab_pop_models.png)](/images/sketchfab_pop_models.png)

Looking through the HTML reveals that all of the displayed 3D models are housed in a ```<div>``` of class ```infinite-grid```. Each 3D model is inside of a ```<li>``` element with class ```item```. One can grab the list of all these list elements as follows:

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
elem = BROWSER.find_element_by_xpath("//div[@class='infinite-grid']")
item_list = elem.find_elements_by_xpath(".//li[@class='item']")
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

It turns out that each Sketchfab model has a unique ID associated with it which we shall call its model ID, or ```mid```. This ```mid``` can be found in each list element through the ```data-uid``` attribute.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
item = item_list[0]
mid = item.get_attribute('data-uid')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

The url for the model is then simply ```https://sketchfab.com/models/mid``` where you replace ```mid``` with the actual unique ID.

I have written a script which automates this collection of each ```mid```. This script is called [crawl.py](https://github.com/EthanRosenthal/rec-a-sketch/blob/master/crawl.py) in the main [repo](https://github.com/EthanRosenthal/rec-a-sketch). To log all model urls, one runs

```bash
python crawl.py config.yml --type urls
```

All told, I ended up with 28,825 models (from October 2016). The model name and associated ```mid``` are in the file ```model_urls.psv``` [here](https://github.com/EthanRosenthal/rec-a-sketch/tree/master/data).

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## [Apple](https://www.youtube.com/watch?v=swWT2UcDv2c) of my API

In order to log which user liked which model, I originally wrote a Selenium script to go to every model's url and scroll through the users that had liked the model. This took for-fucking-ever. I realized that maybe Sketchfab serves up this information via an API. I did a quick Google search and stumbled upon [Greg Reda's](http://www.gregreda.com/) blog [post](http://www.gregreda.com/2015/02/15/web-scraping-finding-the-api/) which described how to use semi-secret APIs for collecting data. Sure enough, this worked perfectly for my task!

With a ```mid``` in hand, one can hit the api by passing the following parameters

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import requests

mid = '522e811044bc4e09bf84431e6c1cc109'
count = 24
params = {'model':mid, 'count':count, 'offset':0}

url = 'https://sketchfab.com/i/likes'
response = requests.get(url, params=params).json()
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Inside of ```response['results']``` is a list of information about each user that liked the model. ```crawl.py``` has a function to read in the model urls file output by ```crawl.py``` and then collect every user that liked that model.

```bash
python crawl.py config.yml --type likes
```

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

After running this script collecting likes on 28,825 models in early October 2016, I ended up with data on 62,583 users and 632,840 model-user-like combinations! This data is thankfully small enough to still fit in a github repo (52 Mb) and lives [here](https://github.com/EthanRosenthal/rec-a-sketch/tree/master/data)

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Even though these likes are public, I felt a little bad about making this data so easy to publicly parse. I wrote a small script called [anonymize.py](https://github.com/EthanRosenthal/rec-a-sketch/blob/master/anonymize.py) which hashes the user ID's for the model likes. Running this script is simple (just make sure to provide your own secret key):

```bash
python anonymize.py unanonymized_likes.csv anonymized_likes.csv "SECRET KEY"
```
The likes data in the main repo has been anonymized.

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

# [Information](https://www.youtube.com/watch?v=F0nOWwiosiw) on the side

An exciting area of recommendation research is the combination of user and item side information with implicit or explicit feedback. In later posts, I will address this space, but, for now, let's just try to grab some side information. Sketchfab users are able to categorize models that they upload (e.g. "Characters", "Places & scenes", etc...) as well as tag their models with relevant labels (e.g. "bird", "maya", "blender", "sculpture", etc...). Presumably, this extra information about models could be useful in making more accurate recommendations.

```crawl.py``` has another function for grabbing the associated categories and tags of a model. I could not find an API way to do this, and the Selenium crawl is extremely slow. Thankfully, I've already got the data for you :) The model "features" file is called model_feats.psv and is in the /data directory of the main repo.

```bash
python crawl.py config.yml --type features
```

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

# What's next?

With all of our data in hand, subsequent blog posts will dive into the wild west of implicit feedback recommendation systems. I'll show you how to train these models, use these models, and then build a simple Flask app, called Rec-a-Sketch, for serving 3D Sketchfab recommendations.

{{% jupyter_cell_end %}}