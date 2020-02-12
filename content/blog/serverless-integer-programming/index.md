---
date: "2018-08-06"
title: "Quick and Dirty Serverless Integer Programming"
hasMath: true
notebook: true
slug: "serverless-integer-programming"
tags:
  - operations research
  - serverless
  - web development
---
{{% jupyter_cell_start markdown %}}



<!-- PELICAN_BEGIN_SUMMARY -->

We all know that Python has [risen](https://www.economist.com/graphic-detail/2018/07/26/python-is-becoming-the-worlds-most-popular-coding-language) above its humble beginnings such that it now powers [billion dollar companies](https://instagram-engineering.com/web-service-efficiency-at-instagram-with-python-4976d078e366). Let's not forget Python's roots, though! It's still an excellent language for running quick and dirty scripts that automate some task. While this works fine for automating my own tasks because I know how to navigate the command line, it's a bit much to ask a layperson to somehow install python and dependencies, open Terminal on a Mac (god help you if they have a Windows computer), type a random string of characters, and hit enter. Ideally, you would give the layperson a button, they hit it, and they get their result.

<!-- PELICAN_END_SUMMARY -->

I recently deployed a solution which allowed me to do just this. Even better - it was all free! In this post, I'll talk about how I used Google Sheets as my input form, [datasheets](https://datasheets.readthedocs.io/en/latest/#) to convert Google Sheets to [pandas](https://pandas.pydata.org/), [Zappa](https://github.com/Miserlou/Zappa) for deploying a serverless [Flask](http://flask.pocoo.org/) app, and [PuLP](https://pythonhosted.org/PuLP/) for solving a quick integer programming problem to make a simple and free ad hoc optimization service.

_Note: all the code for this service is located on my [github](https://github.com/EthanRosenthal/fml)_

## FML

Every project should start as a problem, and mine was no different. My wife competes in [fantasy movie league](https://fantasymovieleague.com/). This is like fantasy football for movie geeks. The rules are simple:

You are a fantasy movie theater owner. You must decide which movies to play on your 8 screens. Each movie costs a different amount to screen, and the goal is to generate the most box office revenue over the weekend given your available budget. Talking with her, I realized that, if one can do a good job predicting box office revenue for the weekend (the hard part of the problem), then deciding how many screens to play each movie becomes a simple integer programming allocation problem.

## Requirements

Now that we have the problem, what are the requirements? 

1. A method for inputting a bunch of data:
    - Movie name
    - Expected revenue
    - Cost to screen
1. Ability to run the allocation problem from a browser.
1. A view of the solution

What's the easiest input form that data scientists hate? 

Excel

What's worse than Excel?

Google Sheets

## Datasheets

Thankfully, [Squarespace](https://www.squarespace.com/) created [datasheets](https://datasheets.readthedocs.io/en/latest/). This is a nice library that makes interactions between `pandas` and Google Sheets impressively painless. The library is worth it for the detailed [OAuth page](https://datasheets.readthedocs.io/en/latest/getting_oauth_credentials.html) alone (I once spent 2 weeks struggling with Google OAuth pain and _really_ wish this page had existed at that time). What's particularly nice about the OAuth page is that it walks through setting up a _service account_ which does not require the end-user to go through the typical OAuth dance of browser redirects to and from the Google login page. This is nce because these redirects can get messed up when moving from local development to production systems in the cloud (or at least they always get messed up when I try to do it!).

Anywho, the first step was to setup my Google Sheets credentials and download the `client_secrets.json` and `service_key.json` files. With these handy, I can now access my Google Sheets spreadsheet using `datasheets`. The spreadsheet is named `FML`, and the `inputs` tab looks like

[![inputs](images/fml/inputs.png)](images/fml/inputs.png)

We can pull this into a pandas DataFrame by setting some `datasheets` environment variables to point to our credentials and then creating a `Client`

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import os
import datasheets

os.environ['DATASHEETS_SECRETS_PATH'] = 'client_secrets.json'
os.environ['DATASHEETS_SERVICE_PATH'] = 'service_key.json'

client = datasheets.Client(service=True)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

If that goes well, we can now grab our workbook (aka the Google Sheets file) and download the tab of data

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
workbook = client.fetch_workbook('FML')
input_tab = workbook.fetch_tab('inputs')
input_data = input_tab.fetch_data()

input_data
```

{{% jupyter_input_end %}}




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie</th>
      <th>revenue</th>
      <th>cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hotel Transylvania</td>
      <td>13600000.0</td>
      <td>157.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ant Man</td>
      <td>9100000.0</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Skyscraper</td>
      <td>5300000.0</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Incredibles 2</td>
      <td>7900000.0</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jurassic World</td>
      <td>6700000.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Purge</td>
      <td>2400000.0</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sorry to Bother</td>
      <td>1800000.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MI: Fallout</td>
      <td>63600000.0</td>
      <td>756.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mamma Mia</td>
      <td>19800000.0</td>
      <td>227.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Equalizer</td>
      <td>18300000.0</td>
      <td>201.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Unfriended</td>
      <td>1600000.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Blindspotting</td>
      <td>3000000.0</td>
      <td>41.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Teen Titans</td>
      <td>13400000.0</td>
      <td>149.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Three Idential Strangers</td>
      <td>1100000.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Eighth Grade</td>
      <td>946000.0</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
</div>



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Allocating Movies

I've [written]({{< ref "/blog/lets-talk-or" >}}) [previously]({{< ref "/blog/towards-optimal-personalization" >}}) about integer programming in Python using the [PuLP](https://pythonhosted.org/PuLP/) package, so I will avoid the introductions to integer programming and `pulp`. For this post, I will just quickly summarize the optimization problem, as it's reasonably simple!

We only have a single _decision variable_ in our problem. In the code, I call this `movie_counts`. In math, we can call it $S\_{m}$ which corresponds to how many screens we will play movie $m$ on for the weekend. This is an _integer_ decision variable with a lower bound of 0 and an upper bound of 8 (the number of screens we have available in our fantasy movie theater). It is an integer variable because we cannot screen a movie on 2.5 screens.

With our decision variable in hand, we must now define an objective function. We simply want to maximize expected revenue. Let's define a quantity $r\_{m}$ which is the amount of money that we expect movie $m$ to bring in (this is the revenue column in the above DataFrame). Our objective function is then simply

$$\sum\_{m} r\_{m} * S\_{m}$$

Lastly, we need some constraints. We only have two, but, before I introduce them, I need to introduce one slight wrinkle in fantasy movie league. You get charged <span>$</span>2 million for every screen that you leave empty. We can incorporate this into our optimization problem by assuming that there is an \_extra\_ movie called "Empty Screen" and that the expected revenue for that movie is \_negative\_ <span>$</span>2 million. Our two constraints can now be defined:

1. Every screen must be assigned a movie
  $$ \sum\_{m} S\_{m} = 8 $$
2. We have a limited budget of <span>$</span>1000. Let's say movie $m$ costs $c_{m}$ to screen. Our budget constraint is thus
  $$ \sum\_{m} c\_{m} * S\_{m} \leq 1000 $$
  
And that's it: one type of decision variable, a simple objective function, and two constraints. If you're interested, I wrap all of the above steps into an `Optimizer` class in the [fml code]((https://github.com/EthanRosenthal/fml/blob/master/fml/optimizer.py).

With the optimization problem complete, I can pack up the solution as a DataFrame and use `datasheets` to write the data back to the `outputs` tab of the spreadsheet

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
solution = ...
outputs_tab = workbook.fetch_tab('outputs')
outputs_tab.insert_data(solution)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Painless Serverless

The final step was to create a tiny Flask app with a button to launch the optimization. I made the simplest barebones site that I could, and then it was time to deploy.

[![website](images/fml/website.png)](images/fml/website.png)

[Zappa](https://github.com/Miserlou/Zappa) is a ridiculously cool Python library that lets you run any Python application as an AWS Lambda function and make it all discoverable via API Gateway. What this means is that you can make a Python website and run it in the cloud without an actual server running the code (as long as your website runs quickly, and uses few resources). You only pay for each time the website runs, but the first million times per month are free. If my wife happens to run this more than 1 million times, then I'll happily pay money.

I was blown away by how easy `Zappa` was to use. Honestly, the hardest part was figuring out how to install python 3.6 on my linux computer because you have to use `virtualenv` instead of `conda` (though there's a [PR](https://github.com/Miserlou/Zappa/pull/108) to fix that).

I'm just going to copy the documentation on how to get `Zappa` working because this is literally all that I had to do:

```bash
pip install zappa
zappa init
zappa deploy
```

After all of your code gets zipped up and sent to the cloud, `Zappa` tells you what cryptic URL at which you can now find your app. You can use custom domains and a gazillion other options, but this is quick and dirty serverless integer programming, remember?

With the website deployed, my wife can now input data into the spreadsheet, hit the `Calculate` button on the website, and then watch the spreadsheet for the optimal movie screens with nary a command line in sight.


{{% jupyter_cell_end %}}