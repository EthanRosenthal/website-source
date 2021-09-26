---
date: 2021-09-28
slug: "sandmat"
tags: ['sandwiches']
title: "Sandwich Data Science 2: Electric Boogaloo"
draft: true
notebook: false
hasMath: false   
---
{{% jupyter_cell_start markdown %}}

I told myself I wouldn't do it again. The [last time]({{< ref "blog/optimal-peanut-butter-and-banana" >}}) nearly broke me. And yet, [just when I thought I was out, they pull me back in](https://www.youtube.com/watch?v=G29DXfcdhBg). 

Against my better judgement, I did another sandwich data science project. Thankfully, this one was significantly simpler.

## An Impenetrable Menu

I work at [Square](https://squareup.com/), and their NYC office is in SoHo. While there are many reasons not to go into the office nowadays, one draw is that I can pick up lunch at [Alidoro](https://www.alidoronyc.com/), a tiny Italian sandwich shop that's nearby. The sandwiches are the quintissential European antithesis to American sandwiches; they consist of only a couple, extremely high quality ingredients.

From these few ingredients emerge 40 different types of sandwiches, and these 40 sandwiches form an impenetrable menu.

<img style='width: 640px; padding: 10px'  src='images/sandmat/alidoro_menu.jpg' />

Naively, you may think you can pick a sandwich that looks close to what you want and then customize it. Perhaps you would like the Romeo but with some fresh mozzarella? Well then perhaps you would be wrong because _customization is not allowed_. Did I mention that there are some [Soup Nazi](https://youtu.be/euLQOQNVzgY) vibes to this place? You can only order what's on the menu, and it took the global pandemic to finally break their will to remain cash only.

Some people like to explore new items on a menu, while I always exploit the one that I've been happy with. Case in point: I get the Fellini on Foccacia every time. Still, I remember what it was like to be a newcomer and encounter that impenetrable menu. 

And so, this blog post is my attempt at data visualization. My goal is to visualize the menu in such a way that one can quickly scan it to find a sandwich they would like. As an added bonus, I'll close with some statistical modeling of the sandwich pricing.

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Packaging and Presentation

Like many of my blog posts, I wrote this one in a Jupyter notebook. While I would prefer to show the full code for the blog post, I didn't want this post to be as impenetrable as Alidoro's menu. I made a small `sandmat` package to house much of the code. The package, along with the Jupyter notebook version of this blog post can be found on GitHub [here](https://github.com/EthanRosenthal/website-source/tree/master/notebooks/2021-09-28-sandmat).

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

from sandmat import scrape, sorting, viz
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Menu Scraping

To start, we need to get the menu and turn it into "data". For whatever reason, I didn't feel like using pandas for this blog post, so everything we'll deal with will be collections of [dataclasses](https://docs.python.org/3/library/dataclasses.html). 

Within the `sandmat` package, I make two `dataclasses`: `Ingredient` and `Sandwich`. Most of the fields are self-explanatory with the exception for the ingredient categories. For these, I manually classify ingredients into `meat`, `cheese`, `topping`, or `dressing`. In hindsight, I probably should've made this an `Enum` field.

```python
@dataclass(frozen=True)
class Ingredient:
    name: str
    category: str


@dataclass(frozen=True)
class Sandwich:
    name: str
    ingredients: Tuple[Ingredient]
    price: float 
```

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

I use [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to scrape the [menu page](https://www.alidoronyc.com/menu/menu/) of the Alidoro website and grab the section of the HTML that pertains to the menu. I then do some parsing, cleaning, and categorization in order to turn the menu into a list of `Sandwich` objects.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
URL = "https://www.alidoronyc.com/menu/menu/"
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
sandwiches = scrape.get_sandwiches(URL)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
print(f"{len(sandwiches)} sandwiches found.")
print("Displaying the first two:\n")
for sandwich in sandwiches[:2]:
    print(sandwich)
    print()
```

{{% jupyter_input_end %}}

    40 sandwiches found.
    Displaying the first two:
    
    Sandwich(name='Matthew', ingredients=(Ingredient(name='prosciutto', category='meat'), Ingredient(name='fresh mozzarella', category='cheese'), Ingredient(name='dressing', category='dressing'), Ingredient(name='arugula', category='topping')), price=14.0)
    
    Sandwich(name='Alyssa', ingredients=(Ingredient(name='smoked chicken breast', category='meat'), Ingredient(name='fresh mozzarella', category='cheese'), Ingredient(name='arugula', category='topping'), Ingredient(name='dressing', category='dressing')), price=14.0)
    


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Ingredient Rank and File

I would like to display the sandwiches in "matrix" form. Each sandwich will be a row, each ingredient will be a column, and the values of the matrix will indicate if a sandwich has a particular ingredient. What's left is to decide on an order to the sandwich rows and an order to the ingredient columns.

In my initial approach, I coded up a traveling salesman problem in which sandwiches were cities and the overlap in ingredients between any two sandwiches was the "distance" between sandwiches. It would've made for the perfect title ("Traveling Sandwich Problem", obviously), but, contrary to the numerical solution, the result was visually suboptimal.

Thankfully, this is a problem where we can rely on domain expertise. As a sandwich eater myself, I thought about how I typically pick a sandwich. I often look at the meats first, then the cheeses, and then everything else. Ok, let's sort the ingredient columns by category "rank": `meat`, `cheese`, `topping`, `dressing`. Within each category, how about using the recsys go-to of sorting in descending order of popularity? Combining category rank and popularity gives us our full ingredient column order. In SQL, we'd want to do something like

```mysql
SELECT
  category
  , CASE 
    WHEN category = 'meat' THEN 1
    WHEN category = 'cheese' THEN 2
    WHEN cateogry = 'topping' THEN 3
    WHEN category = 'dressing' THEN 4
  END AS category_rank
  , ingredient
  , COUNT(DISTINCT sandwich) as num_sandwiches
FROM sandwich_ingredients
GROUP BY category, category_rank, ingredient
ORDER BY category_rank ASC, num_sandwiches DESC
```

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
ranked_categories = sorting.get_ranked_categories(sandwiches)
ordered_ingredients = sorting.get_ordered_ingredients(ranked_categories)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

For ordering our sandwich rows, let's sort them by a special `key` which is a tuple that contains their most popular ingredient in each category where the tuple is in order of `meat`, `cheese`, `topping`, `dressing`.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
ordered_sandwiches = sorting.get_ordered_sandwiches(sandwiches, ranked_categories)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Visualizing the Matrix

Finally, with our ordered ingredients and sandwiches, we can visualize the Alidoro sandwich menu as a matrix.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
sandwich_mat = viz.make_sandwich_matrix(ordered_sandwiches, ordered_ingredients)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

### _[Ed ecco qua](https://translate.google.com/?sl=auto&tl=it&text=et%20voila&op=translate&hl=en)_ 

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
fig, ax = viz.plot_sandwiches(sandwich_mat, ordered_sandwiches, ordered_ingredients)
plt.show();
```

{{% jupyter_input_end %}}


{{< figure src="./index_15_0.png" >}}


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

# Pricing Analysis

Just for prosciuttos and giggles, I decided to treat my sandwich matrix as a [design matrix](https://en.wikipedia.org/wiki/Design_matrix). I'll fit a linear regression on the sandwich matrix with the sandwich price as the target variable. The model coefficients will thus be the price of each ingredient, and a bias term will take care of the base price of the sandwich (which includes the bread). As you can see, the model is pretty well-calibrated! I guess Alidoro's sandwich pricing is pretty consistent.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import statsmodels.api as sm
import numpy as np
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
y = np.array([sandwich.price for sandwich in ordered_sandwiches])
X = sandwich_mat.copy()

X = sm.add_constant(X, prepend=True)

model = sm.OLS(y, X)
res = model.fit()
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
res.summary(
    yname="Price ($)", xname=["Base Sandwich Price"] + list(ordered_ingredients)
)
```

{{% jupyter_input_end %}}




<pre><code class="nohighlight"><table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Price ($)</td>    <th>  R-squared:         </th> <td>   0.971</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.940</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   31.39</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 26 Sep 2021</td> <th>  Prob (F-statistic):</th> <td>1.48e-10</td>
</tr>
<tr>
  <th>Time:</th>                 <td>10:22:02</td>     <th>  Log-Likelihood:    </th> <td>  9.6979</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   22.60</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    19</td>      <th>  BIC:               </th> <td>   58.07</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    20</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Base Sandwich Price</th>   <td>    8.0451</td> <td>    0.265</td> <td>   30.334</td> <td> 0.000</td> <td>    7.490</td> <td>    8.600</td>
</tr>
<tr>
  <th>prosciutto</th>            <td>    2.1138</td> <td>    0.166</td> <td>   12.769</td> <td> 0.000</td> <td>    1.767</td> <td>    2.460</td>
</tr>
<tr>
  <th>sopressata</th>            <td>    1.9554</td> <td>    0.152</td> <td>   12.875</td> <td> 0.000</td> <td>    1.638</td> <td>    2.273</td>
</tr>
<tr>
  <th>smoked chicken breast</th> <td>    2.0618</td> <td>    0.182</td> <td>   11.323</td> <td> 0.000</td> <td>    1.681</td> <td>    2.443</td>
</tr>
<tr>
  <th>tuna</th>                  <td>    1.7025</td> <td>    0.171</td> <td>    9.940</td> <td> 0.000</td> <td>    1.344</td> <td>    2.061</td>
</tr>
<tr>
  <th>salami</th>                <td>    2.1288</td> <td>    0.279</td> <td>    7.641</td> <td> 0.000</td> <td>    1.546</td> <td>    2.712</td>
</tr>
<tr>
  <th>capicollo</th>             <td>    2.0982</td> <td>    0.327</td> <td>    6.421</td> <td> 0.000</td> <td>    1.414</td> <td>    2.782</td>
</tr>
<tr>
  <th>mortadella</th>            <td>    3.0738</td> <td>    0.359</td> <td>    8.573</td> <td> 0.000</td> <td>    2.323</td> <td>    3.824</td>
</tr>
<tr>
  <th>sardines or mackerel</th>  <td>    2.4387</td> <td>    0.375</td> <td>    6.497</td> <td> 0.000</td> <td>    1.653</td> <td>    3.224</td>
</tr>
<tr>
  <th>fresh mozzarella</th>      <td>    1.3168</td> <td>    0.174</td> <td>    7.581</td> <td> 0.000</td> <td>    0.953</td> <td>    1.680</td>
</tr>
<tr>
  <th>smoked mozzarella</th>     <td>    1.3141</td> <td>    0.210</td> <td>    6.271</td> <td> 0.000</td> <td>    0.875</td> <td>    1.753</td>
</tr>
<tr>
  <th>m. bel paese</th>          <td>    1.2748</td> <td>    0.223</td> <td>    5.707</td> <td> 0.000</td> <td>    0.807</td> <td>    1.742</td>
</tr>
<tr>
  <th>provolone cheese</th>      <td>    1.3559</td> <td>    0.250</td> <td>    5.429</td> <td> 0.000</td> <td>    0.833</td> <td>    1.879</td>
</tr>
<tr>
  <th>arugula</th>               <td>    1.2985</td> <td>    0.129</td> <td>   10.076</td> <td> 0.000</td> <td>    1.029</td> <td>    1.568</td>
</tr>
<tr>
  <th>artichokes</th>            <td>    1.2708</td> <td>    0.140</td> <td>    9.074</td> <td> 0.000</td> <td>    0.978</td> <td>    1.564</td>
</tr>
<tr>
  <th>sun dried tomatoes</th>    <td>    1.2414</td> <td>    0.147</td> <td>    8.458</td> <td> 0.000</td> <td>    0.934</td> <td>    1.549</td>
</tr>
<tr>
  <th>sweet roasted peppers</th> <td>    1.1692</td> <td>    0.135</td> <td>    8.637</td> <td> 0.000</td> <td>    0.886</td> <td>    1.453</td>
</tr>
<tr>
  <th>hot peppers</th>           <td>    1.0734</td> <td>    0.183</td> <td>    5.850</td> <td> 0.000</td> <td>    0.689</td> <td>    1.458</td>
</tr>
<tr>
  <th>caponata of eggplant</th>  <td>    1.0643</td> <td>    0.210</td> <td>    5.074</td> <td> 0.000</td> <td>    0.625</td> <td>    1.503</td>
</tr>
<tr>
  <th>dressing</th>              <td>    1.0242</td> <td>    0.172</td> <td>    5.963</td> <td> 0.000</td> <td>    0.665</td> <td>    1.384</td>
</tr>
<tr>
  <th>olive paste</th>           <td>    0.5690</td> <td>    0.285</td> <td>    1.998</td> <td> 0.060</td> <td>   -0.027</td> <td>    1.165</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14.030</td> <th>  Durbin-Watson:     </th> <td>   2.450</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  17.010</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.089</td> <th>  Prob(JB):          </th> <td>0.000202</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.337</td> <th>  Cond. No.          </th> <td>    15.8</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.</pre></code>



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We can inspect this model visually by plotting the prices of all of the ingredients. I had no idea mortadella was the most expensive meat.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
viz.plot_ingredients(ordered_ingredients, res)
```

{{% jupyter_input_end %}}




<pre><code class="nohighlight">
<div id="altair-viz-f94987caf2cc4251a973f52f42e9f6e0"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-f94987caf2cc4251a973f52f42e9f6e0") {
      outputDiv = document.getElementById("altair-viz-f94987caf2cc4251a973f52f42e9f6e0");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-287075b304946e3c32a201a75c24ba95"}, "mark": {"type": "bar", "opacity": 0.5}, "encoding": {"color": {"type": "nominal", "field": "Category", "scale": {"domain": ["meat", "cheese", "topping", "dressing"], "range": ["red", "orange", "green", "purple"]}}, "tooltip": [{"type": "nominal", "field": "Ingredient"}, {"type": "quantitative", "field": "Price", "format": "$.2f"}, {"type": "quantitative", "field": "SE", "format": "$.2f", "title": "Standard Error"}], "x": {"type": "quantitative", "axis": {"format": "$.2f"}, "field": "Price"}, "y": {"type": "nominal", "field": "Ingredient", "sort": "-x"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-287075b304946e3c32a201a75c24ba95": [{"x": 0, "Ingredient": "prosciutto", "Price": 2.1137890738912115, "Category": "meat", "SE": 0.16553756283787763}, {"x": 1, "Ingredient": "sopressata", "Price": 1.9554333127912042, "Category": "meat", "SE": 0.15188038837468665}, {"x": 2, "Ingredient": "smoked chicken breast", "Price": 2.061791626777762, "Category": "meat", "SE": 0.1820867178780412}, {"x": 3, "Ingredient": "tuna", "Price": 1.702522105709174, "Category": "meat", "SE": 0.17127351938871166}, {"x": 4, "Ingredient": "salami", "Price": 2.1288226011070925, "Category": "meat", "SE": 0.2785895972166576}, {"x": 5, "Ingredient": "capicollo", "Price": 2.0981922186735495, "Category": "meat", "SE": 0.3267739353914283}, {"x": 6, "Ingredient": "mortadella", "Price": 3.073757442638847, "Category": "meat", "SE": 0.3585421657495058}, {"x": 7, "Ingredient": "sardines or mackerel", "Price": 2.438688161612248, "Category": "meat", "SE": 0.37538403803199477}, {"x": 8, "Ingredient": "fresh mozzarella", "Price": 1.3168277014499852, "Category": "cheese", "SE": 0.17370329102356766}, {"x": 9, "Ingredient": "smoked mozzarella", "Price": 1.3140874880818842, "Category": "cheese", "SE": 0.20954867900674967}, {"x": 10, "Ingredient": "m. bel paese", "Price": 1.2748439661518889, "Category": "cheese", "SE": 0.22336733082318977}, {"x": 11, "Ingredient": "provolone cheese", "Price": 1.3559397435849672, "Category": "cheese", "SE": 0.24973996622933248}, {"x": 12, "Ingredient": "arugula", "Price": 1.2985122076405897, "Category": "topping", "SE": 0.12887617620140684}, {"x": 13, "Ingredient": "artichokes", "Price": 1.2708178771531484, "Category": "topping", "SE": 0.14004657276507868}, {"x": 14, "Ingredient": "sun dried tomatoes", "Price": 1.2413716536238235, "Category": "topping", "SE": 0.14676768598822215}, {"x": 15, "Ingredient": "sweet roasted peppers", "Price": 1.169190231266147, "Category": "topping", "SE": 0.13537276585159277}, {"x": 16, "Ingredient": "hot peppers", "Price": 1.0734309870961365, "Category": "topping", "SE": 0.18349972353887048}, {"x": 17, "Ingredient": "caponata of eggplant", "Price": 1.0643186372991267, "Category": "topping", "SE": 0.20976977462655036}, {"x": 18, "Ingredient": "dressing", "Price": 1.0241749100207613, "Category": "dressing", "SE": 0.17174446515781613}, {"x": 19, "Ingredient": "olive paste", "Price": 0.5690495318335698, "Category": "dressing", "SE": 0.2847504995318473}]}}, {"mode": "vega-lite"});
</script></pre></code>



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

And last but not least, we can compare the sandwich price to the model's predicted price in order to get an idea if any sandwich's price is wildly inconsistent. Most sandiwch prices are consistent, although the Gabriella is apparently cheaper than expected at $11.00 for (only!) fresh mozzarella, dressing, and arugula. I don't know if I'd call that cheap, but, then again, neither is SoHo.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
y_pred = model.predict(res.params)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
chart = viz.plot_actual_vs_pred(y, y_pred, ordered_sandwiches)
chart.properties(width=400, height=400)
```

{{% jupyter_input_end %}}




<pre><code class="nohighlight">
<div id="altair-viz-1c9a9820f1a144838b12b8aa2e9ada38"></div>
<script type="text/javascript">
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-1c9a9820f1a144838b12b8aa2e9ada38") {
      outputDiv = document.getElementById("altair-viz-1c9a9820f1a144838b12b8aa2e9ada38");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.8.1?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function loadScript(lib) {
      return new Promise(function(resolve, reject) {
        var s = document.createElement('script');
        s.src = paths[lib];
        s.async = true;
        s.onload = () => resolve(paths[lib]);
        s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
        document.getElementsByTagName("head")[0].appendChild(s);
      });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else if (typeof vegaEmbed === "function") {
      displayChart(vegaEmbed);
    } else {
      loadScript("vega")
        .then(() => loadScript("vega-lite"))
        .then(() => loadScript("vega-embed"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"data": {"name": "data-f0562a4079f94c3185bd873d91891489"}, "mark": "point", "encoding": {"color": {"type": "nominal", "field": "Vegetarian"}, "tooltip": [{"type": "nominal", "field": "Sandwich"}, {"type": "nominal", "field": "Ingredients"}], "x": {"type": "quantitative", "axis": {"format": "$.2f"}, "field": "Price", "scale": {"domain": [10.0, 15.5]}}, "y": {"type": "quantitative", "axis": {"format": "$.2f"}, "field": "Prediction", "scale": {"domain": [10.0, 15.5]}}}}, {"data": {"name": "data-edebebbba8106f8ffacea64eec8a8b00"}, "mark": {"type": "line", "color": "black", "strokeDash": [5, 5]}, "encoding": {"x": {"type": "quantitative", "axis": {"title": "Price"}, "field": "x"}, "y": {"type": "quantitative", "axis": {"title": "Prediction"}, "field": "y"}}}], "height": 400, "width": 400, "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json", "datasets": {"data-f0562a4079f94c3185bd873d91891489": [{"Price": 14.0, "Prediction": 13.798400111614582, "Sandwich": "Matthew", "Ingredients": "prosciutto, fresh mozzarella, dressing, arugula", "Vegetarian": false}, {"Price": 15.0, "Prediction": 15.169386069844151, "Sandwich": "Pinocchio", "Ingredients": "prosciutto, sopressata, fresh mozzarella, sweet roasted peppers, olive paste", "Vegetarian": false}, {"Price": 12.5, "Prediction": 12.499887903973992, "Sandwich": "Donatella", "Ingredients": "prosciutto, fresh mozzarella, dressing", "Vegetarian": false}, {"Price": 11.5, "Prediction": 11.47571299395323, "Sandwich": "Galileo", "Ingredients": "prosciutto, fresh mozzarella", "Vegetarian": false}, {"Price": 14.0, "Prediction": 14.012856641849543, "Sandwich": "Dino", "Ingredients": "prosciutto, smoked mozzarella, sun dried tomatoes, arugula", "Vegetarian": false}, {"Price": 12.5, "Prediction": 12.642163011851277, "Sandwich": "Lucia", "Ingredients": "prosciutto, smoked mozzarella, sweet roasted peppers", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.88676823082494, "Sandwich": "Michelangelo", "Ingredients": "prosciutto, provolone cheese, hot peppers, arugula", "Vegetarian": false}, {"Price": 11.5, "Prediction": 11.514825036088213, "Sandwich": "Trombino", "Ingredients": "prosciutto, provolone cheese", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.689300427589949, "Sandwich": "Fellini", "Ingredients": "sopressata, fresh mozzarella, hot peppers, arugula", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.857241094117636, "Sandwich": "Puccini", "Ingredients": "sopressata, fresh mozzarella, sun dried tomatoes, arugula", "Vegetarian": false}, {"Price": 11.5, "Prediction": 11.317357232853222, "Sandwich": "Dondolo", "Ingredients": "sopressata, fresh mozzarella", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.883947104278862, "Sandwich": "Frugoletto", "Ingredients": "sopressata, smoked mozzarella, artichokes, arugula", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.638204342494845, "Sandwich": "Geppetto", "Ingredients": "sopressata, caponata of eggplant, m. bel paese, arugula", "Vegetarian": false}, {"Price": 12.5, "Prediction": 12.444563728821272, "Sandwich": "Verona", "Ingredients": "sopressata, sweet roasted peppers, m. bel paese", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.746402664501133, "Sandwich": "Alyssa", "Ingredients": "smoked chicken breast, fresh mozzarella, arugula, dressing", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.963599408104194, "Sandwich": "Marina", "Ingredients": "smoked chicken breast, fresh mozzarella, sun dried tomatoes, arugula", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.753675006278412, "Sandwich": "Romeo", "Ingredients": "smoked chicken breast, hot peppers, m. bel paese, arugula", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.9305300278815, "Sandwich": "Brando", "Ingredients": "smoked chicken breast, provolone cheese, sweet roasted peppers, arugula", "Vegetarian": false}, {"Price": 12.5, "Prediction": 12.429574963051147, "Sandwich": "Alidoro", "Ingredients": "smoked chicken breast, dressing, arugula", "Vegetarian": false}, {"Price": 12.5, "Prediction": 12.676217930183535, "Sandwich": "Collodi", "Ingredients": "smoked chicken breast, artichokes, arugula", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.43638922050792, "Sandwich": "Daniela", "Ingredients": "tuna, fresh mozzarella, hot peppers, arugula", "Vegetarian": false}, {"Price": 12.0, "Prediction": 12.362958233411785, "Sandwich": "Rombolino", "Ingredients": "tuna, fresh mozzarella, arugula", "Vegetarian": false}, {"Price": 12.5, "Prediction": 12.191652521739243, "Sandwich": "Figaro", "Ingredients": "tuna, sweet roasted peppers, m. bel paese", "Vegetarian": false}, {"Price": 12.0, "Prediction": 12.167876705205304, "Sandwich": "Vivaldi", "Ingredients": "tuna, provolone cheese, caponata of eggplant", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.34112331913571, "Sandwich": "Melody", "Ingredients": "tuna, artichokes, dressing, arugula", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.330613930155831, "Sandwich": "Sofia", "Ingredients": "salami, fresh mozzarella, artichokes, olive paste", "Vegetarian": false}, {"Price": 15.0, "Prediction": 15.16938606984413, "Sandwich": "Pavarotti", "Ingredients": "salami, smoked mozzarella, sun dried tomatoes, artichokes, sweet roasted peppers", "Vegetarian": false}, {"Price": 14.0, "Prediction": 13.999999999999982, "Sandwich": "Marcello", "Ingredients": "capicollo, fresh mozzarella, sun dried tomatoes, arugula", "Vegetarian": false}, {"Price": 13.5, "Prediction": 13.499999999999993, "Sandwich": "Fiorello", "Ingredients": "mortadella, fresh mozzarella, caponata of eggplant", "Vegetarian": false}, {"Price": 13.0, "Prediction": 12.999999999999995, "Sandwich": "Da Vinci", "Ingredients": "sardines or mackerel, sun dried tomatoes, m. bel paese", "Vegetarian": false}, {"Price": 11.0, "Prediction": 11.684611037723371, "Sandwich": "Gabriella", "Ingredients": "fresh mozzarella, dressing, arugula", "Vegetarian": true}, {"Price": 12.0, "Prediction": 11.829626358968756, "Sandwich": "Cabiria", "Ingredients": "fresh mozzarella, sweet roasted peppers, arugula", "Vegetarian": true}, {"Price": 12.0, "Prediction": 11.733867114798745, "Sandwich": "Marco Polo", "Ingredients": "fresh mozzarella, hot peppers, arugula", "Vegetarian": true}, {"Price": 13.0, "Prediction": 12.971904400666183, "Sandwich": "Mona Lisa", "Ingredients": "fresh mozzarella, artichokes, caponata of eggplant, m. bel paese", "Vegetarian": true}, {"Price": 10.5, "Prediction": 10.632741797215168, "Sandwich": "Casanova", "Ingredients": "fresh mozzarella, artichokes", "Vegetarian": true}, {"Price": 13.0, "Prediction": 13.097704022753803, "Sandwich": "Valentino", "Ingredients": "smoked mozzarella, artichokes, sweet roasted peppers, arugula", "Vegetarian": true}, {"Price": 12.0, "Prediction": 11.722014551633634, "Sandwich": "Scorsese", "Ingredients": "smoked mozzarella, caponata of eggplant, arugula", "Vegetarian": true}, {"Price": 12.0, "Prediction": 11.871373237470891, "Sandwich": "Sinatra", "Ingredients": "smoked mozzarella, artichokes, sun dried tomatoes", "Vegetarian": true}, {"Price": 10.5, "Prediction": 10.600555360317742, "Sandwich": "Pacino", "Ingredients": "smoked mozzarella, sun dried tomatoes", "Vegetarian": true}, {"Price": 13.0, "Prediction": 13.024988188295744, "Sandwich": "Ortoletto", "Ingredients": "sun dried tomatoes, artichokes, sweet roasted peppers, arugula", "Vegetarian": true}], "data-edebebbba8106f8ffacea64eec8a8b00": [{"x": 10.0, "y": 10.0}, {"x": 10.055, "y": 10.055}, {"x": 10.11, "y": 10.11}, {"x": 10.165, "y": 10.165}, {"x": 10.22, "y": 10.22}, {"x": 10.275, "y": 10.275}, {"x": 10.33, "y": 10.33}, {"x": 10.385, "y": 10.385}, {"x": 10.44, "y": 10.44}, {"x": 10.495, "y": 10.495}, {"x": 10.55, "y": 10.55}, {"x": 10.605, "y": 10.605}, {"x": 10.66, "y": 10.66}, {"x": 10.715, "y": 10.715}, {"x": 10.77, "y": 10.77}, {"x": 10.825, "y": 10.825}, {"x": 10.88, "y": 10.88}, {"x": 10.935, "y": 10.935}, {"x": 10.99, "y": 10.99}, {"x": 11.045, "y": 11.045}, {"x": 11.1, "y": 11.1}, {"x": 11.155, "y": 11.155}, {"x": 11.21, "y": 11.21}, {"x": 11.265, "y": 11.265}, {"x": 11.32, "y": 11.32}, {"x": 11.375, "y": 11.375}, {"x": 11.43, "y": 11.43}, {"x": 11.485, "y": 11.485}, {"x": 11.54, "y": 11.54}, {"x": 11.595, "y": 11.595}, {"x": 11.65, "y": 11.65}, {"x": 11.705, "y": 11.705}, {"x": 11.76, "y": 11.76}, {"x": 11.815, "y": 11.815}, {"x": 11.870000000000001, "y": 11.870000000000001}, {"x": 11.925, "y": 11.925}, {"x": 11.98, "y": 11.98}, {"x": 12.035, "y": 12.035}, {"x": 12.09, "y": 12.09}, {"x": 12.145, "y": 12.145}, {"x": 12.2, "y": 12.2}, {"x": 12.254999999999999, "y": 12.254999999999999}, {"x": 12.31, "y": 12.31}, {"x": 12.365, "y": 12.365}, {"x": 12.42, "y": 12.42}, {"x": 12.475, "y": 12.475}, {"x": 12.53, "y": 12.53}, {"x": 12.585, "y": 12.585}, {"x": 12.64, "y": 12.64}, {"x": 12.695, "y": 12.695}, {"x": 12.75, "y": 12.75}, {"x": 12.805, "y": 12.805}, {"x": 12.86, "y": 12.86}, {"x": 12.915, "y": 12.915}, {"x": 12.97, "y": 12.97}, {"x": 13.025, "y": 13.025}, {"x": 13.08, "y": 13.08}, {"x": 13.135, "y": 13.135}, {"x": 13.19, "y": 13.19}, {"x": 13.245000000000001, "y": 13.245000000000001}, {"x": 13.3, "y": 13.3}, {"x": 13.355, "y": 13.355}, {"x": 13.41, "y": 13.41}, {"x": 13.465, "y": 13.465}, {"x": 13.52, "y": 13.52}, {"x": 13.575, "y": 13.575}, {"x": 13.629999999999999, "y": 13.629999999999999}, {"x": 13.685, "y": 13.685}, {"x": 13.74, "y": 13.74}, {"x": 13.795, "y": 13.795}, {"x": 13.85, "y": 13.85}, {"x": 13.905, "y": 13.905}, {"x": 13.96, "y": 13.96}, {"x": 14.015, "y": 14.015}, {"x": 14.07, "y": 14.07}, {"x": 14.125, "y": 14.125}, {"x": 14.18, "y": 14.18}, {"x": 14.235, "y": 14.235}, {"x": 14.29, "y": 14.29}, {"x": 14.344999999999999, "y": 14.344999999999999}, {"x": 14.4, "y": 14.4}, {"x": 14.455, "y": 14.455}, {"x": 14.51, "y": 14.51}, {"x": 14.565000000000001, "y": 14.565000000000001}, {"x": 14.620000000000001, "y": 14.620000000000001}, {"x": 14.675, "y": 14.675}, {"x": 14.73, "y": 14.73}, {"x": 14.785, "y": 14.785}, {"x": 14.84, "y": 14.84}, {"x": 14.895, "y": 14.895}, {"x": 14.95, "y": 14.95}, {"x": 15.004999999999999, "y": 15.004999999999999}, {"x": 15.059999999999999, "y": 15.059999999999999}, {"x": 15.115, "y": 15.115}, {"x": 15.17, "y": 15.17}, {"x": 15.225, "y": 15.225}, {"x": 15.280000000000001, "y": 15.280000000000001}, {"x": 15.335, "y": 15.335}, {"x": 15.39, "y": 15.39}, {"x": 15.445, "y": 15.445}, {"x": 15.5, "y": 15.5}]}}, {"mode": "vega-lite"});
</script></pre></code>



{{% jupyter_cell_end %}}