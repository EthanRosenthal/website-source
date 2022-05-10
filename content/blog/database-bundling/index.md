---
date: 2022-05-10
draft: false
hasMath: false
notebook: false
slug: "database-bundling"
tags: ['machine-learning-engineering', 'mlops']
title: "Let's Continue Bundling into the Database"   
---

A very silly blog post came out a couple months ago about [The Unbundling of Airflow](https://blog.fal.ai/the-unbundling-of-airflow-2/). I didn't fully read the article, but I saw its title and skimmed it enough to think that it might've been too thin of an argument to hold water but just thick enough to clickbait the VC world with the word "unbundling" while simultaneously [Cunningham's Law-ing](https://meta.wikimedia.org/wiki/Cunningham%27s_Law) the data world. There was certainly a Twitter [discourse](https://twitter.com/search?q=unbundling%20airflow&src=typed_query).

They say imitation is the sincerest form of flattery, but I don't know if that applies here. Nevertheless, you're currently reading a blog post about data things that's probably wrong and has the word "~~un~~bundling" in it. 

I actually don't care that much about the bundling argument that I will make in this post. Truthfully, I just want to argue that [feature stores]({{< ref "/blog/2021-02-03-feature-stores-as-data-warehouse" >}}), [metrics layers](https://benn.substack.com/p/metrics-layer), and [machine learning monitoring](https://www.shreya-shankar.com/rethinking-ml-monitoring-1/) tools are all abstraction layers on the same underlying concepts, and 90% of companies should just implement these "applications" in SQL on top of streaming databases.

I basically want to argue this [Materialize](https://materialize.com/) quote-tweet of [Josh Wills'](https://twitter.com/josh_wills) second viral tweet:

<blockquote class="twitter-tweet"><p lang="und" dir="ltr"><a href="https://t.co/0JfQeATA8Y">https://t.co/0JfQeATA8Y</a> <a href="https://t.co/VD72F9UqY8">pic.twitter.com/VD72F9UqY8</a></p>&mdash; Materialize (@MaterializeInc) <a href="https://twitter.com/MaterializeInc/status/1508762505547554816?ref_src=twsrc%5Etfw">March 29, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

And I _will_ argue this. You, however, will have to first wade through my bundling thesis so that I can make my niche argument a bit more grandiose and thought leader-y.

# [MAD World](https://youtu.be/ccWrbGEFgI8)

While I don't really buy the unbundling Airflow argument, I do admit that things are not looking great for Airflow, and that's great! In this golden age of Developer Experience™, Airflow feels like the DMV. A great Airflow-killer has been [dbt](https://www.getdbt.com/). Instead of building out gnarly, untestable, brittle Airflow DAGs for maintaining your data warehouse, `dbt` lets you build your DAG purely in SQL. Sure, you may need to schedule `dbt` runs with Airflow, but you don't have to deal with the mess that is Airflow orchestration.

If anything, `dbt` bundled everybody's custom, hacky DAGs that mixed SQL, python, bash, YAML, XCom, and all sorts of other things into straight SQL that pushes computation down to the database. This is a lovely bundling, and people are clearly [very happy](https://www.getdbt.com/blog/next-layer-of-the-modern-data-stack/) about it. On top of this, we now have the Modern Data Stack™ which means that we have connectors like [Fivetran](https://www.fivetran.com/) and [Stitch](https://www.stitchdata.com/) that help us to shove all sorts of disparate data into [Snowflake](https://www.snowflake.com/), the cloud data warehouse du jour.

My blog (and I) tend to skew fairly machine learning-y, so for those unfamiliar -- everything I've described constitutes the wonderful new world of Analytics Engineering™, and it really is a fantastic time to be an Analytics Engineer. There's a clearly defined stack, best practices exist, and a single person can stand up a massively scalable data warehouse at a young startup. This is a perfect match for [consulting]({{< ref "/blog/freelance-ds-consulting" >}}), by the way.

And then there's the machine learning engineering world, or MLOps™ as it's regrettably being called. I can confidently say that there's no Modern ML Stack. This field is a clusterfuck that's surely in need of some bundling, so I propose that we ride the `dbt` train and consolidate into the database.

{{< figure src="Data-and-AI-Landscape-2021-v3-small.jpg" link="https://mattturck.com/data2021/" caption="The 2021 Machine Learning, AI and Data (MAD) Landscape" >}}

# [It's my data and I need it now!](https://youtu.be/HX0fIi3H-es)

Coming down from the lofty thought leader clouds, I'm going to pick ML monitoring as my straw man case study since this is an area that I'm personally interested in, having unspectacularly [failed]({{< ref "/blog/starting-shutting-quickly" >}}) to launch a startup in this space a couple years ago. For the sake of this blog post, ML monitoring involves monitoring the _performance_ or _quality_ of your machine learning model. At their core, ML models predict _things_, and then we later find out information about those _things_. An ad ranking model may predict that a user will click on an ad, and then we later fire an event if the user clicked. A forecasting model may predict the number of units that will be sold next month, and at the end of next month we calculate how many units were sold. The model performance corresponds to how accurately the predictions match the eventual outcomes, or _ground truth_.

The way most ML monitoring tools work is that you have to send them your model predictions and send them your ground truth. The tools then calculate various performance metrics like accuracy, precision, and recall. You often get some nice plots of these metrics, you can slice and dice along various dimensions, setup alerts, etc...

While this is all well and good and there are many [companies](https://gantry.io/) [raising](https://www.arthur.ai/blog/announcing-series-a) [millions](https://techcrunch.com/2021/11/04/whylabs-raises-10m-series-a-for-its-ai-observability-platform/) of [dollars](https://techcrunch.com/2021/09/28/battery-ventures-leads-arize-ais-19m-round-for-ml-observability/) to do this, I would argue that standard ML performance metrics (the aforementioned accuracy, precision, recall, and so on) are _not_ the best metrics to be measuring. Ideally, your ML model is [impacting the world]({{< ref "/blog/2022-01-18-autoretraining-is-easy" >}}) in some way, and what you _actually_ care about are some business metrics that are either related to or downstream of the model. For a fraud detection model, you may want to weight your false positives and false negatives by some cost or financial loss. In order to monitor these other business metrics, you need to get that data into the ML monitoring system. And this is annoying.

For every ML model that I've ever deployed, I end up writing complex SQL that joins across many tables. I may even need to query multiple databases and then stitch the results together with Python. I don't want to have to send all of my data to some third party tool; I'd rather the tool come to my data.

# Glorified ETL

I ran a [survey]({{< ref "/blog/production-ml-survey" >}}) a couple years ago, and many people either manually monitored their ML models or automated a SQL query or script. If that's the case, then why use a SaaS company for monitoring?

You could make the argument that an ML monitoring tool provides a unified way to monitor ML models such that you don't have to rewrite SQL queries and reinvent the wheel for each new model that you deploy. You know, mo' models, mo' problems.

I don't think this is where the value lies, and I don't think this is what you should be doing. For any new type of model "use case" that you deploy, you should probably be thinking deeply about the metrics that matter for measuring the model's business impact rather than resting on the laurels of the bare F1 score. 

I would argue that the real benefit of ML monitoring tools is that they provide advanced infrastructure that allows you to monitor performance in real time. This is a requirement for models that are generating real time predictions and carry enough risk that you need to monitor the model quicker than some fixed schedule. To monitor in close to "real time", you will need a system that does two specific things with low latency:

1. Deduplicate and join high cardinality keys for data that comes from different sources and arrives at different points in time.
1. Calculate aggregate statistics over specific time windows. 

Let me be more concrete. Imagine I build a conversion prediction model for my SaaS company. Users land on the landing page, they give me their email address, and then some of them purchase a subscription while others don't. I may store a lead's email address in Postgres, I predict whether they'll convert and fire an event to my eventing system, and then I record their conversion information in Postgres. 

If I want to calculate the accuracy of my conversion prediction model, I need to join each lead's data between Postgres and wherever my events live. The conversion information may occur much later than either the lead or prediction information. I then need to aggregate this data across some window of time in order to calculate the model accuracy.

In SQL, the trailing 3 hour accuracy would look something like

```sql
SELECT
  -- Count Distinct leads since we will could fire 
  -- duplicate prediction events.
  COUNT(DISTINCT 
    IFF(
        -- True Positive
        (predictions.label = 1 AND conversions.lead_id IS NOT NULL)
        -- True Negative
        OR (predictions.label = 0 AND conversions.lead_id IS NULL)
        , leads.id
        , NULL
    )
  )::FLOAT / COUNT(DISTINCT leads.id) AS accuracy
FROM leads
-- Predictions may come from some event system, whereas 
-- leads + conversions may be in the OLTP DB.
JOIN predictions ON predictions.lead_id = leads.id
LEFT JOIN conversions ON conversions.lead_id = leads.id
WHERE
  leads.created_at > NOW() - INTERVAL '3 hours'
  -- NOTE: we are ignoring the bias incurred by recent leads 
  -- which likely have not had time to convert since we're 
  -- assuming that they did not convert. When the lack of a 
  -- prediction label is a label, time plays an important role!
```

Yes, I can shove all my data into Snowflake and run a query to do this. But, I can't do this for recent data with low enough latency that this feels like a real time calculation. ML monitoring tools do this for you. 

# Humble DB Bundle

If we zoom back out to other areas that I mentioned earlier in this post, such as feature stores and metrics layers, we can see that they have the same 2 requirements as an ML monitoring system. 

Imagine we build a feature in our feature store for the conversion prediction model which is "the trailing 3 hour conversion rate for leads referred by Facebook". This is an aggregate statistic over a 3 hour time window. For training data in our feature store, we need to calculate this feature value at the time of previous predictions. You don't want to leak future data into the past, and this is easy to do when calculating conversion rates since conversions happen some time after the prediction time. For real time serving of the feature, we need to calculate the _current_ value of this feature with low latency.

So how do we do these calculations? We can use separate vendors for each use case (ML monitoring, feature store, metrics layer, etc...). We could also try to build a general system ourselves, although it's honestly not super easy right now. Next generation databases like [ClickHouse](https://clickhouse.com/) and [Druid](https://druid.apache.org/) are options. I'm particularly excited about [Materialize](https://materialize.com/). It's simple like Postgres, it's easy to integrate with various data sources, and they're making moves to [horizontally scale](https://materialize.com/materialize-unbundled/). Lest you think I'm just a crank, Materialize recently wrote a [blog post](https://materialize.com/real-time-feature-store-with-materialize/) (and provided [code](https://github.com/MaterializeInc/demos/tree/main/feature-store)) about a proof of principle feature store built on Materialize (though, this does not prove that I'm _not_ a crank).

To be clear, I'm not trying to pull a "[build your own Dropbox](https://news.ycombinator.com/item?id=9224)". All these vendors surely provide significant value. Materialize likely won't be able to satisfy some teams' latency requirements, there are complicated issues around ACLs and PII, intelligent alerting is tough, there are smart things you can do on top of these systems like model explainability, etc... My argument is that there are likely a decent number of companies that can get by _just fine_ with building their own system on a performant database. Only some fraction of companies need to solve these niche problems, and it's an even smaller fraction that have more stringent requirements than a streaming database can solve for.

# Pure Software

If a streaming database becomes the substrate on which to build things like ML monitoring and feature stores, then how should we build such tools? If you take out the infrastructure piece that is a core part of vendor's solutions right now, then what's left? I guess it's largely software that's left? Vendors can (and will) provide this software, but that decreases their value proposition. 

 You can write your own code to manage your own ML monitoring system. I think this scenario seems ripe for an open source framework. I'd love to see some opinionated framework around how to build an ML monitoring tool in SQL. Get me setup with best practices, and allow me to avoid some boilerplate. I guess you can monetize by offering a managed solution if you really need to. I don't, so maybe I'll try to put some open source code where my mouth is.

--

_Author's Notes:_
- _I should note that I've never _actually_ used `dbt`. As a [former physicist](https://physicstoday.scitation.org/doi/10.1063/1.1564350), I do feel confident that I understand this area outside of my expertise well enough to opine on it, though._
- _Speaking of unfound confidence, you'll note that I didn't really delve into metrics layers. I don't know that much about them, but it feels like they fit this same paradigm as feature stores and ML monitoring tools? Besides, the analytics engineering world is much bigger than the ML world, so mentioning metrics layers increases the top of my blog funnel._
