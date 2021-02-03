---
date: "2021-02-03"
slug: "feature-stores-self-service"
title: "Feature Stores for Self-Service Machine Learning"
tags:
  - data science industry
  - mlops
---

Features stores are now becoming a _thing_. 

Google Cloud is [supporting](https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning) Feast, an [open source](https://github.com/feast-dev/feast) feature store, AWS [announced](https://press.aboutamazon.com/news-releases/news-release-details/aws-announces-nine-new-amazon-sagemaker-capabilities) the [SageMaker Feature Store](https://aws.amazon.com/sagemaker/feature-store/) in December 2020, and [tecton.ai](https://tecton.ai) raised a [$35 Million Series B](https://www.tecton.ai/blog/feature-store-in-general-availability-series-b-funding/) in the same month.

While it's going to be a while, I think that feature stores will do to machine learning what data warehouses did to analytics. Just as any department can now calculate metrics and setup dashboards thanks to a centralized data warehouse empowering "self-service analytics", any data scientist will be able to quickly deploy machine learning models with little ops lift.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Feature Store : ML :: Data Warehouse : Analytics</p>&mdash; Ethan Rosenthal (@eprosenthal) <a href="https://twitter.com/eprosenthal/status/1354849088542535684?ref_src=twsrc%5Etfw">January 28, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<!-- Add a line break under Twitter embed. -->
<br/>

Even with all of the froth and my grandiose statements, feature stores are still new and not widely known among machine learning practitioners. What's more, the scope of what a feature store entails is ill-defined. My current gig is the first company that I've worked for with a proper feature store, and I think it's _amazing_. If you train and deploy machine learning models and have not seen the light of feature stores, then I am here to spread the gospel in this blog post.

## Future Feature Store Story

Feature stores are often explained from the data/platform engineer's point of view. This makes sense because these are the people that will need to implement or integrate with the store. However, it is the model builders that reap the rewards of the feature store. Let's don our product manager outfit and walk through a _user story_ of what it's like to interact with a feature store as a model builder:

I'm a data scientist at an evil digital advertising company, and I need to train a model to predict the likelihood that a user will click on an ad. Each time an ad is shown (an "impression"), it has a unique `impression_id` associated with it. My workflow looks something like

```python
import sql_client
import feature_store
import xgboost as xgb

query = """
  SELECT 
    impression_id
    , was_clicked AS label
  FROM impressions
"""
labels = sql_client.run_query(query)

features = feature_store.get_historical_features(
    entity="impression", entity_ids=labels["impression_id"]
)

model = xgb.XGBClassifier()
model.fit(features, labels["label"])
```

When I need to score ads in production, the exact features that I trained on are available in real time

```python
import feature_store

def predict(model, impression_id):
    features = feature_store.get_current_features(
        entity="impression", entity_id=impression_id
    )
    return model.predict(features)
```

Before you object and complain that I have left out a significant number of implementation details, let's marinate in this ideal scenario for a spell.

...

_Ahhhhh_

...

Just imagine it, though. Let's say I need to create a model that specifically predicts click probability for German mobile sites. All I have to do is modify my `labels` query to specifically grab German mobile ads and **bam**, I have a new model in production. 

My evil advertising company also tracks users and would like to find the top N users most likely to make a purchase from an ad in the next 30 days. I can treat this as a classification problem where each row of my training data is a combination of a `user` and the `date` on which an ad was served, and the label corresponds to whether or not they made a purchase in the next 30 days. Based on a model trained on this data, I can generate predictions for all users and pick the top N scoring users. (Yes, there are many caveats and possible causal issues with this model, but we're still in "ignoring implementation" land.)

Again, I only have to modify my `label` query to pull down (`user`, `date`) combos. My feature store will make sure to return the features associated with the `user` at the time of the `date` in question:

```python
query = """
  SELECT 
    impressions.user_id
    , impressions.created_at::DATE AS impression_date
    , conversions.user_id IS NOT NULL AS converted
  FROM impressions
  LEFT JOIN conversions 
    ON conversions.ad_id = impressions.ad_id
    AND conversions.user_id = impressions.user_id
    AND conversions.created_at - impressions.created_at < interval '30 day'
"""
labels = sql_client.run_query(query)

features = feature_store.get_historical_features(
    entity="user", 
    entity_ids=[labels["user_id"], labels["impression_date"]]
)
model = xgb.XGBClassifier()
model.fit(features, labels["label"])
```

Are you starting to see? With the feature store, the bulk of my work involves:

1. Generating labels for a training dataset.
2. Deciding what features to use (although XGBoost is quite good at ignoring features that are not useful).

And that's it. 

The implication is that I can now quickly deploy ML models to target all sorts of patterns, as long as I can find well-defined labels. As we'll later discuss, this opens up a whole new slew of problems.


## Taking Inventory

The above stories purposefully omitted any details about what the feature store contains or how it works. Let's take a stab at the first piece. What features might you use to train a model that predicts the probability that an ad will be clicked? Those are the features we would like our feature store to have. Features related to previous clicks are going to be quite useful:

- User's previous ad click rate
- The ad's previous ad click rate
- The website's overall ad click rate

Maybe we want to pick up on whether these click rates are trending up or down, so we time-box these features:

- User's ad click rate over the last week
- The ad's ad click rate over the last day.

There are thousands of other features that we could think of. One way to organize is to think about which _entities_ these features relate to. We may have `user` features, like the user's device type, the time since the user last clicked on an ad, and the user's IP location. `ad` features could include the ad category ("automotive"), the time since the ad first ran, and the type of ad ("banner"). `website` features may include the website language and features related to the content on the website.

Anybody who has calculated ML features before knows that some of the above features can be quite complicated to calculate. Take the ad's click rate. A common mistake is using the following SQL query to build this feature for your training data, and then joining to your training data on `ad_id`:

```mysql
SELECT
  ad_id
  , SUM(was_clicked)::FLOAT / COUNT(*) AS click_rate
FROM impressions
```

The problem is that the above query calculates the ad's click rate across _all time_. When you are scoring real-time ads, you will not know future information about the click rate. The way to remedy this is to calculate the value of the click rate at the time that the impression is shown:

```mysql
SELECT
  impression_id
  , SUM(was_clicked)::FLOAT / COUNT(*)
    OVER (
      PARTITION BY ad_id ORDER BY created_at ASC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      EXCLUDE CURRENT ROW
    ) 
    AS click_rate
FROM impression_id
```

This is one of the requirements that the feature store satisfies: it makes your training features available to you such that the feature values correspond to exactly what was known at the time of the training sample (aka the row in your feature matrix). Feature stores make sure not to leak future information into the past.

The other requirement that we have for the feature store, for real-time predictions in particular, is that I should be able to query for these very same features' _current_ values. The idea here is to reduce "train test skew". We want to use training features that directly correspond to the features that we would expect to have when predicting in real time.

## Logistics

How does a feature store work under the hood? Hell if I know. I'm just an end user.

At the very least, I do know that there are two basic requirements:

1. A data store for large amounts of historical training data.
1. A data store for fast access to real-time data.

For `1`, a data warehouse works, or flat files in cloud storage. For `2`, a high performance key-value store such as DynamoDB might make sense.

Where the data is stored is less interesting and difficult than how the data is stored and how to get the data in there. Some feature store "solutions" only offer a place to store already calculated features. It's up to you to do the calculations elsewhere prior to insertion to the store. Other solutions provide a framework for defining the calculations that transform raw data into features. Think of how you can choose different types of metrics in Datadog, such as a rate or a count. You likely want generalized methods for calculating different types of features.
A framework such as this is critical to lowering the barrier to feature creation and ensuring high quality features.

For any feature that you build, you have to make sure that there is parity in the way it is calculated for both historical training data and real-time data. Training data for newly created features must be backfilled. All of these constraints limit the expressiveness of features that you can create.

While these are unavoidable logistical hurdles, there is real reward for those that succeed. Each feature added to an entity stands to improve _all models_ that relate to that entity. For an entity like `user`, an organization may have thousands of models that benefit from new `user` features. Feature stores are high leverage.

## Growing Pains

What follows a feature store? All of the pains of having a new, critical piece of infrastructure. With the proliferation of features, you will want to organize and document them. The feature store may become bloated, and you will find yourself with the positive problem of feature selection. You will have to figure out how to version your features because each feature will have so many downstream model dependencies. Deleting features will be particularly difficult. Maintaining high feature quality will be important and a challenge.

The feature store will accelerate the number of models in production at your company, and you will then have to build out automation for managing the lifecycle of models. If you only have a couple models in production, each data scientist can handle each model's lifecycle, such as retraining and [monitoring]({{< ref "/blog/starting-shutting-quickly" >}}). This manual management becomes untenable as the number of models per data scientist grows.

As people rely on the feature store, the boundary between the feature store and the model will blur. Should your feature store serve up text embeddings, or do those belong in your model? Should categorical features be cleaned up ("I.B.M" -> "IBM") prior to insertion to the feature store, or should the cleaning be up to the end user's model? 

## Commoditization

Developing a machine learning model requires iterating over some search space to determine the optimal parameters. That search space often includes types of models and their associated hyperparameters. There are many more knobs to tune. You can modify the training data that you use similar to fine tuning, you can iterate on objective functions, you can change neural net architectures, you can use different features, and you can use different feature engineering algorithms. With lots of horizontally scalable compute, we can afford large search spaces when the potential model improvements are proportional to the model's impact. Being able to quickly define new features in a feature store will be essential for executing large scale model development experiments.

I often roll my eyes at AutoML, but, with a proper feature store, it makes a lot of sense. It's up to you, the modeler, to define the ground truth labels and to think through the trickiness of evaluating your model and managing its lifecycle. For many problems, everything in between is becoming a search space for a computer to declaratively explore.
