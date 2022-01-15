---
date: "2022-01-18"
slug: "autoretraining-is-easy"
title: "Autoretraining is Easy if You Skip the Hard Parts"
tags:
  - machine learning engineering
  - mlops
---

## You Can Not Measure What You Do Not Care To Manage

When I started my first data scientist job in 2015, the team I joined had a recommendation system that would run every night to compute new recommendations for all users of our platform. This was the easiest way to handle the [cold start](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)) problem. At the next company I worked at, we had a rule that every machine learning model must be setup to automatically retrain ("autoretrain") on fresh data on a periodic basis. Most models autoretrained nightly.

Since those startups, I've worked with or on a number of other data teams, and I've observed that autoretraining is likely more the exception than the rule. When I first started to notice this, I scoffed at the immaturity of these teams. 

As I became more acquainted with these teams' domains, I realized that I had been able to implement autoretraining years earlier because my models didn't matter. Less glib -- my models were significantly lower impact than these new teams'. 

You see, it's not _so_ hard nowadays to autoretrain a model. You have to be able to schedule and run batch jobs in the cloud, and you may need to automate the deploy of your model to some real-time prediction API. Sure, it's all still harder than it should be, but this has been doable for quite some time. However, if your model _really matters_, then you're going to need to do more. If you're training a model with the ability to spend money, and a "bad" model will incur millions in losses, then you'd better be damn sure that the new model you've trained isn't bad. 

This is where the difficulty lies. If your models have high impact, then deploying a bad model carries high risk. All of the steps to mitigate this risk are often much more difficult to implement effectively than orchestrating a retraining job.

You can't manage what you don't measure, and you can't mitigate autoretraining deployment risk without monitoring your models. And monitoring models is _hard_. It's entirely different than setting up DataDog to track your CPU utilization. Models are fuzzy, and it's often difficult to know when something systematically bad has happened to them.

This is where we round the bend and the title of this blog post comes into view. How do we make autoretraining easy? We don't monitor anything. It's that simple! Monitoring's the hard part, so let's just not do it. But how do you not monitor anything? You either write perfect code, so that there's never a bug, or you work on models that can fail. And the models that can fail are often those that don't have high impact.

## Set It and Forget It

You know, there's an odd corollary to this. Quite often, the models that have the highest impact are _not_ autoretrained. Even odder still, these models may not be monitored either, even though they are so important.

How can this be? I think this is easiest understood by breaking up "model monitoring" into [two buckets](https://twitter.com/eprosenthal/status/1465712671517458433):

- Catastrophic failure detection
- Model degradation

Catastrophic failure detection corresponds to identifying whether or not something terrible happened to your model. Maybe your classifier is predicting all `False`; maybe the data being fed into the model has been horribly corrupted; maybe you didn't hard-pin your dependencies, and a new library version has been [hacked to mine crypto](https://thehackernews.com/2021/10/popular-npm-package-hijacked-to-publish.html).

Model degradation in this case corresponds to some relation between your model's predictions and the ground truth, beyond the scenarios of catastrophic failure. This could be standard supervised ML performance metrics, like your regressor's R² or your classifier's precision and recall. Alternatively, you may look at business metrics related to the model, like the revenue-weighted mean absolute error of an inventory forecast.

While monitoring model degradation can be used for catastrophic failure detection, it's often more difficult or impossible. For example, there could be a long time delay between predictions and ground truth. You may not know the model accuracy for a couple weeks, whereas a model predicting all `False` will manifest itself immediately. 

Depending on your model deployment setup, catastrophic failures may be unlikely. Have you ever deployed a Lambda function, and it just [runs and runs](https://github.com/EthanRosenthal/citi-bikecaster) for months or years on end without you having to do anything? Maybe that's what happens when you train a model once, deploy it, and move on. In that case, maybe you don't need catastrophic failure detection; or, you may have some business alerts way downstream of your model that will detect catastrophic failure.

However, if you ever want to retrain that model, then that's going to carry a lot of risk because you're changing up this wonderful system that isn't broken. It's hard enough to manually retrain the model and _safely_ deploy it; it feels insurmountable to automate that process. You might have a bunch of ad hoc checks that you do when you manually deploy a new model, and you eventually convince yourself that it "looks good" (kind of like how you pick A/B test winners when you don't have statistical significance). Writing down your ad hoc checks in code is hard. And so you don't. 

If you empirically know that the model does not degrade in performance too quickly, then maybe you don't need high quality performance monitoring either. Running an ad hoc SQL query every couple months when you wake up in a cold sweat wondering if your model's actually still doing anything ought to suffice.

## The Easy Thing About Hard Things

Ok, so what have we learned? 

1. Monitoring models is the hard part.
1. You can skip the hard part and autoretrain your models as long as the models don't matter.
1. You can skip the hard part and _not_ autoretrain you models as long as the rest of your infrastructure is solid.

I know I'm being facetious here, but I do think it's important to point out that the above _is truly fine_ in some, if not many, scenarios. Personally, it has worked, and it continues to work in some cases. We also only have so much time, there's pressures to push out new models, and everybody continues to underestimate the enormous high interest on [ML tech debt](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). 

## The MLDLC

With all this said, I think that the goal for ML practitioners should be to setup their models to automatically retrain and to do this safely ([Level 2](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) in Google's world). The argument here is almost perfectly parallel to an ideal version of [Continuous Integration and Continuous Deployment](https://en.wikipedia.org/wiki/CI/CD)(CI/CD). CI/CD increases the [iteration velocity](https://erikbern.com/2017/07/06/optimizing-for-iteration-speed.html) of a team, and our model should not be left behind.

In modern software development, the code that is in the version control system matches the code that is running in production. That code must run through a series of tests, it then gets automatically deployed, and monitoring is in place to detect regressions and roll back to a previous, "good" version of the code if necessary.

Autoretraining models ensures that the code in version control is the code that produced the model. When manually training, there is no guarantee that _any_ of the code that produced the model is in version control. Automating the deploy of the new model reduces the risk of bugs that can be introduced by hacky, manual deployments. Adding model monitoring is like adding tests. Tests reduce the risk of changing code just as monitoring reduces the risk of changing the model. (Also, if you are committing your autoretraining code, then you will hopefully write some actual code tests!).

And finally, to stretch this a bit further, just as tests force us to write better code such that it becomes more easily testable, writing monitoring code forces us to think deeply about the model and how it impacts the system that it operates within.
