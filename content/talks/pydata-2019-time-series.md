
---
title: "Time series for scikit-learn people"
slug: "pydata-2019-time-series"
draft: false
hideDate: true
slides: true
---

Talk for [PyData 2019](https://pydata.org/nyc2019/schedule/presentation/15/time-series-for-scikit-learn-people/).

### Description

This talk will frame the topic of time series forecasting in the language of machine learning. This framing will be used to introduce the skits library which provides a scikit-learn-compatible API for fitting and forecasting time series models using supervised machine learning. Finally, a real-world deployment of skits involving thousands of forecasts per hour will be demonstrated.

### Abstract

Time series forecasting and machine learning are often presented as two entirely separate disciplines in the data science literature. When first learning about these topics, I distinctly recall wondering, “Where does machine learning end and time series begin?”, and “How do I use features in a time series model?”. This talk will answer these questions by marrying the concepts of time series and machine learning. I will do so by framing time series in a language familiar to anyone who is comfortable with using scikit-learn.

This framing will motivate the introduction of the skits library. skits provides a scikit-learn-compatible API for fitting and forecasting time series models. By building off of scikit-learn, skits allows one to build robust and reproducible time series models that enjoy access to the rest of the scikit-learn ecosystem like cross validation tools, standard scoring functions, etc…

I will close by showing how skits is being used to generate thousands of forecasts per hour of the number of Citi Bikes at every station in NYC using only modest computational resources.

<div class="google-slides-container">
  <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQFf9sC6-lhXo39e6ZgBB6OIirJr90OsqLWYO2yhC30zKtOFGFNuRi4IFzZTBe6tRdquECOUwjHYetL/embed?start=false&loop=false&delayms=60000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</div>
