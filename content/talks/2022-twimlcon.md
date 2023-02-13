
---
title: "Closing the Loop: Automated Model Retraining and Deployment"
slug: "twimlcon-2022"
draft: false
hideDate: true
slides: true
---

Talk for [TWIMLCon 2022](https://twimlai.com/conf/twimlcon/2022/).

### Abstract

It's hard enough to train and deploy a machine learning model to make real-time predictions. By the time a model's out the door, most of us would rather move on to the next model. And maybe that is what most of us do, until a couple months or years pass and the original model's performance has steadily decayed over time. The simplest way to maintain a model's performance is to retrain the model on fresh data, but automating this process is nontrivial. 


In this talk, I will present a case study showing how my team automated the model retraining process for a language model that powers a key product feature. In particular, I will describe the framework through which we are able to automate the deployment of the freshly trained model in a safe, staged manner. Central to this framework is the idea that retrained models are state machines, and models change states based on metrics calculated from the ecosystem that they operate within. After this talk, the audience will be able to apply the same state machine framework to their own machine learning model lifecycle.


<div class="google-slides-container">
  <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRsUxgkcfitx2uaSKVakr44uA2EBo5QpHFp5seN3UMD8L4tS01hm8Cy2wtBoE4dG2Yd2xLM7jrivvO3/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</div>
