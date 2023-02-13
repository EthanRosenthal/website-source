
---
title: "Beyond Static Open Data"
slug: "nyc-open-data-2022"
draft: false
hideDate: true
slides: true
---

Talk for [NYC Open Data Week 2022](https://archive.open-data.nyc/).

### Abstract

Most open data is static. It often corresponds to either a snapshot in time or some historical summary. While static data is surely useful, looking at how data changes over time opens up new avenues for exploration. Looking backward, we can identify trends and garner insights. Looking forward, we can generate forecasts and try to predict the future. 

Citi Bike is the primary bikeshare in NYC, and they [open up](https://ride.citibikenyc.com/system-data) a lot of their data. They publish datasets about trips that riders have taken, and they have a real-time API that publishes the current information about all stations, such as the number of bikes and docks available. However, this data is largely static. If I want to answer questions like, “Will there be a dock available for me by the time I get to my destination station?”, I need to be able to forecast the number of docks at the destination station. And for that forecast, I need a time series of historical data about the number of docks at that station in order to build a forecasting model.

To answer such questions, I started pinging the Citi Bike API every 2 minutes back in 2016, and I have been collecting this data ever since. Data from August 2016 to December 2021 is publicly available on [Kaggle](https://www.kaggle.com/rosenthal/citi-bike-stations). In this event, I will show how the data collection system works, and how I keep its operations cheap and worry-free. This data collection system can be reused for other open, real-time APIs. I’ll then show how we can analyze and visualize the data in order to learn about different Citi Bike stations in NYC. Finally, I’ll answer my original question by building a model to forecast the number of bikes available at a given station.

<div class="google-slides-container">
  <iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQ3h1088au7ovcBUmIrIE8JC8hPSyxkK3WFlZNemEYsyl2bMZifBFLXqOlY5kPLNMlPGfGJX_3Lfn2Z/embed?start=false&loop=false&delayms=3000" frameborder="0" width="1280" height="749" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</div>
