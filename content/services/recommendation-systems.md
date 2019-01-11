---
title: "Recommendation Systems"
date: "2019-01-11"
draft: false
hideDate: true
---

Recommendation systems have become integral to personalizing and driving increased value for consumer technology products. Spotify's Discover Weekly, Netflix's homepage, and the "Customers who bought this item also bought" section of Amazon product pages are all examples of successful recommendation systems that continue to provide incremental revenue. I have built numerous recommendation systems for companies and seen first-hand the additional revenue that these systems provide. 

I have built [simple systems]({{< ref "/blog/intro-to-collaborative-filtering/index.md" >}}) that recommend similar products or [personalized recommendations]({{< ref "/blog/implicit-mf-part-1/index.md" >}}) for particular users. I have also worked on the state of the art in recommendation systems to realize their full potential. For example, it can be difficult to handle brand new customers or products (the so-called "cold start" problem), but it is possible to utilize [extra information]({{< ref "/blog/implicit-mf-part-2/index.md" >}}) to maintain quality recommendations. Deep learning techniques can be particularly powerful for [visual similarity recommendations]({{< ref "/blog/recasketch-keras/index.md" >}}). 

Sometimes recommendation systems are used to influence human decision making in a "human-in-the-loop" process. In such scenarios, it is crucial to provide context about _why_ a recommendation is relevant to a user. One sees this context on the Netflix homepage (e.g. "Because you movies with a strong female lead"), and I have worked extensively on these problems for private companies. Publicly, I have given [talks](https://youtu.be/Pm4ZQMKoz7Q) on this topic, and it remains an active area of interest for me.
