---
title: "Optimization, Automation, and Logistics"
date: "2019-01-11"
draft: false
hideDate: true
---

Employees at early-stage startups are scrappy and figure out ways to manually solve problems. For example, one may have to schedule some part-time workers' hours given their availabilities and the company's needs. Media buyers may need to allocate their marketing budget across different channels. While manually solving these problems is often suitable in the short-term, eventually the processes become untenable and inefficient as the organization grows. Scheduling 15 employees is fine. Scheduling 150 is quite painful. Running SQL queries and hoping to manually balance everything in a spreadsheet only works for so long.

I have built a number of fully-automated solutions to these sorts of problems, from inventory allocation to employee scheduling. These problems were [particularly common]({{< ref "/blog/lets-talk-or/index.md" >}}) among the ecommerce companies that I worked at because these companies sold "boxes" of merchandise. This requires one to figure out how to generate optimal assortments of products given various constraints (such as inventory levels in the warehouse, customer preferences, etc...). Not only do these solutions save significant employee time, they solve these problems better than humans by optimally optimizing [whatever you want]({{< ref "/blog/towards-optimal-personalization/index.md" >}}). For example, I gave a [talk](https://www.datacouncil.ai/speaker/scaling-personalization-via-machine-learned-assortment-optimization) on building a system to maximize merchandise revenue.
