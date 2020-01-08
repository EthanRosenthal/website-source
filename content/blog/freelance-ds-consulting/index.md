---
date: "2020-01-08"
title: "Doing Freelance Data Science Consulting in 2019"
slug: "freelance-ds-consulting"
draft: false
---

About 15 months ago, I left my full-time job as a machine learning team lead with the goal of doing independent / freelance data science consulting. Since then, I've gotten a lot of questions about what that means and entails. I have not found too much information about this type of work, other than Greg Reda's [fantastic post](http://www.gregreda.com/2017/01/07/freelance-data-science-experience/). I hope this blog post answers some of those questions for anybody interested in becoming or hiring a data science consultant. 

Full disclosure: I only _really_ consulted for the first ~8 months. I spent the next 6 months of my time starting and stopping a startup, which will be the subject of a future post.

## What is freelancing, consulting, contracting, etc...?

I think of this work as existing on a spectrum:

contracting -> agency work -> consulting

Sorry to be confusing, but I will end up using "freelancing" and "consulting" interchangeably throughout this post to describe the work that I did which spanned this spectrum. FWIW, I would recommend calling yourself a consultant because it sounds fancier than anything else.

### Contracting

For whatever reason, it is sometimes easier or preferable for companies to hire employees who are part-time and/or are not directly employed by the company doing the hiring. For example, this is common in Washington DC where many people do not work directly for the federal government but are instead contracted out _to_ the government by private companies. For me, contracting arrangements felt fairly similar to being an employee (or fractional employee), albeit somewhat removed from day-to-day politics.

One of my first freelance gigs consisted of me being a sub-contractor for a Fortune 500 company. This company wanted some help forecasting performance of their advertising campaigns. Due to... "bureaucracy", it was difficult for this company to directly hire me as a contractor. Instead, they had an "approved vendor" company which hired me and then sub-contracted me out to the client. Actually, to be clear, this approved vendor company hired _my_ company, Rosenthal Data, LLC. So many layers of minutiae.

In this arrangement, I got a company email address, I attended weekly meetings (virtually, as this company was on the opposite coast), and I had coworkers that I worked with. As I said, it felt like being a fractional employee. I billed the approved vendor for X hours / week until my contract was up or we renewed it.

### Agency Work

Contracting work is probably the least risky, and consequently least lucrative side of freelancing. You typically sign a contract for X weeks of Y hours / week at $Z / hour. Sure, the contract can typically be ended at any time. But besides that, you can somewhat count on a steady stream of money over this time as long as the client is happy. 

I never truly understood "risk versus reward" until I started freelancing. Anytime you are willing to take on a little more risk, the potential for greater reward increases. With contracting, your time is directly coupled to your income. You can ask for higher and higher hourly rates (and you should!), but all potential clients will balk at some rate. 

One of the things that bothered me about this setup was that the incentives are poorly aligned. If you are charging hourly and get twice as fast at your job, you will earn half as much, unless you hustle and find more clients. Yes, you can try to charge twice as much, but there is likely an upper limit to how much you can successfully charge. 

You can decouple your time from your money by switching to "project-based billing", which I think of as "agency work". In this scenario, a client hires you (or your agency) to do some project for them. This could be a small project, like an analysis, or a big project, like a production machine learning model deployment. You agree to some price for this project, and away you go. If you can agree upon a price proportional to the _value_ this project will bring, rather than the amount of work that it takes, then you stand to potentially make a lot of money. Imagine you start to build out your own code to make it quick and easy to deploy machine learning models. You could reuse this code while quickly churning out models for different projects and charge a high amount for each one. The inevitable extrapolation of this is to turn yourself from a _services_ company into a _product_ company.

Due to some failed negotiations at higher rates and an increasing appetite for risk, I sold a project-based contract. The client was a small startup that wanted to generate a time series forecast for a bunch of different "entities" each night. They came to me with this goal and a description of the data that they had. I wrote up a long proposal describing how I would go from auditing their data to deploying a nightly forecast. This proposal consisted of multiple "milestones" that I would hit (e.g. a working model prototype with achieving X performance metric), and a timeline for these milestones. I quoted a weekly rate but did not mention how many hours I expected to work. On the backend, I calculated my weekly rate by picking an hourly rate and estimating my time. This is a clear risk<>reward tradeoff. If the project took me longer to do than I planned, then I would end up eating the cost. If it took me less time, then I was able to achieve a higher effective hourly rate. 

### Consulting

When I think of consulting, I think of paying somebody a lot of money to come into your company and tell you things. This person likely does less technical work (e.g. writing code) and works fewer hours per week than contractors or agency-based freelancers. On the other hand, the consultant may charge a very high hourly rate to make up for this. 

I ended up as this version of a consultant for a couple clients. Largely due to my expertise in certain areas of data science, I had a couple clients for whom I would provide "a la carte" consulting on an ad hoc basis. These clients were free to ask questions via email or through setting up a meeting. I would provide advice, feedback, and occasional one-off analyses, and I would bill the clients for my time. Typically, I would sign contracts which allowed to the clients to reach out up to X hours per week on an as-needed basis over a period of Y months.

Consulting like this can be extremely lucrative, as you can typically charge higher hourly rates. If you make a name for yourself and become an expert in your field (or at least convince others that you're an expert), then you can earn a lot of money doing a couple hours of week per client for a relatively few number of clients.

## But what does a _data science_ freelancer do?

Much of the above may be familiar to people who work in select industries. Agencies are common in advertising. There are more contractors than government workers in DC. On the other hand, _data science_ freelancers are rare in 2019. Many people have asked me what type of work I actually did this past year. My glib answer was, "whatever people paid me to do", which was partially true.

One of the reasons I wanted to do freelancing was that I wanted a broader view of the industry. I had literally spent my entire (albeit short) data science career working at _box-based retail companies_. Thus, I was particularly interested in working with new types of businesses and new data science problem domains, particularly time series modeling. At the same time, I wanted to continue to do production machine learning work. My goals in freelancing guided what I worked on, and so my experience could very well be different than other data science freelancers.

I was reasonably successful in my goals. I did very little work with retail companies, but I did find work with both a B2B SaaS company and a university research lab. The majority of the projects that I worked on involved various forms of time series modeling, like time series forecasting and classification. I also got to deep dive into some gnarly Bayesian modeling.

### Deliverables

My deliverables were typically either jupyter notebook-based reports or full-fledged deployed models in the client's cloud. Deploying models was a great way to diversify my technical experience beyond what one can typically achieve at an individual company. At a previous job, I had used Luigi for data pipelines, but I got the opportunity to use Airflow with one of my clients. Previous jobs were primarily AWS (or not in the cloud 😬), but I got to use both GCP and Heroku with clients.

Sometimes clients would have an idea for what they want me to build ("We want to forecast this thing every night"). Other times, clients would simply have a problem, and I would come up with a solution to this problem. In either scenario, I would typically write up a proposal about the requirements and my approach for building the machine learning system. This would prove some of my expertise to the client and give the client a chance for input. Most importantly, this document would be a way to hold us both accountable as the project progressed. For example, requested changes could be clearly demarcated as "out-of-scope" if they were not included in the original proposal. At the same time, milestones that I had promised to hit were clear.

### Uncertainty

Many other data scientists who I have talked to found the prospect of doing data science consulting terrifying due to a particular line of thought. I had the same thoughts initially. The thought process is as follows:

_If somebody wants to hire you to build them an ecommerce store, then you at least know that many people have built stores before. Ecommerce stores are a known thing that have been built many times. But, if somebody asks you to build them an algorithm to predict something, you don't know if it's actually possible to build an accurate algorithm. What's worse, it will take serious time to get to the point where you can even determine the accuracy of the algorithm!_

I think there are two issues with the above line of thought. Firstly, it places too much certainty on "regular" engineering work, like building an ecommerce store. All engineering work brings uncertainty, and we are still terrible as an industry at work estimation. Secondly, as long as you communicate with the client and try to think things through in your proposal / statement of work / contract / etc..., then the uncertainty of algorithmic success is just another stage of the project.

Any Data Scientist knows that no matter what a company claims, you won't _really_ know what's going on until you start to look through the company's data. Consequently, my projects started with a data audit. The data audit consisted of me looking through the company's data, calculating some metrics, making some plots, and other analyses that seemed relevant for the eventual machine learning model that I was asked to build. The deliverable was often a jupyter notebook report. These were helpful for the clients because they often got a view into their data that they had not seen before. This process was helpful for me to both get a feel for the data and domain as well as identify any data issues. For instance, if a company always overwrites each row of their transactional database, then it may not be possible to build out historical training data!

After the audit, there was typically a model Proof of Concept (POC) period. If possible, the client and I would agree on some performance metric ahead of time that the model ought to be able to achieve. Often, it's easiest to use a simple baseline model's performance metric. A fancy time series model should be able to beat a simple moving average. The POC period is a nice alignment of incentives. The freelancer wants to achieve good model quality, otherwise they won't get to work on the rest of the project (and subsequently collect that pay). At the same time, explicit time has been carved out to acknowledge that this is an exploratory period.

## How did I find work?

Alas, I do not have a hack or silver bullet for this. The majority of my clients came through my network. I am fortunate to have worked at multiple jobs that laid people off and which I quit. This gives me a fairly wide net of former coworkers. Additionally, the alumni network of [Insight Data Science](https://www.insightdatascience.com/) is fantastic. I occasionally had clients who stumbled on this very blog and reached out for consulting help. That was super cool.

While I did not have a terrible time finding work, things likely would have gotten more difficult. Just like D2C companies who face rising customer acquisition costs as they churn through their best customers, I likely would have started to exhaust my close network and had to invest more heavily in both marketing and networking.

## What did I like?

There was a lot to like about consulting. In my short time, I got to considerably broaden my experience. I worked on new company domains, new data science techniques, new infrastructure, new frameworks, etc... 

I really enjoyed setting my own schedule. I had minimal meetings compared to full-time jobs. This allowed me to work from wherever I wanted, and I'm now a strong proponent of remote work. I got into a great groove working from my [bougie gym](https://www.chelseapiers.com/fitness/locations/chelsea/) where there is a cafe with great WiFi. When I was working full time prior to consulting, I would get to the gym in the morning, race through my workout, and then still have to shower and commute to the office. With consulting, I could workout in the morning, and then I would already be at the office. In the summertime, I could work from the gym's ridiculous outdoor cabanas that look out over the Hudson. Once noon hit, the cabanas were no longer in the shade, so I would ride my bike home, clear my head on the ride, make some lunch, and work the rest of the afternoon, recharged, from home.

{{< figure src="IMG_20190228_090132.jpg" caption="Gym Cafe" >}}
{{< figure src="IMG_20190605_100007.jpg" caption="Cabanas">}}
{{< figure src="IMG_20190903_102622.jpg" caption="\"Working\"" >}}

I met with a _lot_ of people while consulting. At the beginning, my mantra was to never say no to a meeting. Tech likes to fashion itself as introverts, so many people may be repulsed at constant meetings. On the contrary, I think it's a great setup for introverts. Meetings mean that you have focused time during which you have to socialize. All the time in between can be spent focusing on technical work. I rather liked this setup. I could meet with people for coffee, learn about all the fascinating things their company is working on, and then go home and "recharge" my social batteries. 

On that note, it was eye opening to learn about how large the business world really is. It's easy to get horse blinders when working at a company or only consuming news from certain sources. The real world is always much larger than you would expect, and industries are always deeper and more interesting than they appear.

Speaking of business, I learned a lot about business! I created an LLC, I opened a business bank account, I tracked expenses, I sent invoices, I followed up on unpaid invoices 🙄, I wrote contracts, I read too many contracts, and I withheld my own taxes. My quick recommendations for anybody looking to do all of this simply are:

- Create a quick Single-Member LLC. It's pretty cheap, gives you a little peace of mind, and it qualifies you for a business bank account.
- Pay a service to create your LLC (at least in NYC where things are a little complicated). I used [Northwest Registered Agent](https://www.northwestregisteredagent.com/). It was $300 total or something. While this might sound like a lot, your time is now literally money, and I'm sure a service can do all of this much quicker than you can in $300 worth of your time.
- Open a business bank account early so that you can put all large, initial expenses on the company credit card. For example, I got like $500 in credit card points by spending enough money in the first 3 months which was easily satisfied by buying a Macbook on the credit card.
- Use [Quickbooks Self-Employed](https://quickbooks.intuit.com/self-employed/) for accounting. It's cheap, simple, and a really good product. It will automatically match-up images of receipts with transactions on your credit card. You can also use it for invoicing clients.

Lastly, freelancing was exciting. There is a thrill to selling, negotiating, and closing deals. Also, nothing is quite as motivating as selling a project to somebody knowing that it's going to be up to you, and you alone, to figure out how the fuck to get this thing to actually work. Sure, this was stressful at times, but it also made me incredibly productive. 

## What did I not like?

There was a lot to not like about consulting. So much, in fact, that I no longer want to do it! 

### Time

When I started consulting, I had this dream of charging a lot of money per hour and working minimal hours. I think I still have this dream, but my definition of "minimal hours" has changed. 

I had hourly contracts for two different clients totaling 30 billable hours / week when I began freelancing. This is too many hours. Most people people would scoff at this and say, "I'm sorry, 30 hours is too _many_?!". In fact, we're just coming off the latest Twitter war about how many hours people in tech work. I'll consider all of the arguments moot without evidence of tracked hours and _what_ work was done during those hours.

Let me explain why 30 hours is too many. Firstly, remember that this is 30 hours that you would feel morally justified in charging $X00 / hour. Often, these hours consist of so-called [Deep Work](https://www.amazon.com/Deep-Work-Focused-Success-Distracted/dp/1455586692), and it's incredibly hard to do a lot of deep work day-in and day-out. Even so, there is a lot of other work to be done as a freelancer. You have to maintain your pipeline of clients. This involves lots of coffee meetings with potential new clients. I am lucky to live in downtown Manhattan, so I am relatively close to most meeting places. Still, I typically allot 30 minutes to travel, an hour for the coffee meeting, and 30 minutes back. Assuming I did billable work right up until I left for the meeting and immediately got back to working when I got home, this single meeting ate up 2 hours leaving me with 6 hours out of an 8 hour workday to do billable work. At 30 hours / week, you are allowed one meeting per day, assuming you are very good at context switching.

Sure, I went on fewer than one meeting per day. But, sometimes, I had multiple meetings. There is also lots of time spent writing proposals for new clients, reviewing contracts, doing accounting, and all of the other things necessary with running a business. Even if I did not have to do any of those things, I would still argue that 6 hours / day is likely the maximum rate for good, real, morally-justifiably-expensive work that anybody can sustain for an extended period of time.

It's true that tech people routinely go to the office for more than 6 hours / day. However, it's much easier to sit through a couple hours of meetings than to _really_ focus and write code for a couple hours straight. This is why understanding _what_ people are doing with their time when they claim long work days is important. There were times in grad school when I worked more than 40 hours per week, including time on weekends. While I've vowed to never do this now that I'm out of academia, this was at least somewhat sustainable (although still not recommended!) because a lot of my time was spent transferring liquid helium, waiting for data to be acquired, or doing other mindless tasks like literally watching glue dry.

Given everything above, I would argue that an ideal number of billable weekly hours is something like 20-25. With a number this low, along with potentially a lot of new expenses (e.g. health care), increased risk (future income is not guaranteed), and no paid time off, you quickly realize that you need to charge pretty high hourly rates in order for freelancing to make any financial sense compared to full-time work (and even then, salaries at big tech companies are [astronomical](https://news.ycombinator.com/item?id=21966465) right now). Many companies will balk at these rates, but you should stand firm. With all of my talk about hourly rates, you can now see why I was so interested in attempting to decouple my time from my money.

### Work

If you have not read Monica Rogati's [Data Science Hierarchy of Needs](https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007), then get out of here and go read that. I bring it up because my goal was to focus on work towards the top of her pyramid, like writing production code to repeatedly train and/or perform inference on machine learning models. There is significantly less freelance need for this type of work than work towards the bottom of the pyramid, like ETL pipelines and analytics. 

For the ML need that _did_ exist, a lot of it was misplaced. I successfully argued myself out of a job multiple times by dissuading companies from hiring me for ML and instead encouraging them to do some basic analytics or setup a data warehouse. 

For the few companies that had a legitimate ML need, I started to feel _morally bad_ about building production ML systems for them. In fact, I would stand behind the following statement for the vast majority of companies: 

>Do not hire a freelancer to build data products in 2020. 

Due to the selection bias of companies that were looking for a freelancer, they tended to be either very early stage startups or had few to zero data scientists. Building out a production ML system requires _so much fucking code right now_. Sure, companies agree to terms up front and this is all business baby, but I really loathed handing over giant piles of code + infrastructure to clients.

To be clear, I pride myself on the code that I write. I write tests, comments, and documentation. I try to keep things DRY and modular. _Regardless_. Many parts of an ML pipeline end up consisting of custom code. Everything will eventually break. The data coming into the pipeline will change. The data volume will grow. Shit will happen, the company will have to maintain this system, and it's going to be a pain to maintain. After all, [ML is the high interest credit card of technical debt](https://research.google/pubs/pub43146/). It's possible to build out tooling to mitigate these risks, but [the list of required infrastructure is long](https://research.google/pubs/pub46555/). 

The crux of this issue is that there is no Rails for ML. If you hire somebody to build you a website and they hand you a pile of Rails code, you can find many other people who will be able to take over this code and understand it. This is not the case with ML right now. It also might never be the case due to the fact that ML code is so tightly coupled with compute / infrastructure.

I also got a lot of pushback from potential clients who wanted this work to be done in house. At the time, this annoyed me (I wanted a deal!). In hindsight, this makes a lot of sense, and I have now come around to the belief that most of this work _should_ be done in house. The systems are just too complicated right now. 

A counterexample to this is that there are a number of companies for whom they basically need simple, one-off data products built. For example, imagine a small-scale ecommerce company that needs a recommendation system. Their scale is small enough that a simple recommendation system will suffice, and improvements to the algorithm will not have large absolute impact due the small scale of the company. 

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">You can have 2 DS and a 2-year plan for them, but, even if they execute flawlessly, you can end up in a place where there&#39;s not much left to do unless (1) there are new DS products to build or (2) the scale of the company warrants marginal improvements on existing DS products</p>&mdash; Ethan Rosenthal (@eprosenthal) <a href="https://twitter.com/eprosenthal/status/1135962682870378496?ref_src=twsrc%5Etfw">June 4, 2019</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

You could imagine hiring a freelancer to build such a one-off recommendation system. This is probably the wrong approach, though. One school of thought comes from [this quote](https://www.infoq.com/presentations/instrumentation-observability-monitoring-ml) by Josh Wills:

>Don't ever do one model in production, do thousands of models or zero models. If you're working on a problem, and you need to deploy to production, but you're never actually going to rebuild the model, that is a strong signal that this problem is not actually worth your time.

Alternatively, one could argue that this system _should_ be built, but not in house and not by a freelancer. You should instead pay a [vendor](https://www.recombee.com/) or some other [managed service](https://aws.amazon.com/personalize/) to both implement and maintain the data product. The industry has advanced enough now that some of these data products are starting to be commoditized.

This has all led me to postulate that the only custom data products that should be built and maintained in house should either be absolutely core to a company's profit or be products for massive scale companies such that marginal improvements have large, absolute impact. Everything else ideally should be handled by a third party which is responsible for maintenance.

Let me now hop down off my soapbox and finish this damn post. Long story short, I got somewhat jaded by _what_ data products should be built at companies, and I started to feel morally icky about dumping giant piles of ML code onto unsuspecting startups. Even if my code did successfully run day in and day out, it would still be hard for these startups to know that my models were _correct_. Just because an API serving an ML model returns a `200` status, this does not mean the model's prediction is _accurate_. Dirty data could have corrupted the previous retraining job and ruined the model's performance. I wanted a way to at least ensure that clients had visibility into their model performance, which is what I ended up spending 6 months building as a startup idea. I'll save that for the next post, though.