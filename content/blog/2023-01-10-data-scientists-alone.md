---
date: "2023-01-10"
slug: "data-scientists-alone"
title: "Data scientists work alone and that's bad"
tags:
  - data science industry
---


##  In Need of a Good Editr

Growing up, I had always considered myself a decent writer based on my decent grades in English class. My sophomore year English teacher made it very clear that I did not, in fact, know how to properly write. All of my essays were returned riddled with red-inked edits culminating in low scores. 

This was disheartening. Thankfully, there was a solution! These essay edits directly told me what I needed to do to improve my writing. It turned out that if I absorbed this feedback and implemented it in on my next essay, then that essay would be returned with fewer red marks and a higher score. Crazy, right? Mind blown. Stochastic Grade-ient Ascent.

## Entry Level Expert

Over a decade later, I found myself at at my first data science job where I thought I knew everything after a single year. I saw how the systems had been built and thought, "Psshhh, I can do that". I left to be the first data scientist at a Seed stage startup. I built some things and felt smart.

The company was smarter, though, since they hired a boss for me. My boss came from a more data engineering-y background, and then they hired some more data engineers. I quickly learned that my mess of spaghetti code, AWS hacking, and complete lack of tests were, perhaps, not following best practices.

My code started going through code review. Just like in high school, my Pull Requests came back riddled in comments. This experience was invaluable. I learned so many things! One big thing I learned was that if I absorbed the feedback and implemented it on my next PR, then that PR would get merged quicker. Crazy, right? Mind blown. 

## You Don't Know What You Don't Know

You see, many data scientists like myself who start with scientific programming and then find their way into software engineering are particularly dangerous. We have the ability to pretty much program whatever we want. This makes us prolific which is a terrible trait to combine with sloppy code.

Without code review, I wonder how long it would have taken me to learn things on my own? How much time would I have spent rewriting test case objects rather than using a [fixture](https://docs.pytest.org/en/latest/explanation/fixtures.html)? I didn't know that a _fixture_ was a thing, so I could not have even googled for it. There are sooooo many examples like this in programming. 

Them: "Hey, I've noticed that you've created zillions of CLI arguments for your script, and you're saving different commands that you run in a text file. Have you considered using a config file?"

Me: "What's a config file?"

## Free Solo

Why am I writing about going through code review with some engineers? It turns out that this is actually an uncommon experience for data scientists. 

The norm is that of a lonely life for the data scientist. Whether they lie near analytics, machine learning, or elsewhere in the large latent space that spans this ill-defined role, just like in the curse of high-dimensionality, they are likely alone.

Analyses, models, one-off-scripts: all of these endeavors are usually built solo. It's just hard to collaborate on training a machine learning model. And so, we often work alone. Each project is a [one-bus](https://en.wikipedia.org/wiki/Bus_factor) project. Beyond the inherent business risks that this represents, this kinda sucks for learning and development.

## Software Engineering is a Team Sport

Software engineers don't have these same problems as data scientists. Multiple engineers often work on the same codebase. Not only do they get to learn by reviewing each others' code, they spend a significant amount of time reading the existing code. This is so powerful! The codebase provides a blueprint around how to (hopefully!) successfully implement whatever the codebase does. They don't necessarily have to read the [Gang of Four](https://en.wikipedia.org/wiki/Design_Patterns); they can absorb some design patterns through osmosis.

An existing codebase also provides a gentle entry point for junior engineers. It's a hell of a lot easier to add an endpoint to an API than it is to create the API from scratch. Poor initial choices in code design produce continual headaches, and the existing codebase can act as a guardrail.

In the Data world, so much gets built from scratch. Often, each machine learning "task" or model gets its own custom training code. It might even get its own code repository. This makes it hard for new people to contribute since they have to start from nothing. Maintenance becomes a massive burden. Each model uses its own training framework, libraries, etc... that are likely only properly known by the original author. Just wait until the author leaves the company and somebody else has to take over.

## From Lone Wolf to a Pack

So how do we fix this? Shared libraries for common data operations are a nice place to start. Instead of copy/pasting the same code in between each Jupyter notebook for authenticating and querying the database, move that code into a shared library. Encouraging the data team to contribute to these libraries will facilitate code review and nudge people towards a team mindset. 

One can take things further. On my current team, we all work within a single repo that handles both training and serving of models. This _constrains_ model training and serving to match the codebase. The benefits of this are that everybody works (and reviews) the same training and serving code which helps with knowledge sharing and reduces the bus factor of a given model. 

On the flip side, this can be constraining! You can't willy nilly grab the code from the latest and greatest language model that you saw on the arXiv last night since it will need to be adapted to the codebase. I think that the pros outweigh the cons for our approach, but it's also true that it's hard to navigate this tradeoff, _especially_ in a domain like deep learning where paradigms shift so quickly.

My hope is that some of the difficulties in data will be resolved as the field matures. I was not in tech then, but I'm guessing that web development was a lot more difficult and bespoke before [Rails](https://rubyonrails.org/) came along. Perhaps there will one day be a Rails of ML which standardizes best practices and provides a starting template for each project.

Even if you do not have a shared codebase among data personnel, finding other ways for code to at least be seen by colleagues is important. Share your Jupyter notebooks (or export it to HTML but [allow the code to be optionally viewed](https://chris-said.io/2016/02/13/how-to-make-polished-jupyter-presentations-with-optional-code-visibility/)); setup regular pair programming sessions; require code review for anything that makes it to production.

The analytics world now has Analytics Engineering, and those people get to review each others' [dbt](https://www.getdbt.com/) code. Here's hoping we soon get Data Science Engineering.
