---
date: "2017-02-05"
slug: "rec-a-sketch"
notebook: true
title: "Rec-a-Sketch: a Flask App for Interactive Sketchfab Recommendations"
tags:
  - web development
  - recommendation systems
---
{{% jupyter_cell_start markdown %}}

<!-- PELICAN_BEGIN_SUMMARY -->

After the long [series]({{< ref "/blog/likes-out-guerilla-dataset" >}}) [of]({{< ref "/blog/implicit-mf-part-1" >}}) [previous]({{< ref "/blog/implicit-mf-part-2" >}}) [posts]({{< ref "/blog/recasketch-keras" >}}) describing various recommendation algorithms using Sketchfab data, I decided to build a website called [Rec-a-Sketch](http://www.rec-a-sketch.science/) which visualizes the different algorithms' recommendations. In this post, I'll describe the process of getting this website up and running on AWS with nginx and gunicorn.

<!-- PELICAN_END_SUMMARY -->

## Goal

The goal of the website was two-fold.

1. I wanted to view the different algorithm's recommendations side-by-side for comparison.
2. I wanted to get "lost" in the recommendations like one gets lost clicking from link to link on Wikipedia.

I organized the page as follows so that (1) all recommendations were visible and (2) one can click on any of the recommended models to be taken to that model's recommendations.

[![main page](images/rec_screenshot.png)](images/rec_screenshot.png)


## Organization of the App

I decided to use Flask to build the web app because I already have some experience with it, and I'm not trying to reinvent the wheel! The functionality itself is fairly simple. Other than an about page, there is only one page and one Flask ```route``` in the whole site.

The functionality is relatively simple. When one initially goes to the page, there is a default list of models to select from, or one can input a link to a custom model. Once a model is selected, this sends a ```GET``` request to the main ```route```. When the ```route``` receives this request, it must do two things:

1. Grab data about the input model (name, url, and thumbnail).
2. Find other recommended models and get their associated data. 

I populated a sqllite database with data about the Sketchfab models. I do not store the thumbnails directly; rather, I include a link to the thumbnail on Sketchfab's servers.

For grabbing recommendations, I created a table with *precomputed* recommendations for each model. The recommendations are stored in the stupidest possible way as a string of comma-separated model ID's. I pull down the string, split on the commas, and place everything in a list. The code looks something like what follows.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# mid is an inputted model ID
# type is the recommendation algorithm type (e.g. learning-to-rank)
c = conn.cursor()
sql = """
    SELECT
      type,
      recommended
    FROM recommendations
    WHERE
      mid = '{}'
""".format(mid)
c.execute(sql)
results = c.fetchall()
recommendations = []
for r in results:
    recommendations.append((r[0], [str(x) for x in r[1].split(',')]))
recommendations = dict(out)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

I should note that the above code is easily subject to [SQL Injection](https://en.wikipedia.org/wiki/SQL_injection). Please don't write code like this on a production server!

The main ```route``` functionality was actually the easiest part of the whole project. The hardest parts were getting things running remotely.

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Deploying to AWS

This [post](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-14-04) from DigitalOcean was super helpful in getting things up and running. In fact, I almost followed that post verbatim.

### EC2

For my purposes, I chose to use Amazon Web Services (AWS) instead of DigitalOcean for hosting the Rec-a-Sketch. This was imply because I had previous experience with AWS. The first step is to setup an EC2 instance which is virtual server. Rec-a-Sketch is lightweight, so I chose a t2.nano instance because it's the cheapest. 

One must create an Elastic IP address for the instance (which costs some money) as well as open ports 80 and 22. The ports can be opened by going to ```Network & Security -> Security Groups``` and creating a security group with the following ports:

[![main page](images/security_groups.png)](images/security_groups.png)

When the EC2 instance is created, you can download a pem file which allows you to ssh into the EC2 box. Save the pem file to your computer, and set the permissions accordingly:

```
chmod 400 pemfile.pem
```

I usually place the file in ```~/.ssh/``` and then add the file to my ```~/.ssh/config``` file for easy ssh-ing later on. The ```config``` file lets you setup quick aliases for ssh-ing (see [here](http://nerderati.com/2011/03/17/simplify-your-life-with-an-ssh-config-file/) for more details).

### The Stack

Once you're able to ssh into the EC2 instance, it's time to setup the stack. The stack consists of the following:


1. nginx | a web server which can handle incoming requests and redirect them to your Flask app.
2. upstart | this makes sure that your Flask app stays up and running. If the app should die, upstart will start it back up again.
3. gunicorn | a python WSGI HTTP server. I freely admit that I don't quite get the purpose of gunicorn. One clear benefit is that you can run multiple "workers" or copies of your Flask app which allows you to process multiple requests at once.

The DigitalOcean posts walks through the setup of this stack quite nicely. Some modifications that I made are that I use [miniconda](https://conda.io/miniconda.html) for managing the python libraries. In my upstart script, I have to make sure to add miniconda to the ```PATH``` environment variable. The upstart script is on github [here](https://github.com/EthanRosenthal/rec-a-sketch/blob/master/flask_app/recasketch.conf), and the nginx configuration is [here](https://github.com/EthanRosenthal/rec-a-sketch/blob/master/flask_app/nginx.conf).

I did run into some issues setting up both the upstart service and nginx (when do things ever work the first time around?). Both services have log files which can be helpful for debugging. nginx had access and error logs in ```/var/log/nginx/```, and each upstart service has its own log in ```/var/log/upstart/```.

## Image Hosting

I mentioned before that I do not actually host the Sketchfab model images on my server. I would have to pay for outgoing bandwidth, and this would add up quite fast (if people actually visit my website!). A simpler way to host images (though maybe morally dubious?) is to point to the url where Sketchfab hosts the image. 

The Sketchfab API easily lets you find the location of an image thumbnail. At first I would just ping the Sketchfab API for each request that came into my Flask app. This proved super slow because I would have to wait for the Sketchfab API response each time. I tried to solve this by running a big script to store all API responses in my own database. 

This worked for a bit, but then the image links started to break. I was confused for a bit, but maybe you can figure out what happened - here's an example image link:

```
https://dg5bepmjyhz9h.cloudfront.net/urls/a1194aa7be824b7da6accb1d0c788132
/dist/thumbnails/93e331260a8142c6ab85d61f6a025476/200x200.jpeg
```

What's going on here? It turns out that Sketchfab smartly hosts their images using a Content Delivery Network, or CDN. CDNs are used to quickly serve files to users by hosting the files much closer to the user. There's no guarantee that the filename should stay the same at the CDN node, and it seems that they do not.

I did not want to go back to pinging the Sketchfab API on every request, so I settled on a compromise. I setup a cron job to run every two days to grab the current image urls. The assumption here is that these urls will not change too quickly, and I am fine with small breakages in the meantime. The cron job script is located [here](https://github.com/EthanRosenthal/rec-a-sketch/blob/master/flask_app/app/update.sh). 

## Closing Thoughts

I had a lot of fun playing with the Sketchfab data and building Rec-a-Sketch. There a lot more algorithms that I would like to try out on the data, but I may like to venture into a different project for the time being. I would encourage you to try playing with the data and see what pops out. In the meantime, try getting "lost" in [Rec-a-Sketch](http://www.rec-a-sketch.science/)!

{{% jupyter_cell_end %}}