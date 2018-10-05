---
date: "2016-12-05"
slug: "recasketch-keras"
hasMath: true
notebook: true
title: "Using Keras' Pretrained Neural Networks for Visual Similarity Recommendations"
---
{{% jupyter_cell_start markdown %}}

<!-- PELICAN_BEGIN_SUMMARY -->

To close out our series on building recommendation models using [Sketchfab data]({{< ref "blog/likes-out-guerilla-dataset" >}}), I will venture far from the [previous]({{< ref "blog/implicit-mf-part-1" >}}) [posts']({{ ref "blog/implicit-mf-part-2" >}}) factorization-based methods and instead explore an unsupervised, deep learning-based model. You'll find that the implementation is fairly simple with remarkably promising results which is almost a smack in the face to all of that effort put in earlier.

<!-- PELICAN_END_SUMMARY -->

We are going to build a model-to-model recommender using thumbnail images of 3D [Sketchfab](https://sketchfab.com) models as our input and the *visual similarity* between models as our recommendation score. I was inspired to do this after reading Christopher Bonnett's [post](http://cbonnett.github.io/Insight.html) on product classification, so we will follow a similar approach.

Since our goal is to measure visual similarity, we will need to generate features from our images and then calculate some similarity measure between different images using said features. Back in the day, maybe one would employ fancy wavelets or SIFT keypoints or something for creating features, but this is the Era of Deep Learning and manual feature extraction is for old people. 

Staying on-trend, we will use a pretrained neural network (NN) to extract features. The NN was originally trained to classify images among 1000 labels (e.g. "dog", "train", etc...). We'll chop off the last 3 fully-connected layers of the network which do the final mapping between deep features and class labels and use the fourth-to-last layer as a long feature vector describing our images.

Thankfully, all of this is extremely simple to do with the pretrained models in [Keras](https://keras.io/). Keras allows one to easily build deep learning models on top of either Tensorflow or Theano. Keras also now comes with pretrained models that can be loaded and used. For more information about the available models, visit the [Applications](https://keras.io/applications/) section of the documentation. For our purposes, we'll use the [VGG16](https://keras.io/applications/#vgg16) model because that's what other people seemed to use and I don't know enough to have a compelling reason to stray from the norm.

Our task is now as follows:

1. Load and process images
2. Feed images through NN.
3. Calculate image similarities.
4. Recommend models!

## Load and process images

The first step, which we won't go through here, was to download all of the image thumbnails. There seems to be a standard thumbnail for each Sketchfab model accessible via their API, so I added a function to the [rec-a-sketch](https://github.com/EthanRosenthal/rec-a-sketch) [crawl.py](https://github.com/EthanRosenthal/rec-a-sketch/blob/master/crawl.py) script to automate downloading of all the thumbnails.

Let's load in our libraries and take a look at one of these images.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import csv
import sys
import requests
import skimage.io
import os
import glob
import pickle
import time

from IPython.display import display, Image, HTML
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skimage.io

sys.path.append('../')
import helpers
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
rand_img = np.random.choice(glob.glob('../data/model_thumbs/*_thumb200.jpg'))
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
Image(filename=rand_img) 
```

{{% jupyter_input_end %}}



{{< figure src="./index_3_0.jpeg" >}}



{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
img = skimage.io.imread(rand_img)
img.shape
```

{{% jupyter_input_end %}}




    (200, 200, 3)



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We see that the image can be represented as a 3D matrix through two spatial dimensions (200 x 200) and then a third RGB dimension. We have to do a couple of preprocessing steps before feeding an image through the VGG16 model. The images must be resized to 224 x 224, the color channels must be normalized, and an extra dimension must be added due to Keras expecting to recieve multiple models. Thankfully, Keras has built-in functions to handle most of this.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
img = kimage.load_img(rand_img, target_size=(224, 224))
x = kimage.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(x.shape)
```

{{% jupyter_input_end %}}

    (1, 224, 224, 3)


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We can now load our model in and try feeding the image through.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# image_top=False removes final connected layers
model = VGG16(include_top=False, weights='imagenet')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
pred = model.predict(x)
print(pred.shape)
print(pred.ravel().shape)
```

{{% jupyter_input_end %}}

    (1, 7, 7, 512)
    (25088,)


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We will later have to flatten the output of the model into a long feature vector. One thing that should be noted is the time that it takes to run a single model though the NN on my 4-core machine:

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
%%timeit -n5
pred = model.predict(x)
```

{{% jupyter_input_end %}}

    5 loops, best of 3: 905 ms per loop


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

This is pretty huge when you consider the fact that we will be processing 25,000 images! We'll now go through the above preprocessing steps for every model that we trained in the previous recommender blog posts. We can find these models by importing our "likes" data, filtering out low-interaction models and users (as before), and pick out the models that are leftover.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
df = pd.read_csv('../data/model_likes_anon.psv',
                 sep='|', quoting=csv.QUOTE_MINIMAL,
                 quotechar='\\')
df.drop_duplicates(inplace=True)
df = helpers.threshold_interactions_df(df, 'uid', 'mid', 5, 5)

# model_ids to keep
valid_mids = set(df.mid.unique())
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Feed images through NN

With our set of valid model IDs in hand, we can now run through the long process of loading in all of the image files, preprocessing them, and running them through the ```VGG``` prediction. This takes a long time, and certain steps blowup memory. I've decided to batch things up below and include some print statements so that one can track progress. Beware: this takes a long time!

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Grab relevant filenames
get_mid = lambda x: x.split(os.path.sep)[-1].split('_')[0]
fnames = glob.glob('../data/model_thumbs/*_thumb200.jpg')
fnames = [f for f in fnames if get_mid(f) in valid_mids]

idx_to_mid = {}
batch_size = 500
min_idx = 0
max_idx = min_idx + batch_size
total_max = len(fnames)
n_dims = preds.ravel().shape[0]
px = 224

# Initialize predictions matrix
preds = sp.lil_matrix((len(fnames), n_dims))

while min_idx < total_max - 1:
    t0 = time.time()
    
    X = np.zeros(((max_idx - min_idx), px, px, 3))
    
    # For each file in batch, 
    # load as row into X
    for i in range(min_idx, max_idx):
        fname = fnames[i]
        mid = get_mid(fname)
        idx_to_mid[i] = mid
        img = image.load_img(fname, target_size=(px, px))
        img_array = image.img_to_array(img)
        X[i - min_idx, :, :, :] = img_array
        if i % 200 == 0 and i != 0:
            t1 = time.time()
            print('{}: {}'.format(i, (t1 - t0) / i))
            t0 = time.time()
    max_idx = i
    t1 = time.time()
    print('{}: {}'.format(i, (t1 - t0) / i))
    
    print('Preprocess input')
    t0 = time.time()
    X = preprocess_input(X)
    t1 = time.time()
    print('{}'.format(t1 - t0))
    
    print('Predicting')
    t0 = time.time()
    these_preds = model.predict(X)
    shp = ((max_idx - min_idx) + 1, n_dims)
    
    # Place predictions inside full preds matrix.
    preds[min_idx:max_idx + 1, :] = these_preds.reshape(shp)
    t1 = time.time()
    print('{}'.format(t1 - t0))
    
    min_idx = max_idx
    max_idx = np.min((max_idx + batch_size, total_max))
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Calculate image similarities

I would recommend writing the predictions to disk here (don't want the kernel to die and lose all that work!). The ```preds``` matrix consists of a single row for each image with 25,088 sparse features as columns. To calculate item-item recommendations, we must convert this feature matrix into a similarity matrix.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
preds = preds.tocsr()
sim = cosine_similarity(preds)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Recommend models!

Using the similarity matrix, we can reuse some old functions from previous posts to visualize some the recommendations. I've added on some HTML so that clicking on the images links out to their Sketchfab pages. Let's look at a couple!

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
def get_thumbnails(sim, idx, idx_to_mid, N=10):
    row = sim[idx, :]
    thumbs = []
    mids = []
    for x in np.argsort(-row)[:N]:
        response = requests.get('https://sketchfab.com/i/models/{}'\
                                .format(idx_to_mid[x])).json()
        thumb = [x['url'] for x in response['thumbnails']['images']
                 if x['width'] == 200 and x['height']==200]
        if not thumb:
            print('no thumbnail')
        else:
            thumb = thumb[0]
        thumbs.append(thumb)
        mids.append(idx_to_mid[x])
    return thumbs, mids

def display_thumbs(thumbs, mids, N=5):
    thumb_html = "<a href='{}' target='_blank'>\
                  <img style='width: 160px; margin: 0px; \
                  border: 1px solid black; display:inline-block' \
                  src='{}' /></a>"
    images = "<div class='line' style='max-width: 640px; display: block;'>"

    display(HTML('<font size=5>'+'Input Model'+'</font>'))
    link = 'http://sketchfab.com/models/{}'.format(mids[0])
    url = thumbs[0]
    display(HTML(thumb_html.format(link, url)))
    display(HTML('<font size=5>'+'Similar Models'+'</font>'))

    for (url, mid) in zip(thumbs[1:N+1], mids[1:N+1]):
        link = 'http://sketchfab.com/models/{}'.format(mid)
        images += thumb_html.format(link, url)

    images += '</div>'
    display(HTML(images))
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
display_thumbs(*get_thumbnails(sim, 100, idx_to_mid, N=10), N=9)
```

{{% jupyter_input_end %}}


<font size=5>Input Model</font>



<a href='http://sketchfab.com/models/b59f0fe68c564f3aba820039e9833854' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/b59f0fe68c564f3aba820039e9833854.jpg' /></a>



<font size=5>Similar Models</font>



<div class='line' style='max-width: 640px; display: block;'><a href='http://sketchfab.com/models/286af23feb8243aba81f1f39368ca61f' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/286af23feb8243aba81f1f39368ca61f.jpg' /></a><a href='http://sketchfab.com/models/55e32670071c4e349ecb15b98da7b885' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='https://dg5bepmjyhz9h.cloudfront.net/urls/55e32670071c4e349ecb15b98da7b885/dist/thumbnails/673ddcb403544e8d8ced58c1b51f6c2e/200x200.jpeg' /></a><a href='http://sketchfab.com/models/8b7e4a8a15974a8984d82e06ff062a31' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/8b7e4a8a15974a8984d82e06ff062a31.jpg' /></a><a href='http://sketchfab.com/models/fe108c417de44663a5973f3fc4601b9a' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/fe108c417de44663a5973f3fc4601b9a.jpg' /></a><a href='http://sketchfab.com/models/9abd3c9c846d40d28af4a10d717fd417' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/9abd3c9c846d40d28af4a10d717fd417.jpg' /></a><a href='http://sketchfab.com/models/a54f997b9cf84bb8be8c0651710caeef' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/a54f997b9cf84bb8be8c0651710caeef.jpg' /></a><a href='http://sketchfab.com/models/a6bdf1d11d714e07b9dd99dda02de965' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/a6bdf1d11d714e07b9dd99dda02de965.jpg' /></a><a href='http://sketchfab.com/models/87114216af9e428e85cb2bca375610ea' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/87114216af9e428e85cb2bca375610ea.jpg' /></a><a href='http://sketchfab.com/models/7206ef1c43d34fc1928ae51cd45a8501' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/7206ef1c43d34fc1928ae51cd45a8501.jpg' /></a></div>


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
display_thumbs(*get_thumbnails(sim, 1000, idx_to_mid, N=10), N=9)
```

{{% jupyter_input_end %}}


<font size=5>Input Model</font>



<a href='http://sketchfab.com/models/f8b09235c2a64bf29afde51e91ce5c8c' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/f8b09235c2a64bf29afde51e91ce5c8c.jpg' /></a>



<font size=5>Similar Models</font>



<div class='line' style='max-width: 640px; display: block;'><a href='http://sketchfab.com/models/8a43d807592947fe9ba2225fe9662b8f' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/8a43d807592947fe9ba2225fe9662b8f.jpg' /></a><a href='http://sketchfab.com/models/6291037118b246c6a2013eccf1b1b626' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/6291037118b246c6a2013eccf1b1b626.jpg' /></a><a href='http://sketchfab.com/models/f5c73f41698f4f168abbc8cf30aec2cc' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/f5c73f41698f4f168abbc8cf30aec2cc.jpg' /></a><a href='http://sketchfab.com/models/6b600e6aef014a0ab77ba9c9ee2887ca' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/6b600e6aef014a0ab77ba9c9ee2887ca.jpg' /></a><a href='http://sketchfab.com/models/44e8e67d6ca84ff7ac5c851e0a44fae4' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/44e8e67d6ca84ff7ac5c851e0a44fae4.jpg' /></a><a href='http://sketchfab.com/models/503f871b71b7436ea16c8b73a5974555' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/503f871b71b7436ea16c8b73a5974555.jpg' /></a><a href='http://sketchfab.com/models/kxnw6yc07Zsu38jyVFYCqsaMvGn' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/kxnw6yc07Zsu38jyVFYCqsaMvGn.jpg' /></a><a href='http://sketchfab.com/models/d977e2e4f37747b88f63de415036fa1e' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/d977e2e4f37747b88f63de415036fa1e.jpg' /></a><a href='http://sketchfab.com/models/22f05d289cf044a09d6a50c7cac28dc0' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/22f05d289cf044a09d6a50c7cac28dc0.jpg' /></a></div>


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
display_thumbs(*get_thumbnails(sim, 1492, idx_to_mid, N=10), N=9)
```

{{% jupyter_input_end %}}


<font size=5>Input Model</font>



<a href='http://sketchfab.com/models/b9b32ca63ff84a33ae97fce2201cfe7b' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/b9b32ca63ff84a33ae97fce2201cfe7b.jpg' /></a>



<font size=5>Similar Models</font>



<div class='line' style='max-width: 640px; display: block;'><a href='http://sketchfab.com/models/82a1a8077d324508a81b1829c24f1c47' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/82a1a8077d324508a81b1829c24f1c47.jpg' /></a><a href='http://sketchfab.com/models/f31a3eb88409404d90e5027fdf32e753' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/f31a3eb88409404d90e5027fdf32e753.jpg' /></a><a href='http://sketchfab.com/models/faee89a8c0d646f99ed5f32962a8a2c8' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/faee89a8c0d646f99ed5f32962a8a2c8.jpg' /></a><a href='http://sketchfab.com/models/7c143c46845647a3a09859cf65c8730e' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/7c143c46845647a3a09859cf65c8730e.jpg' /></a><a href='http://sketchfab.com/models/3a686e566b27428b9f41c4657378e203' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/3a686e566b27428b9f41c4657378e203.jpg' /></a><a href='http://sketchfab.com/models/742d139255dd4fac94b894ee9ceda3a1' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/742d139255dd4fac94b894ee9ceda3a1.jpg' /></a><a href='http://sketchfab.com/models/59401daa797e408e8538052766ce2ab1' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/59401daa797e408e8538052766ce2ab1.jpg' /></a><a href='http://sketchfab.com/models/d976a960ba8340668923daa5c2937727' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/d976a960ba8340668923daa5c2937727.jpg' /></a><a href='http://sketchfab.com/models/279cb19a8a22407c8c94588314487872' target='_blank'>                  <img style='width: 160px; margin: 0px;                   border: 1px solid black; display:inline-block'                   src='images/recasketch-keras/279cb19a8a22407c8c94588314487872.jpg' /></a></div>


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Conclusion

Wow! With this completely unsupervised method and zero hyperparameter tuning, we get strikingly well-matched images. This might feel somewhat frustrating - why did we spend all that time with those math-heavy, brain-stretching factorization algorithms when we could just feed everything through a deep learning model? Firstly, it may be difficult to perform user-to-item recommendations or the tag-recommendations from last post. Secondly, it seems that this visual similarity model and the implicit feedback models serve different purposes. 

The NN does exactly what we expect - it finds similar images. The implicit feedback model finds other models that similar users have liked. What tends to happen is that the likes-based recommendations find models that share similar themes or appeal to certain clusters of users. For example, we may see that various anime characters get grouped together, or renderings of medieval armor and weapons. If we were to feed one of the medieval weapons into the NN, then we would find other examples of *only* that exact weapon which likely span across many periods of time.

I did attempt to combine the LightFM model with this NN model by taking the NN output features and using them as side information in the LightFM model. There were typically ~2500 nonzero NN features for each model which totally blew up the training time of the LightFM model. It took 30 minutes to compute the precision at k. I shuttered at the idea of calculating learning curves and grid searches, so I gave up! Maybe someday I'll spin up a giant EC2 box and see what happens.

Next post, I wrap up this series by writing about how I built out a Flask app on AWS called Rec-a-Sketch to serve up interactive Sketchfab recommendations. Thanks for reading!

{{% jupyter_cell_end %}}