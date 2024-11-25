---
date: "2017-06-20"
slug: "matrix-factorization-in-pytorch"
notebook: true
title: "Matrix Factorization in PyTorch"
hasMath: true
tags:
  - deep learning
  - recommendation systems
---
{{% jupyter_cell_start markdown %}}

<!-- PELICAN_BEGIN_SUMMARY -->

_Update 7/8/2019: Upgraded to PyTorch version 1.0. Removed now-deprecated `Variable` framework_

_Update 8/4/2020: Added missing `optimizer.zero_grad()` call. Reformatted code with [black](https://github.com/psf/black)_

Hey, remember when I wrote those [ungodly]({filename}/2016-01-09-explicit-matrix-factorization-als-sgd.md) [long]({filename}/2016-10-19-implicit-mf-part-1.md) [posts]({filename}/2016-11-7-implicit-mf-part-2.md) about matrix factorization chock-full of gory math? Good news! You can forget it all. We have now entered the Era of Deep Learning, and automatic differentiation shall be our guiding light. 

<!-- PELICAN_END_SUMMARY -->

Less facetiously, *I* have finally spent some time checking out these new-fangled deep learning frameworks, and damn if I am not excited.

In this post, I will show you how to use [PyTorch](http://pytorch.org/) to bypass the mess of code from my old post on Explicit Matrix Factorization and instead implement a model that will converge faster in fewer lines of code. 

But first, let's take a trip down memory lane.

## The Dark Ages

If you recall from the original matrix factorization post, the key to the derivation was calculus. We defined a loss function which was the mean squared error (MSE) loss between the matrix factorization "prediction" and the actual user-item ratings. In order to minimize the loss function, we of course took the derivative and set it equal to zero. 

For the Stochastic Gradient Descent (SGD) derivation, we iterated through each sample in our dataset and took the derivative of the loss function with respect to each free "variable" in our model, which were the user and item latent feature vectors. We then used that derivative to smartly update the values for the latent feature vectors as we surfed down the loss function in search of a minima.

I just described what we did in two paragraphs, but it took much longer to actually derive the correct gradient updates and code the whole thing up. The worst part was that, if we wanted to change anything about our prediction model, such as adding regularization or a user bias term, then we had re-derive all of the gradient updates by hand. And if we wanted to play with new techniques, like dropout? Oof, that sounds like a pain to derive!

## The Enlightenment

The beauty of modern deep learning frameworks is that they utilize automatic differentiation. The phrase means exactly what it sounds like. You simply define your model for prediction, your loss function, and your optimization technique (Vanilla SGD, Adagrad, ADAM, etc...), and the computer will automatically calculate the gradient updates and optimize your model.

Additionally, your models can often be easily parallelized across multiple CPU's or turbo-boosted on GPU's with little effort compared to, say, writing Cython code and parallelizing by hand.

## Minimum Viable Matrix Factorization

We'll walk through the three steps to building a prototype: defining the model, defining the loss, and picking an optimization technique. The latter two steps are largely built into PyTorch, so we'll start with the hardest first.

### Model

All models in PyTorch subclass from [torch.nn.Module](http://pytorch.org/docs/nn.html#torch.nn.Module), and we will be no different. For our purposes, we only need to define our class and a `forward` method. The `forward` method will simply be our matrix factorization prediction which is the dot product between a user and item latent feature vector.

In the language of neural networks, our user and item latent feature vectors are called _embedding_ layers which are analogous to the typical two-dimensional matrices that make up the latent feature vectors. We'll define the embeddings when we initialize the class, and the `forward` method (the prediction) will involve picking out the correct rows of each of the embedding layers and then taking the dot product. 

Thankfully, many of the methods that you have come to know and love in numpy are also present in the PyTorch tensor library. When in doubt, I treat things like numpy and usually get 90% there.

We'll also make up some fake ratings data for playing with in this post.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import numpy as np
from scipy.sparse import rand as sprand
import torch

# Make up some random explicit feedback ratings
# and convert to a numpy array
n_users = 1_000
n_items = 1_000
ratings = sprand(n_users, n_items, density=0.01, format="csr")
ratings.data = np.random.randint(1, 5, size=ratings.nnz).astype(np.float64)
ratings = ratings.toarray()
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
model = MatrixFactorization(n_users, n_items, n_factors=20)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

### Loss

While PyTorch allows you to define custom loss functions, they thankfully have a default mean squared error loss function in the library.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
loss_func = torch.nn.MSELoss()
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

### Optimization

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

There are a bunch of different optimization methods in PyTorch, but we'll stick with straight-up Stochastic Gradient Descent for today. One of the fun things about the library is that you are free to choose how to optimize each parameter of your model. Want to have different learning rates for different layers of your neural net? Go for it.

We'll keep things simple, though, and use the same technique for all parameters in our model. This requires instantiating an optimization class and passing in the parameters which we want this object to optimize.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # learning rate
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

### Train

With the three steps complete, all that's left is to train the model! We have to convert our data PyTorch tensors which will talk with the automatic differentiation library, [autograd](http://pytorch.org/docs/autograd.html). The PyTorch tensors must be Python-native datatypes like `float` and `long` rather than standard `numpy` datatypes, and it can be a little difficult to cast everything correctly.

Nonetheless, below we shuffle our data and iterate through each sample. Each sample is converted to a tensor, the loss is calculated, backpropagation is carried out to generate the gradients, and then the optimizer updates the parameters. And voilá! We have matrix factorization.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Sort our data
rows, cols = ratings.nonzero()
p = np.random.permutation(len(rows))
rows, cols = rows[p], cols[p]

for row, col in zip(*(rows, cols)):
    # Set gradients to zero
    optimizer.zero_grad()
    
    # Turn data into tensors
    rating = torch.FloatTensor([ratings[row, col]])
    row = torch.LongTensor([row])
    col = torch.LongTensor([col])

    # Predict and calculate loss
    prediction = model(row, col)
    loss = loss_func(prediction, rating)

    # Backpropagate
    loss.backward()

    # Update the parameters
    optimizer.step()
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Experimentation

The beauty of the above code is that we can rapidly experiment with different parts of our model and training. For example, adding user and item biases is quite simple via an embedding layer with one dimension:

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
class BiasedMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors, sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors, sparse=True)
        self.user_biases = torch.nn.Embedding(n_users, 1, sparse=True)
        self.item_biases = torch.nn.Embedding(n_items, 1, sparse=True)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

$L_{2}$ regularization can easily be added to the entire model via the optimizer:

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
reg_loss_func = torch.optim.SGD(model.parameters(), lr=1e-6, weight_decay=1e-5)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

And if you want to get smart about decaying your learning rate, try out a different optimization algorithm:

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
adagrad_loss = torch.optim.Adagrad(model.parameters(), lr=1e-6)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

But that's all just the tip of the iceberg. The real fun is when we stop thinking of recommendation systems as simply user-by-item matrices and incorporate all sorts of side information, context, temporal data, etc... and pass this all through novel network topologies without having to calculate everything by hand. Let me know if you do any of that :)

## Caveat

To now throw a little water on this optimistic post, it's worth pointing out that there are some costs to abandoning bespoke recommendation systems in favor of repurposing deep learning frameworks. For one, _well-written_ custom solutions are still much faster. Alternating Least Squares is an elegant and more efficient method than SGD for factorizing matrices, and packages like [implicit](https://github.com/benfred/implicit) show this off with their blazingly fast speed. 

Support for sparse feature learning is still a bit spotty, too. For example, [LightFM](https://github.com/lyst/lightfm) is built from the ground up with sparse features in mind and consequently operates much quicker than a similar solution in a deep learning framework. 

This post may (read: will) eventually date itself. As GPUs get larger and the frameworks support more sparse operations on the GPU, it may make sense to fully switch over to deep learning frameworks from bespoke CPU solutions.

## Further Exploration

This post is just a brief introduction to implementing a recommendation system in PyTorch. There are many parts of the code that are poorly optimized. In particular, it is quite helpful to have a generator function/class for loading the data when training. Additionally, one can push computation to a GPU or train in parallel in a [HOGWILD](http://pytorch.org/docs/notes/multiprocessing.html) manner.

If you would like to see more, I have started to play around with some explicit and implicit feedback recommendation models Pytorch in a repo [here](https://github.com/ethanrosenthal/torchmf). Maciej Kula, the author of LightFM, also has a repo with recommendation models in Pytorch [here](https://github.com/maciejkula/netrex) (_Update 11/12/2017: Maciej turned that repo into [Spotlight](https://github.com/maciejkula/spotlight) which supports a wide range of recommendation models in PyTorch_).

Lastly, why PyTorch for this post? I don't have a great reason other than that I have _really_ enjoyed its general ease of use: it's super easy to install (even with GPU support), it integrates nicely with numpy, and the documentation is fairly extensive. It also feels like there is less noise out there when I am googling for help compared to more popular frameworks like Tensorflow. The downside is the library is a bit rough around the edges, and the API is rapidly changing.


{{% jupyter_cell_end %}}