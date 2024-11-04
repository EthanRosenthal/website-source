---
date: 2024-11-05
draft: false
hasMath: false
notebook: true
slug: "yaml-ml"
tags: ['data-science', 'machine-learning-engineering']
title: "Why can't we separate YAML from ML?"
---
{{% jupyter_cell_start markdown %}}

## Why can't I just write code?

I'm coming up on 10 years, and half as many jobs, in data science and machine learning. No matter what, in every role, I find myself reinventing a programming language on top of YAML in order to train machine learning models. 

In the beginning, there's a script.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset.csv", parse_dates=["timestamp"])
y = df["label"]
ts = df["timestamp"]
X = df.drop(columns=["label", "timestamp"])
train_mask = ts < "2022-01-01"
test_mask = ts >= "2022-01-01"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

model = LogisticRegression(C=10, random_state=666)
model = model.fit(X_train, y_train)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

But then, you find yourself hand-designing switch points.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
from sklearn.ensemble import RandomForestClassifier

def get_model(model_type: str):
    if model_type == "lr":
        return LogisticRegression(C=10, random_state=666)
    elif model_type == "rf":
        return RandomForestClassifier(random_state=666)

def get_train_data(end_date: str):
    ...

def get_test_data(start_date: str):
    ...
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

You keep forgetting parameters to pass to each of these switch points, which are actually now nodes in a [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph). So, you get the big brained idea to put all of your parameters into a single configuration file. You pick YAML because it's easier to use than JSON, and you still don't really know what TOML is.

You want maximum flexibility, so you push the configuration language to the brink with custom constructors that [dynamically instantiate Python classes](https://matthewpburruss.com/post/yaml/) and [import other yaml files](https://pypi.org/project/pyyaml-include/).

You sit back and marvel at the beauty of declarative configuration.

```yaml
model:
  !Load 
  cls: sklearn.ensemble.RandomForestClassifier
  params:
    n_estimators: 3000
    random_state: 666

data: 
  train: !Include ../data/train_config.yaml
  test: !Include ../data/test_config.yaml
```

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

You hand off your beautiful configuration framework to the rest of the team, and nobody has any idea what the fuck is going on.

## [What happened](https://www.youtube.com/watch?v=0M8zNwh5AgA)?

Whoops! It turns out you created a full on programming language without any modern benefits. Global YAML nodes are used across multiple files without any way to track references. Type validation is minimal. The IDE and mypy are unaware of all these python references lurking in YAML. And we end up writing un-pythonic code that's been shoehorned to work with YAML.

A couple years ago, I started midwitting the whole thing.

{{< figure src="/images/yaml-ml/midwit_code.jpg" >}}

It was easy to smugly preach that people should just write code. Just use [pydantic](https://docs.pydantic.dev/latest/) instead of YAML! I even did this in a previous role. 

Like everything in programming, easier said than done. Somehow, even though everything was type-validated, pydantic-based Python, we still spent way too much time fitting the code into the configuration framework that we'd built. 

Why can't I just write code? 

## What is unique to ML?

I finally realized why it's hard to just write code. There are 3 primary requirements for an ML config system:

### 1. The ability to call different code throughout the DAG.

ML is research. It's experimentation. It requires iteration and flexibility.

I want to be able to try out different classifiers, different loss functions, different preprocessing methods, etc... The easiest way to support this is by allowing code to be written and called. 

We don't want to hand-change parameters deep within a nested DAG in order to experiment with things. We'd rather define all of our parameters at the start of the DAG, and then let the DAG code handle the rest.

### 2. Lazy-instantiation of classes in the DAG.

If I am fine-tuning a pretrained model, I don't want to have to download the entire pretrained model when I define my configuration. 

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Some quick imports

import logging
import warnings

import transformers
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForSequenceClassification, PreTrainedModel

transformers.logging.set_verbosity(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm.*")
```

{{% jupyter_input_end %}}

    /Users/erosenthal/personal/website-source/notebooks/yaml-ml/.venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

In the code below, all of BERT gets downloaded when I define my configuration object. You can imagine worse scenarios where instantiating a model requires allocating huge amounts of GPU memory. We should not do this when our config is defined. This is especially bad if we're running a distributed training job and need to wait until we're on a given node before instantiating the model.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Bad
class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: PreTrainedModel
    

config = TrainingConfig(
    # The model weights get downloaded when defining the config.
    model=AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased"
    )
)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

So, model instantiation should be deferred as long as possible.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Better
class TrainingConfig(BaseModel):
    model_id: str 

config = TrainingConfig(model_id="bert-base-uncased")

...

# Sometime Later
model = AutoModelForSequenceClassification.from_pretrained(
    config.model_id
)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

### 3. Tracking of all configuration (hyper)parameters

We run so many experiments when training ML models that we have to pay for experiment trackers like [Weights & Biases](https://wandb.ai/site/). Crucial to experiment tracking is tracking what changed in between different training runs. The easiest way to do this in a tool like W&B is to put all configuration into a dictionary and log the dict for a given run. 

It turns out this tracking requirement is at direct odds with "just writing code". 

It's easy to just write the code to instantiate a model in PyTorch:


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import torch 

class MyModel(torch.nn.Module):
    def __init__(
        self, 
        num_features: int, 
        num_classes: int,
        hidden_size: int = 128
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
    
model = MyModel(1_000, 16)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

But, now I can't track the model's constructor arguments. 

A side benefit of ensuring our configuration is trackable is that we get some semblance of reproducibility, since our configuration must be serializable into something like JSON or YAML. One reason YAML is so popular is that we don't have to serialize anything!

## Why can't we separate YAML from ML?

YAML solves for all of our requirements. 

1. We can write custom constructors to call arbitrary code throughout our DAG.
2. No python code is instantiated when we instantiate our config
3. YAML is easily trackable and reproducible.

The problem, of course, is everything that I mentioned at the beginning.


## What are the solutions?

If you're okay with refactoring your entire codebase, then you should make all your classes take a single configuration object as their arguments. This will solve for all three problems while giving you nice, structured, non-YAML configs.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
from typing import Type

import torch

class MyModelConfig(BaseModel):
    num_features: int
    num_classes: int
    hidden_size: int = 128

class MyModel(torch.nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.num_features, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_cls: Type[MyModel]
    model_cfg: MyModelConfig


# No models are instantiated in the config.
config = TrainingConfig(
    model_cls=MyModel,
    model_cfg=MyModelConfig(num_features=100, num_classes=10)
)
# Lazy-load the model.
model = config.model_cls(config.model_cfg)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

You can even try to be a little fancy and define the class configs inside the class, so that they're always coupled to each other. This makes it easy to dynamically instantate the class, based on the class config alone.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import importlib

def load_config(config: BaseModel):
    module = config.__class__.__module__
    parent = config.__class__.__qualname__.split(".")[0]
    return getattr(importlib.import_module(module), parent)(config)


class MyModel(torch.nn.Module):
    class Config(BaseModel):
        num_features: int
        num_classes: int
        hidden_size: int = 128
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.num_features, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_cfg: BaseModel


config = TrainingConfig(
    model_cfg=MyModel.Config(num_features=100, num_classes=10)
)
model = load_config(config.model_cfg)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

The problem with this approach, of course, is that it requires refactoring your entire codebase. This also become particularly obnoxious since it requires creating a config for _everything_, including classes that you would rather not lazily instantiate. 

What I find that people often do is they avoid refactoring their codebase by creating separate configs that match their class constructor arguments.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
class MyModelConfig(BaseModel):
    num_features: int
    num_classes: int
    hidden_size: int = 128

class MyModel(torch.nn.Module):
    def __init__(
        self, 
        num_features: int, 
        num_classes: int,
        hidden_size: int = 128
    ):
        ...
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

The code duplication here is painful.

A better solution does 3 things:

1. No class constructor refactoring. Call classes as they would be called.
1. Track all class constructor arguments in a serializable, centralized place.
1. Somehow _also_ allow for lazy-instantation, even though the class constructors are being called.

Step 3 could be skipped if you refactor to ensure that all model's constructors are "lightweight" and don't involve anything like downloading giant weights, allocating lots of GPU memory, etc... but that places major constraints on your codebase.

Does a solution for all 3 steps exist? I haven't seen it yet, but I think I have some ideas.

{{% jupyter_cell_end %}}