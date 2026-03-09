---
date: 2026-03-10
draft: false
enableEmoji: true
hasMath: false
slug: "confingy"
tags: ['data-science', 'machine-learning-engineering']
title: "Introducing confingy"   
---

I'm incredibly excited to finally introduce `confingy` to the world.


{{< figure src="./confingy.svg" >}}

- GitHub: https://github.com/runwayml/confingy
- Docs: https://runwayml.github.io/confingy


This is a library that solves a problem that I've had for over a decade at every company at which I've worked. The library is also the answer to my previous blog post, [Why can't we separate YAML from ML?]({{< ref "/blog/yaml-ml" >}}).

Major shoutout to my employer, [Runway](https://runwayml.com), for letting me open source the library. Internally, all research configuration code was migrated off of YAML and onto `confingy` months ago.

## So what actually is `confingy`?

I'm calling `confingy` an _implicit configuration system_ for Python. By adding a simple `@track` decorator to any class:

```python
from confingy import track


@track
class Foo:
    def __init__(self, bar: str):
        ...
```

you get 3 features:

1. All arguments to the class' constructor get tracked for later.
2. Lazily-instantiate any tracked class.
3. Serialize tracked classes to JSON and deserialize back into Python.

## Who needs this?

`confingy` is designed for anybody who builds iterative or flexible workflows, such as machine learning model training, data pipelines, DAGs, prompt orchestration, and so on. These workflows often involve YAML configs, [OmegaConf](https://omegaconf.readthedocs.io/), dataclasses, or [Pydantic](https://docs.pydantic.dev/latest/) models that define different steps in the workflow. These configs become _interfaces_ to your code and inevitably require duplication of logic for each new step.

People often build out these complicated configuration systems because they need [~~reproducibility~~ reusability]({{< ref "/blog/2026-03-09-08-reproducibility.md" >}}). Their systems inevitably grow into a mess of dynamic code instantiation and other separations between configuration and code that actively harm the software architecture, such as by incentizing inheritance over composition.

With `confingy`, your code _is_ the config.

## That's it?

There is a lot more information in the repo and docs. With the 3 features above as core principles, many benefits fall out of this library. We get type validation for free. There's a transpiler to go from a serialized JSON config back into actual python code. There's a visualization server for diffing configs, and a CLI for quick serialization/deseriliazition/transpilation. Please take a look and provide feedback!

