---
date: "2016-07-20"
slug: "lets-talk-or"
title: "I'm all about ML, but let's talk about OR"
hasMath: true
notebook: true
---
{{% jupyter_cell_start markdown %}}

<!-- PELICAN_BEGIN_SUMMARY -->

You've studied machine learning, you're a dataframe master for massaging data, and you can easily pipe that data through a bunch of machine learning libraries. 

<!-- PELICAN_END_SUMMARY -->

You go for a job interview at a SAAS company, you're given some raw data and labels and asked to predict churn, and come on - are these guys even trying? You generate the shit out of some features, you overfit the hell out of that multidimensional manifold just so you can back off and show off your knowledge of regularization, and then you put the icing on the cake by cross validating towards a better metric for the business problem than simple accuracy. 

Next.

You roll up on an ecommerce company, and they trick you by basically giving you no features. Just users, products, and clicks? Ha! Nice try, but you know that's a classic recommender system. Boom! Out of nowhere, the ratings matrix is factorized, and you're serving up top-notch recommendations a-plenty.

Next.

Your uncle calls you and asks for a "family favor". He manufactures artisinal dog collars in a Brooklyn warehouse and sells directly to the consumer. He doesn't know what the word churn means. He needs to decide how many of his 5 types of dog collars he should build next month with his limited supplies. Each collar in the catlog has a different profit margin, a different minimum build quantity, and there's only a 10K budget. Uncle asks you, "How many of each collar should we build to maximize profit?"

Uhhhhh.

You panic and try to apply ML to everything. *Hmmm, profit is a not a label but more a continuous variable. Okay, not a classification problem. Maybe this is a regression problem? Do I even want to predict the profit? Wait, what are my features? Shit, there's like no data...*

And so you write a program to enumerate all possible combinations of dog collar quantities along with the associated profit, and you find the maximum of the list of answers. Your uncle's happy, but you feel like an idiot because you know that this isn't scalable and god help you if you're ever asked this in an interview and there's just *got* to be a better way, right?

Well, there's a better way.

There's a better way! And I feel like nobody talks about it because the Data's not Big, you're not Learning Deep things, and there's nary a chatbot in sight. It's boring, old operations research which was something that I guess your university offered but nobody really knew what it meant. Full disclosure: I still don't really know what it means. I *do* know that the job of the data scientist is to bring value to the company, and having some operations and optimization in your toolbelt is quite valuable!

## Problem formulation

Let's try to solve Uncle's problem to get a feel for how we think about these things. I am not going to in any way cover how programs actually solve these problems (because I still don't know) but instead how one sets up this problem and asks a solver to solve this in Python.

What was the original question? "How many of each collar should we build to maximize profit?" Maximize profit is the key phrase here. The way to think of optimization problems like this are in terms of the objective function and the constraints. We want to maximize profit, so our objective function will be the total profit from all collars produced (assuming all get sold). If we say that we have a variable, $c\_{j}$, which says how many of collar $j$ we will produce, and $p\_{j}$ is a constant denoting the profit that we will make on that collar, then our objective function is simply

$$\sum\limits\_{j}p\_{j}c\_{j}$$

We want to maximize this function subject to specific constraints. One constraint is that we only have a 10K budget. The simplest method of solving these types of problems is to keep all constraints linear. If we have a known constant $w\_{j}$ which is the cost in dollars of producing collar $j$, then the constraint could be expressed in mathematical form as

$$\sum\limits\_{j}w\_{j}c\_{j} \leq 10000$$

Isn't this fun? All of these optimization problems consist of figuring out how to define your constraints in terms of *math* which is basically like the ultimate nerd-puzzle. The constraint on figuring out your constraints (see what I did there?) is that you should keep your constraints linear. This can get fairly tricky (read: fun).

Alright, let's figure out the last two constraints, and then we can start coding this thing up. If $r\_{j}$ is the minimimum run size for each collar, then this is relatively simple:

$$\forall\, j,\, c\_{j} \geq r\_{j} $$

By the way, that means "for all $j$, $c\_{j}$ is greater than or equal to $r\_{j}$". We can lastly deal with the limited "supplies" for building collars by assuming that $m\_{i}$ is the max quantity of supply $i$. If each collar $j$ requires $s\_{ji}$ quantity of supply $i$, then the limited supplies constraint is written

$$\forall\, i,\, \sum\limits\_{j}s\_{ij} \leq m\_{i}$$

## Math $\Rightarrow$ Code

At Birchbox, I programmed and solved a number of optimization problems, and I had the luxury to use [Gurobi](http://www.gurobi.com/), a state of the art mixed-integer programming solver with API's in many languages. What does that mean? That means you can build your problem in the language of your choice, the API's will translate your code into whatever Gurobi likes as input (C?), and then the Gurobi solver will solve the problem. Gurobi is most definitely not free, so I won't use it in this blog post on the off chance that you want to program along. Instead, we'll use [GLPK](https://www.gnu.org/software/glpk/) as our solver and [pulp](https://pythonhosted.org/PuLP/) as the Python API. I've included instructions for installing both libraries on Linux at the bottom of this post.

We'll start by importing our libraries and placing some made up data into two dataframes to get an idea of how the data's organized. The columns with material names are assumed to denote the "quantity" that the collar requires. For the sake of just trying to learn something, let's ignore what "0.70 metal" might mean and move along.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
from pulp import *
import numpy as np
import pandas as pd
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
names = ['leather', 'metal', 'canvas', 'cost', 'profit', 'min_run_size']
collars = pd.DataFrame([(0.50, 0.25, 0.30, 26.00, 10.50, 30),
                        (0.30, 0.70, 1.20, 29.00, 12.00, 40),
                        (0.90, 0.60, 0.57, 22.00, 09.00, 25),
                        (1.10, 0.45, 0.98, 26.50, 11.50, 60),
                        (0.75, 0.95, 0.55, 20.00, 08.50, 50)],
                      columns=names)
collars.index.name = 'Dog collar type'
collars
```

{{% jupyter_input_end %}}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>leather</th>
      <th>metal</th>
      <th>canvas</th>
      <th>cost</th>
      <th>profit</th>
      <th>min_run_size</th>
    </tr>
    <tr>
      <th>Dog collar type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>0.25</td>
      <td>0.30</td>
      <td>26.0</td>
      <td>10.5</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.30</td>
      <td>0.70</td>
      <td>1.20</td>
      <td>29.0</td>
      <td>12.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.90</td>
      <td>0.60</td>
      <td>0.57</td>
      <td>22.0</td>
      <td>9.0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.10</td>
      <td>0.45</td>
      <td>0.98</td>
      <td>26.5</td>
      <td>11.5</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.75</td>
      <td>0.95</td>
      <td>0.55</td>
      <td>20.0</td>
      <td>8.5</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>



{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
quants = pd.DataFrame([400, 250, 300],
                      index=['leather', 'metal', 'canvas'],
                      columns=['max_quantity'])
quants
```

{{% jupyter_input_end %}}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>leather</th>
      <td>400</td>
    </tr>
    <tr>
      <th>metal</th>
      <td>250</td>
    </tr>
    <tr>
      <th>canvas</th>
      <td>300</td>
    </tr>
  </tbody>
</table>
</div>



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We now define variables from the dataframes that match the math from above.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
p = collars.profit
w = collars.cost
r = collars.min_run_size
m = quants.max_quantity
s = collars[['leather', 'metal', 'canvas']]
collar_index = range(collars.shape[0]) # j
material_index = range(s.shape[1]) # i
budget = 10000.0
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

The rest of the code follows exactly as we thought of the original math. We first think of the objective function. We know that we want to maximize profit, so we instantiate a maximization ```LpProblem```. We then build the *variables* in our problem. I always confuse the term "variable" with any of the letters/symbols in our math problem, but we really mean a quantity that is variable (changes value, not the constants). The only variable is $c\_{j}$, the number of collar $j$ to build. Using pulp, we can create $c\_{j}$ variables with some pre-provided domain knowledge: We know that $c\_{j}$ must be an integer. We also know that we cannot build more collars than our budget allows, so let's save our solver some time by passing an upper bound. Lastly, we add the objective function to the problem using the newly created variable. For taking sums in pulp, you should use the ```lpSum``` function.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
prob = LpProblem('Dog Collar Problem', LpMaximize)
# Collar counts (the variable in our problem)
c = []

for j in collar_index:
    max_count = np.floor(budget / w[j])
    c.append(LpVariable('c_{}'.format(j),
                        lowBound=0,
                        upBound=max_count,
                        cat='Integer'))
    
# For pulp, add objective function first
prob += lpSum([i * j for i, j in zip(p, c)]), 'Total profit'
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

The constraints can now be added to the problem. The inequalities are written just as they would in math. A good rule of thumb is to use an explicit for loop any time we have a $\forall$ in our math.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Budget constraint
prob += lpSum([w[j] * c[j] for j in collar_index]) <= budget, 'Budget'

# Min run size constraint
for j in collar_index:
    prob += c[j] >= r[j], 'MinBatchSize_{}'.format(j)

# Max supplies quantity
for i in material_index:
    prob += lpSum([s.iloc[j, i] * c[j] for j in collar_index]) <= m[i]
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Our variables are created, our objective is defined, and all constraints have been added. What's left? Solve!

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
prob.solve()
```

{{% jupyter_input_end %}}




    1



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We returned a 1 which means that the problem solved. The pulp documentation provides some [examples](https://pythonhosted.org/PuLP/CaseStudies/index.html) of how to inspect the solved model which I've repurposed below

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
print("Status: {}".format(LpStatus[prob.status]))

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print("{} = {} ".format(v.name, v.varValue))

# The optimised objective function value is printed to the screen
print('Total profit = ${:6.2f}'.format(value(prob.objective)))
print('Total cost = ${:6.2f}'.format(np.sum(x * v.varValue for x, v in zip(w, prob.variables()))))
```

{{% jupyter_input_end %}}

    Status: Optimal
    c_0 = 68.0 
    c_1 = 40.0 
    c_2 = 25.0 
    c_3 = 151.0 
    c_4 = 126.0 
    Total profit = $4226.50
    Total cost = $9999.50


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Awesome! You now know how to solve simple optimization problems in Python. There are many more details that could be covered (constraints on the max of an array, if-else statement constraints, etc...), but hopefully this whet your appetite to play a little on your own.

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## INSTALLATION NOTES

You will need to install GLPK and pulp The following instructions worked for me on Ubuntu 14.04 using Anaconda.

#### GLPK
```bash
sudo apt-get install glpk-utils
```

#### pulp
There is no default conda channel. I used the following channel which seemed to work fine. Note: this did not work with python 3.

```bash
conda install --channel https://conda.anaconda.org/timcera pulp
```
You can test the installation by opening python and running the following test

```python
import pulp
pulp.pulpTestAll()
```
You should expect to fail some of the tests if you don't have CPLEX, COIN, Gurobi, etc... However, the GLPK tests should work. I saw the following printout (I only have GLPK installed on my home computer)
```
	 Testing zero subtraction
	 Testing inconsistant lp solution
	 Testing continuous LP solution
	 Testing maximize continuous LP solution
	 Testing unbounded continuous LP solution
	 Testing Long Names
	 Testing repeated Names
	 Testing zero constraint
	 Testing zero objective
	 Testing LpVariable (not LpAffineExpression) objective
	 Testing Long lines in LP
	 Testing LpAffineExpression divide
	 Testing MIP solution
	 Testing MIP solution with floats in objective
	 Testing MIP relaxation
	 Testing feasibility problem (no objective)
	 Testing an infeasible problem
	 Testing an integer infeasible problem
	 Testing column based modelling
	 Testing dual variables and slacks reporting
	 Testing fractional constraints
	 Testing elastic constraints (no change)
	 Testing elastic constraints (freebound)
	 Testing elastic constraints (penalty unchanged)
	 Testing elastic constraints (penalty unbounded)
* Solver pulp.solvers.PULP_CBC_CMD passed.
Solver pulp.solvers.CPLEX_DLL unavailable
Solver pulp.solvers.CPLEX_CMD unavailable
Solver pulp.solvers.CPLEX_PY unavailable
Solver pulp.solvers.COIN_CMD unavailable
Solver pulp.solvers.COINMP_DLL unavailable
	 Testing zero subtraction
	 Testing inconsistant lp solution
	 Testing continuous LP solution
	 Testing maximize continuous LP solution
	 Testing unbounded continuous LP solution
	 Testing Long Names
	 Testing repeated Names
	 Testing zero constraint
	 Testing zero objective
	 Testing LpVariable (not LpAffineExpression) objective
	 Testing LpAffineExpression divide
	 Testing MIP solution
	 Testing MIP solution with floats in objective
	 Testing MIP relaxation
	 Testing feasibility problem (no objective)
	 Testing an infeasible problem
	 Testing an integer infeasible problem
	 Testing column based modelling
	 Testing fractional constraints
	 Testing elastic constraints (no change)
	 Testing elastic constraints (freebound)
	 Testing elastic constraints (penalty unchanged)
	 Testing elastic constraints (penalty unbounded)
* Solver pulp.solvers.GLPK_CMD passed.
Solver pulp.solvers.XPRESS unavailable
Solver pulp.solvers.GUROBI unavailable
Solver pulp.solvers.GUROBI_CMD unavailable
Solver pulp.solvers.PYGLPK unavailable
Solver pulp.solvers.YAPOSIB unavailable
```

{{% jupyter_cell_end %}}