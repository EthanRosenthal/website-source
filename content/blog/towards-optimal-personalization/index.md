---
date: "2016-08-30"
title: "Towards optimal personalization: synthesisizing machine learning and operations research"
slug: "towards-optimal-personalization"
hasMath: true
notebook: true
---
{{% jupyter_cell_start markdown %}}

<!-- PELICAN_BEGIN_SUMMARY -->

Last [post]({{< ref "/blog/lets-talk-or" >}}) I talked about how data scientists probably ought to spend some time talking about optimization (but not too much time - I need topics for my blog posts!). While I provided a basic optimization example in that post, that may have not been so interesting, and there definitely wasn't any machine learning involved. 

<!-- PELICAN_END_SUMMARY -->

Right now, I think that the most exciting industrial applications of optimization are those that synthesize machine learning and optimization in order to obtain optimal personalization at scale.

Here, I'll talk about a more concrete use case of this synthesis that you might see at a company.

## All the ML and nowhere to go

Let's say you start working at a Software-As-A-Service (SAAS) company, and you end up in a meeting with the Marketing team. Everybody's talking about churn. Marketing has been trying all sorts of things - they've sent coupons, they've called customers, they've sent emails, and everything else in order to decrease churn. Some things work, some things are expensive, and there are lots of questions. Nobody knows SQL, so you offer to look at the data.

It turns out that it seems like there might be some clear differences in customers who eventually churn and customers who do not. You offer to build an algorithm to predict customer churn broken out by intervention medium (e.g. email, phone call, no intervention, etc...). 

You get the greenlight to hack away. Of course, this takes much longer than you or Marketing expects (because pretty much all machine learning does), but in the end you're left with multiple classification models that are well-tuned with a bunch of features.

*You're in a great place. You actually built machine learning models that work.*

But what now?

You can go the common route. You write a long script that will run the churn model every so often and populate a database with the results. You tell Marketing and everybody else that this information is now available, and you hope that they will use it. 

And they might. 

Or, those numbers will sit there.

Or, Marketing will randomly target the top X% of people most likely to churn with their expensive intervention (say, phone call) and email the rest. 

None of this is optimal.

## Optimization to the rescue

When there's lots of decisions to make and there's a clear goal, then optimization is a great friend to have. The goal here is to prevent churn. We will have some constraints (mainly money). Let's make up some data and walk through how to solve this in python.

## Defining (making up) the problem

We'll assume that we have 4 different types of churn prevention messages at 4 different prices:

| Media  | Price  |
|---|---|
| Email  | 0.25  |
| Push notification  | 0.30  |
| Text message  | 0.85  |
| Phone Call  | 5.00  |

Also, Marketing has a monthly budget of $2,000 to spend on messaging 2,000 customers. Let's optimize! We'll start by making up some fake outputs of your supposed machine learning model.

*Author's note: There are probably more dimensions to this problem that you would care about in reality. For example, a customer lifetime value (LTV) model would help in order to decide which customers are more valuable post churn prevention.*

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
import numpy as np
import pandas as pd
np.random.seed(2016)
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
n_customers = 2000

customer_ids = np.random.randint(1000000, 2000000, size=n_customers)

# Define retention success probabilities.
email_prob = np.random.random(size=n_customers) * 0.2
push_prob = np.random.random(size=n_customers) * 0.3
text_prob = np.random.random(size=n_customers) * 0.4
phone_prob = np.random.random(size=n_customers) * 0.9

prob_df = pd.DataFrame({'email': email_prob,
                   'push': push_prob,
                   'text': text_prob,
                   'phone': phone_prob},
                  index=customer_ids)
# assure column order
message_types = ['email', 'push', 'text', 'phone']
prob_df = prob_df[message_types]
prob_df.head()
```

{{% jupyter_input_end %}}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>email</th>
      <th>push</th>
      <th>text</th>
      <th>phone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1949099</th>
      <td>0.028448</td>
      <td>0.277781</td>
      <td>0.171171</td>
      <td>0.877097</td>
    </tr>
    <tr>
      <th>1291518</th>
      <td>0.143909</td>
      <td>0.214208</td>
      <td>0.281430</td>
      <td>0.014325</td>
    </tr>
    <tr>
      <th>1062730</th>
      <td>0.137871</td>
      <td>0.003360</td>
      <td>0.016220</td>
      <td>0.646562</td>
    </tr>
    <tr>
      <th>1397907</th>
      <td>0.143817</td>
      <td>0.048553</td>
      <td>0.140430</td>
      <td>0.268116</td>
    </tr>
    <tr>
      <th>1311949</th>
      <td>0.045251</td>
      <td>0.248514</td>
      <td>0.112957</td>
      <td>0.317091</td>
    </tr>
  </tbody>
</table>
</div>



{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Setting up the math: Objective + Constraints

As with all optimization problems, you should start with what you want to optimize (this is your objective function, cost function, loss function, or whatever else you want to call it). Here, we want to minimize churn, or, conversely, maximize retention. Let us define a matrix $M$ consisting of element variables $m\_{ij} \in \{0, 1\}$ which indicate whether or not customer $i$ receives a message from intervention medium $j$.

We can then use our retention probabilities as coeffecients to $M$ in order to create our objective function to maximize. We define $p\_{ij}$ as the probability that customer $i$ is retained by intervention medium $j$. Our objective function is thus

$$R = \sum\limits\_{i,j}p\_{ij}m\_{ij}$$

Let's reflect for a second on how cool this is. The *outputs* of a fancy machine learning model are being fed in as *inputs* to this new optimization problem.

Okay enough reflection - we have our function to maximize. The next question is, say it with me class, "What are the constraints?"

We only have two, and they're relatively simple:

* Each customer can only get one message.
$$\forall i \sum\limits\_{j}m\_{ij} = 1$$

* There's a limited budget of 2K. We'll take $c\_{j}$ to be the price of sending a message of type $j$.
$$\sum\limits\_{ij}m\_{ij} c\_{j} \leq 2,000$$

## Math $\Rightarrow$ Code

For ease of translation, I will use the same variable names in the code as I used in the above math. Note: some people might get angry if you do this in production code.

We start with defining index lists and the matrices for the constants in the problem.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Indices
customer_index = range(prob_df.shape[0]) # i
message_index = range(len(message_types)) # j
tier_index = range(4) # k
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Matrices of constants

# Retention probabilities
p = prob_df.as_matrix()
p.round(decimals=4)

# Pricing
c = [0.25, 0.30, 0.85, 5.00]

budget = 2000
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

With the constants defined, we can now build the variables to our optimization problem. Just like how we wrote out the math, we'll define our variables and objective function before adding the constraints. Similarly to my last post, I'll use the Python library [pulp](https://pythonhosted.org/PuLP/).

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
from pulp import *

# Create variables
prob = LpProblem('Churn Prevention Problem', LpMaximize)
m = {}
a = {}
t = {}

for i in customer_index:
    for j in message_index:
        m[i, j] = LpVariable('m_{}_{}'.format(i, j),
                             lowBound=0,
                             upBound=1,
                             cat='Binary')      
    
# Add the objective functions
prob += (lpSum([m[i, j] * p[i, j]
               for i in customer_index
               for j in message_index]),
        'Total retention probability')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Simple constraint
for i in customer_index:
    prob += (lpSum([m[i, j] for j in message_index]) == 1,
             'One message for cust {}'.format(i))
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Max budget constraint
prob += (lpSum([m[i, j] * c[j]
               for i in customer_index
               for j in message_index]) <= budget,
        'Budget')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Our problem is now fully defined, and we'll use [GLPK](https://www.gnu.org/software/glpk/) in order to solve it. I've also added a GLPK option to log the output so that we can view it in the Jupyter notebook.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
prob.solve(GLPK_CMD(msg=3, options=['--log', 'glpk.log']))
```

{{% jupyter_input_end %}}




    1



{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
with open('glpk.log', 'r') as f:
    for line in f:
        print(line.rstrip('\n'))
```

{{% jupyter_input_end %}}

    GLPSOL: GLPK LP/MIP Solver, v4.52
    Parameter(s) specified in the command line:
     --cpxlp /tmp/21270-pulp.lp -o /tmp/21270-pulp.sol --log glpk.log
    Reading problem data from '/tmp/21270-pulp.lp'...
    2001 rows, 8000 columns, 16000 non-zeros
    8000 integer variables, all of which are binary
    14597 lines were read
    GLPK Integer Optimizer, v4.52
    2001 rows, 8000 columns, 16000 non-zeros
    8000 integer variables, all of which are binary
    Preprocessing...
    2001 rows, 8000 columns, 16000 non-zeros
    8000 integer variables, all of which are binary
    Scaling...
     A: min|aij| =  2.500e-01  max|aij| =  5.000e+00  ratio =  2.000e+01
    Problem data seem to be well scaled
    Constructing initial basis...
    Size of triangular part is 2001
    Solving LP relaxation...
    GLPK Simplex Optimizer, v4.52
    2001 rows, 8000 columns, 16000 non-zeros
          0: obj =   9.052573077e+02  infeas =  8.000e+03 (0)
        500: obj =   7.312807825e+02  infeas =  5.625e+03 (0)
       1000: obj =   5.524409426e+02  infeas =  3.250e+03 (0)
       1500: obj =   3.750374626e+02  infeas =  8.750e+02 (0)
    *  1685: obj =   3.187530007e+02  infeas =  0.000e+00 (0)
    *  2000: obj =   3.600976718e+02  infeas =  5.568e-15 (0)
    *  2500: obj =   4.024145484e+02  infeas =  2.986e-14 (0)
    *  3000: obj =   4.352597569e+02  infeas =  9.182e-14 (0)
    *  3500: obj =   4.811161931e+02  infeas =  1.698e-13 (0)
    *  4000: obj =   5.376977102e+02  infeas =  3.274e-13 (0)
    *  4500: obj =   5.894224210e+02  infeas =  6.210e-13 (0)
    *  5000: obj =   6.144242918e+02  infeas =  7.468e-13 (0)
    *  5181: obj =   6.165775504e+02  infeas =  0.000e+00 (0)
    OPTIMAL LP SOLUTION FOUND
    Integer optimization begins...
    +  5181: mip =     not found yet <=              +inf        (1; 0)
    Solution found by heuristic: 616.329183066
    Solution found by heuristic: 616.397534532
    Solution found by heuristic: 616.450556569
    Solution found by heuristic: 616.487520962
    Solution found by heuristic: 616.560272941
    Solution found by heuristic: 616.565833175
    +  5734: >>>>>   6.165693049e+02 <=   6.165769339e+02 < 0.1% (205; 103)
    +  5830: >>>>>   6.165699159e+02 <=   6.165768357e+02 < 0.1% (190; 215)
    +  5898: >>>>>   6.165734196e+02 <=   6.165768069e+02 < 0.1% (206; 251)
    +  5957: >>>>>   6.165740306e+02 <=   6.165766934e+02 < 0.1% (124; 454)
    +  6063: >>>>>   6.165749676e+02 <=   6.165762775e+02 < 0.1% (125; 509)
    +  6110: >>>>>   6.165755786e+02 <=   6.165761893e+02 < 0.1% (65; 664)
    +  6174: mip =   6.165755786e+02 <=     tree is empty   0.0% (0; 853)
    INTEGER OPTIMAL SOLUTION FOUND
    Time used:   4.9 secs
    Memory used: 10.8 Mb (11354098 bytes)
    Writing MIP solution to `/tmp/21270-pulp.sol'...


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
print("Status: {}".format(LpStatus[prob.status]))
```

{{% jupyter_input_end %}}

    Status: Optimal


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Voil&aacute;! We now have a fully personalized strategy for churn prevention and can now explore the solution to our hearts' content.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
for j, m_type in enumerate(message_types):
    print('Total {} messages: {}'.format(
            m_type, np.sum([m[i, j].varValue for i in customer_index])))
```

{{% jupyter_input_end %}}

    Total email messages: 312
    Total push messages: 744
    Total text messages: 728
    Total phone messages: 216


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
total_cost = np.sum([m[i, j].varValue * c[j]
                     for i in customer_index
                     for j in message_index])
print('Total cost: ${:.2f}'.format(np.round(total_cost, 2)))
```

{{% jupyter_input_end %}}

    Total cost: $2000.00


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
obj = np.sum([m[i, j].varValue * p[i, j]
              for i in customer_index
              for j in message_index])
print('Probability sum (i.e. the objective function): {}'.format(obj))
```

{{% jupyter_input_end %}}

    Probability sum (i.e. the objective function): 616.575578582


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

## Tricks of the optimization trade

That was fun, but now I'd like to complicate things (and make them more realistic) by assuming that there are pricing plans for each of these message types that change with number of messages sent:

| Media  | <500  |  500-999  | 1,000-1,499  | >1,500  |
|---|---|---|---|---|
| Email  | 0.25  | 0.20  | 0.15  | 0.10  |
| Push notification  | 0.30  | 0.25  | 0.225  | 0.20  |
| Text message  | 0.85  | 0.65  | 0.50  | 0.30  |
| Phone Call  | 5.00  | 5.00  | 5.00  | 5.00  |

You employ your own employees for the phones, so this cost does not benefit from economies of scale.

The budget constraint will now have to be rewritten, and it will be much more difficult. Seriously, it's pretty tricky.

With the tiered pricing, we introduce step functions to the cost. What follows is the best method I could think up to solve this problem, but I would love to know if anybody has better ideas.

To start, we'll consider $<$500 messages to be our *base* tier. We will then call each subsequent tier Tier 1, Tier 2, and Tier 3 (500-1,000, 1,000-1,499, and $>$1,500 messages, respectively). We must create two new types of variables. 

The first will be indicator variables which indicate whether or not we have "activated" a particular tier for a particular message type. Mathematically, let us define $a\_{jk} \in \{0, 1\}$ to indicate whether or not we have sold *at least* the minimum amount of messages of type $j$ in Tier $k$.

The second type of variable will be integer variables $t\_{jk} \in \mathbb{Z}\_{\geq 0}$ which tell us how many messages of type $j$ that we have sent above the minimum amount of messages for tier $k$. The way that we will calculate the price for sending $N$ messages of type $j$ will be to calculate the base cost and then subtract "discounts" from the total cost as we move up in tiers.

Okay, to summarize, we need to define both $a\_{jk}$, our tier activation indicators, and $t\_{jk}$, our tier message counters. Let's let $l\_{k}$ be the minimum (lower) amount of messages needed to be sent in order to move up in a Tier. With the following two constraints, the behavior of $a$ will be fully defined

$$\forall j, k \geq 1 \quad \sum\limits\_{i}m\_{ij} \leq l\_{k} - 1 + X a\_{jk}$$
$$\forall j,k \geq 1 \quad \sum\limits\_{i}m\_{ij} \geq l\_{k} - X(1 - a\_{jk})$$

where $X$ is a sufficiently large number (like the maximum of $\sum\limits\_{i}m\_{ij}$). This technique of using this large number is called the [Big M method](https://en.wikipedia.org/wiki/Big_M_method) from the common nomenclature of using an $M$ instead of my choice of $X$ (I already used an $M$ earlier!). The two constraints above are quite confusing. I know this much is true because it took me a long time to make sure that I had them right. I would suggest walking through an example to make sure that you understand what's going on. You can pick a single $k$, like the first Tier where $l\_{1} = 500$. Now, imagine if 499 messages of some type $j$ had been sent such that $\sum\limits\_{i}m\_{ij} = 499$. What do the constraints say that $a\_{j1}$ must be? Now, perform the same test for 500 and 501 messages and assure yourself that the constraints never disagree with each other and that $a\_{j1}$ flips from 0 to 1 when it is supposed to.

With $a$ defined, it is one more set of mental gymnastics in order to define $t$:

$$\forall j,k \geq 1 \quad  t\_{jk} \leq Xa\_{jk}$$
$$\forall j,k \geq 1 \quad  t\_{jk} + l\_{jk} \geq \sum\limits\_{i}m\_{ij} - X (1 - a\_{jk})$$
$$\forall j,k \geq 1 \quad  t\_{jk} + l\_{jk} \leq \sum\limits\_{i}m\_{ij} + X (1 - a\_{jk})$$

The first constraint ensures that if $a$ is 0 then $t$ must be 0, as well. The other two constraints guarantee that when $a$ is 1, then $t\_{jk} = \sum\limits\_{i}m\_{ij} - l\_{jk}$ which is exactly the number of messages greater than the minimum threshold for Tier $k$.

With all of these god forsaken constraints defined, we can finally write our budget constraint. We will take $c\_{jk}$ to be the cost for sending a single message from Tier $k$ where $k=0$ is the base cost. For $k>0$, we will take $c\_{jk}$ to be the extra discount obtained by reaching that tier (it is positive, and we subtract it from the base cost).

$$\sum\limits\_{j}\Big(\sum\limits\_{i}m\_{ij}c\_{j0} - \sum\limits\_{k \geq 1}t\_{jk}c\_{jk}\Big) \leq 2,000$$

Phew!

## Math $\Rightarrow$ Code Redux

Finally, we get back to the code. We can reuse some of the old variables, but we'll need to rewrite our cost matrix and add some tier information.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# New Indices
tier_index = range(4) # k

# New matrices of constants

# Tier pricing
c = np.array([[0.25, 0.20, 0.15, 0.10],
              [0.30, 0.25, 0.225, 0.20],
              [0.85, 0.65, 0.50, 0.30],
              [5.00, 5.00, 5.00, 5.00]])

# Recall that we must transform c
# to get the marginal discount.
# There's probably a fancy numpy way to do this
# but ain't nobody got time for that.

# Note we must slice in reverse if we
# want to do this in place.
for j in range(c.shape[1] - 1, 0, -1):
    c[:, j] -= c[:, j-1]

c = np.abs(c)
    
# Tier thresholds
l = [0, 500, 1000, 1500]
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

We solve the problem similarly to before but need our two new variables along with their associated constraints.

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Create variables
prob = LpProblem('Tiered Churn Prevention Problem', LpMaximize)
m = {}
a = {}
t = {}

for i in customer_index:
    for j in message_index:
        m[i, j] = LpVariable('m_{}_{}'.format(i, j),
                             lowBound=0,
                             upBound=1,
                             cat='Binary')
for j in message_index:
    for k in tier_index[1:]:
        a[j, k] = LpVariable('a_{}_{}'.format(j, k),
                             lowBound=0,
                             upBound=1,
                             cat='Binary')
        
        t[j, k] = LpVariable('t_{}_{}'.format(j, k),
                             lowBound=0,
                             upBound=len(customer_index) - l[k],
                             cat='Integer')
        
    
# Add the objective functions
prob += (lpSum([m[i, j] * p[i, j]
               for i in customer_index
               for j in message_index]),
        'Total retention probability')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Simple constraint
for i in customer_index:
    prob += (lpSum([m[i, j] for j in message_index]) == 1,
             'One message for cust {}'.format(i))
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Hard pricing tier constraints
X = len(customer_index) # Max value of sum_{i}(m_{ij})

for j in message_index:
    m_sum = lpSum([m[i, j] for i in customer_index])
    for k in tier_index[1:]:
        prob += (m_sum <= l[k] - 1 + X*a[j, k],
                 'hard constraint 1 {}_{}'.format(j, k))
        prob += (m_sum >= l[k] - X*(1 - a[j, k]),
                 'hard constraint 2 {}_{}'.format(j, k))
        prob += (t[j, k] <= X * a[j, k],
                 'hard constraint 3 {}_{}'.format(j, k))
        prob += (t[j, k] + l[k] >= m_sum - X*(1 - a[j, k]),
                 'hard constraint 4 {}_{}'.format(j, k))
        prob += (t[j, k] + l[k] <= m_sum + X*(1 - a[j, k]),
                 'hard constraint 5 {}_{}'.format(j, k))
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
# Max budget constraint
prob += (lpSum([lpSum([m[i, j] * c[j, 0] for i in customer_index])
               - lpSum([t[j, k] * c[j, k] for k in tier_index[1:]])
               for j in message_index]) <= budget,
        'Budget')
```

{{% jupyter_input_end %}}

{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Even with our relatively small problem of 2000 customers, GLPK has a bit of difficulty in converging when using the default optimization parameters. I'll add a slightly smaller ```mipgap``` (difference between upper bound and current feasible solution) in order to lower the tolerance for this example. Note that this is not necessarily GLPK's fault. There are other constraints that I could have written that do help the solver e.g.

$$\forall j, k \geq 1 \quad t\_{j, k-1} \geq t\_{j, k} $$

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
glpk_options = ['--log', 'glpk.log', '--mipgap', '0.0001']
prob.solve(GLPK_CMD(msg=3, options=glpk_options))
```

{{% jupyter_input_end %}}




    1



{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
with open('glpk.log', 'r') as f:
    for line in f:
        print(line.rstrip('\n'))
```

{{% jupyter_input_end %}}

    GLPSOL: GLPK LP/MIP Solver, v4.52
    Parameter(s) specified in the command line:
     --cpxlp /tmp/21270-pulp.lp -o /tmp/21270-pulp.sol --log glpk.log --mipgap
     0.0001
    Reading problem data from '/tmp/21270-pulp.lp'...
    2061 rows, 8024 columns, 112105 non-zeros
    8024 integer variables, 8012 of which are binary
    28352 lines were read
    GLPK Integer Optimizer, v4.52
    2061 rows, 8024 columns, 112105 non-zeros
    8024 integer variables, 8012 of which are binary
    Preprocessing...
    48 constraint coefficient(s) were reduced
    2061 rows, 8024 columns, 112105 non-zeros
    8024 integer variables, 8012 of which are binary
    Scaling...
     A: min|aij| =  2.500e-02  max|aij| =  2.000e+03  ratio =  8.000e+04
    GM: min|aij| =  4.076e-01  max|aij| =  2.454e+00  ratio =  6.020e+00
    EQ: min|aij| =  1.662e-01  max|aij| =  1.000e+00  ratio =  6.016e+00
    2N: min|aij| =  1.250e-01  max|aij| =  1.466e+00  ratio =  1.173e+01
    Constructing initial basis...
    Size of triangular part is 2061
    Solving LP relaxation...
    GLPK Simplex Optimizer, v4.52
    2061 rows, 8024 columns, 112105 non-zeros
          0: obj =   9.052573077e+02  infeas =  3.002e+04 (0)
        500: obj =   7.318330422e+02  infeas =  1.729e+04 (0)
       1000: obj =   7.313162448e+02  infeas =  1.727e+04 (0)
       1500: obj =   5.770586355e+02  infeas =  8.593e+03 (0)
       2000: obj =   5.768708977e+02  infeas =  8.558e+03 (0)
       2500: obj =   4.002909225e+02  infeas =  1.818e+03 (0)
       3000: obj =   4.002909225e+02  infeas =  1.818e+03 (0)
    *  3194: obj =   3.519623224e+02  infeas =  0.000e+00 (0)
    *  3500: obj =   4.194004898e+02  infeas =  1.628e-14 (0)
    *  4000: obj =   4.763028465e+02  infeas =  6.725e-15 (0)
    *  4500: obj =   5.312186300e+02  infeas =  9.320e-13 (0)
    *  5000: obj =   5.841207596e+02  infeas =  0.000e+00 (0)
    *  5500: obj =   6.213808386e+02  infeas =  1.005e-15 (0)
    *  6000: obj =   6.556284187e+02  infeas =  4.441e-16 (0)
    *  6291: obj =   6.722113467e+02  infeas =  4.686e-28 (0)
    OPTIMAL LP SOLUTION FOUND
    Integer optimization begins...
    +  6291: mip =     not found yet <=              +inf        (1; 0)
    Solution found by heuristic: 608.911217118
    + 10112: mip =   6.089112171e+02 <=   6.672873502e+02   9.6% (60; 1)
    + 10372: mip =   6.089112171e+02 <=   6.672873502e+02   9.6% (154; 1)
    + 11536: mip =   6.089112171e+02 <=   6.633982957e+02   8.9% (202; 7)
    Solution found by heuristic: 612.579768421
    + 11865: mip =   6.125797684e+02 <=   6.633982957e+02   8.3% (302; 7)
    + 16301: mip =   6.125797684e+02 <=   6.518788965e+02   6.4% (184; 394)
    Solution found by heuristic: 613.936474108
    + 16607: mip =   6.139364741e+02 <=   6.518788965e+02   6.2% (296; 394)
    + 16787: mip =   6.139364741e+02 <=   6.518788965e+02   6.2% (418; 394)
    + 18177: mip =   6.139364741e+02 <=   6.485283998e+02   5.6% (301; 740)
    + 18567: mip =   6.139364741e+02 <=   6.485283998e+02   5.6% (427; 740)
    + 18902: mip =   6.139364741e+02 <=   6.485283998e+02   5.6% (567; 740)
    + 20030: mip =   6.139364741e+02 <=   6.485283998e+02   5.6% (688; 740)
    Solution found by heuristic: 622.714526489
    Time used: 60.0 secs.  Memory used: 20.9 Mb.
    + 21481: mip =   6.227145265e+02 <=   6.366592257e+02   2.2% (795; 743)
    + 21623: mip =   6.227145265e+02 <=   6.366592257e+02   2.2% (870; 743)
    Solution found by heuristic: 624.961551511
    Solution found by heuristic: 625.000663038
    Solution found by heuristic: 625.049770205
    Solution found by heuristic: 625.098838944
    Solution found by heuristic: 625.147751412
    + 22462: mip =   6.251477514e+02 <=   6.251976645e+02 < 0.1% (52; 2498)
    RELATIVE MIP GAP TOLERANCE REACHED; SEARCH TERMINATED
    Time used:   71.6 secs
    Memory used: 24.5 Mb (25716666 bytes)
    Writing MIP solution to `/tmp/21270-pulp.sol'...


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
print("Status: {}".format(LpStatus[prob.status]))
```

{{% jupyter_input_end %}}

    Status: Optimal


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Took a little longer, huh?

{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
for j, m_type in enumerate(message_types):
    print('Total {} messages: {}'.format(
            m_type, np.sum([m[i, j].varValue for i in customer_index])))
```

{{% jupyter_input_end %}}

    Total email messages: 243
    Total push messages: 691
    Total text messages: 848
    Total phone messages: 218


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
total_cost = np.sum([np.sum([m[i, j].varValue * c[j, 0] for i in customer_index])
                     - np.sum([t[j, k].varValue * c[j, k] for k in tier_index[1:]])
                     for j in message_index])
print('Total cost: ${:.2f}'.format(np.round(total_cost, 2)))
```

{{% jupyter_input_end %}}

    Total cost: $1999.70


{{% jupyter_cell_end %}}{{% jupyter_cell_start code %}}


{{% jupyter_input_start %}}

```python
obj = np.sum([m[i, j].varValue * p[i, j]
              for i in customer_index
              for j in message_index])
print('Probability sum (i.e. the objective function): {}'.format(obj))
```

{{% jupyter_input_end %}}

    Probability sum (i.e. the objective function): 625.147751412


{{% jupyter_cell_end %}}{{% jupyter_cell_start markdown %}}

Note that we get to a higher objective value with these pricing tier discounts.

{{% jupyter_cell_end %}}