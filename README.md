
# Sampling and the Central Limit Theorem

![sample](https://media.giphy.com/media/OsOP6zRwxrnji/giphy.gif)

# Agenda 

1. Differentiate terms: discriptive/inferential statistics population/sample, paramater/statistic, sample distribution/sampling distribution
2. Use Numpy to randomly sample a distribution
3. Describe the central limit theorem and connect it to our knowledge of distributions and sampling.
4. Define and Calculate Standard Error

## Probability vs Statistics
- Probability starts with known probabilities and obtains how probable any particular observation would be
- Statistics works the other way around. Start with and observations (data) and try to determine its probability

## Descriptive vs Inferential Statistics
- Descriptive Statistics
   > simply describe what is observed. The average height of a high school football team can be directly calculated by measuring all of the current players height.
- Inferential statistics 
    > try to say something general about a larger group of subjects than those we have measured. For example, we would be doing inferential statistics if we wanted to know about the average height of all high school football teams.
    - To put it another way, statistical inference is the process by which we take observations of a subset of a group and generalize to the whole group.

## Population Inference

The mayor's office has hired Flatiron Data Science Immersive students to determine a way to fix traffic congestion. A good starting point is to determine what proportion of the population of Seattle owns a car.

![traffic](https://media.giphy.com/media/3orieWY8RCodjD4qqs/giphy.gif)

In order for us to make any determinations about a population, we must first get information about it.

Because it's usually completely impractical to get data about *everyone* in a population, we must take a sample.

## Key Terms
 - the entire group is known as the **population**  
 - the subset is a known as the **sample**


![pop](./img/sample_pop.png)

- We would use samples if the population is:
    - Too big to enumerate
    - too difficult/time consuming or expensive to sample in its entirety.

**Random sampling is not easy to do**  
Continuing our Seattle car example, how would we take a sample? 

Here are two strategies we might employ:

* Stand outside of Flatiron at 12 pm and ask random people until *n* responses


* Go to a randomly assigned street corner and at a random time and ask *n* people if they own a car

Which strikes you as better?

What do we want our sample to look like?

In particular, what relationship do we want between the sample and the population? What steps can we take to improve our odds of success in achieving this?

# Discussion

![talk amongst yourselves](https://media.giphy.com/media/l2SpQRuCQzY1RXHqM/giphy.gif)




## Population v Sample Terminology
Characteristics of populations are called **parameters**

Characteristics of a sample are called **statistics**

A sample statistic is a **point estimate** of the population parameter

![imgsample](./img/sample_stats.png)

# A Simulation to Reinforce Our Definitions

Let's create a population of systolic blood pressure of adult males in Chicago, assuming a mean of 114 mmHg with a standard deviation of 11 mmHg.  We will also assume the adult male population to be 1.5 million. 

It is impossible to measure the systolic blood pressure of every man in Chicago, but let's assume multiple investigations have led to the conclusion the the mean and std of this population is 114 and 11, respecively. These are therefore estimators of the population parameter.

$\Large\hat\mu = 114$  
$\Large\hat\sigma = 11$




```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pop = int(1.5*10**6)
# Use numpy to generate a normal distribution of the 
sys_pop = np.random.normal(loc=114, scale=11, size=pop)

fig, ax = plt.subplots()

sns.kdeplot(sys_pop, ax=ax, shade=True)
ax.set_title('Distribution of Adult Male Systolic Blood Pressure')
ax.set_xlabel('Systolic BP')
```




    Text(0.5, 0, 'Systolic BP')




![png](index_files/index_13_1.png)


Let's then imagine we develop an effective manner of random sampling, and simulate with numpy. Our sample size is 40 people.



```python
sample_size = 40
sample = np.random.choice(sys_pop, sample_size)

# We can look at the distribution of the values in the sample.
```


```python
fig, ax = plt.subplots()
sns.distplot(sample, ax=ax, bins=15)
ax.set_title('Sample Distribution of Systolic BP Measurements')
```




    Text(0.5, 1.0, 'Sample Distribution of Systolic BP Measurements')




![png](index_files/index_16_1.png)


We can then calculate the sample statistics:


```python
print(f'Sample mean: {sample.mean()}')
print(f'Sample standard deviation: {sample.std()}')
print(f'Sample median: {np.median(sample)}')
```

    Sample mean: 115.13518836266857
    Sample standard deviation: 10.8170085734385
    Sample median: 114.23470724109055


If we repeated this process, taking samples of the population repeatedly, we would get an array of sample statistics.


```python

number_of_samples = 1000
sample_size = 50
sample_stats = []

for _ in range(number_of_samples):
    sample = np.random.choice(sys_pop, sample_size)
    # collect the mean of each of the 1000 samples in sample stats
    sample_stats.append(sample.mean())

```

The collection of sample stats represents our __sampling distribution__


```python
fig, ax = plt.subplots()
ax.hist(sorted(sample_stats), bins=20)
ax.set_title('Sampling Distribution\n of Systolic BP')
ax.set_xlabel("Systolic Blood Pressure")
ax.set_ylabel('Count');
```


![png](index_files/index_22_0.png)


An interesting property of this sampling distribution:
    
As we continue to sample, the mean of the sampling distribution gets closer and closer to the population mean.


```python
pop_mean = 114
pop_sample_difference = []
for n in range(1,1000):
    number_of_samples = n
    sample_size = 20
    sample_stats = []
    for _ in range(number_of_samples):
        sample = np.random.choice(sys_pop, sample_size)
        # collect the mean of each of the 1000 samples in sample stats
        sample_stats.append(sample.mean())
        
    pop_sample_difference.append(np.absolute(pop_mean - np.mean(sample_stats)))
    
```


```python
fig, ax = plt.subplots()
ax.bar(list(range(1,1000)), pop_sample_difference)
```




    <BarContainer object of 999 artists>




![png](index_files/index_25_1.png)


# Word Exercise 
Put the variables in the correct place.



```python

var_1 = 'population'
var_2 = 'sample'
var_3 = 'point estimate'
var_4 = 'statistic'
var_5 = 'parameter'
var_6 = 'sampling'


print(f"""We sampled 40 bee hives and calcuted the mean colony population 
          to be 75,690 bees. 75,690 is a {var_1} of the population paramter\n""")

print(f"""We repeatedly sample 40 people at random from Seattle and 
        measure their heart rate,then calculate the mean of each sample. 
        We call the plot of this collection of statistics
        the {var_2} distribution.
        """)

print(f"""There are exactly 58 Javan Rhino's left in the wild. 
        Their mean length has been measured accurately at 5 feet.
        This mean length is considered a population {var_3}. 
        """)

print(f"""If we plot a histogram of individual pistil lengths 
      measured on 50 hibiscus flowers, we would be plotting the distribution 
      of an attribute of our {var_4} of hibiscus flowers. 
        """)

print(f"""Since every restaurant in Chicago is required by law to register
        with the city, we can accurately count the number of active pizza restaurants
        actively operating right now.  This group represents the {var_5} of actively 
        operating, registered pizza restaurants in Chicago.
    """)

print(f"""The mean number of hourly hits to Jelle's Marble Racing website 
            randomly sampled across a seven day period represents a sample
            {var_6}.
        """)
```

    We sampled 40 bee hives and calcuted the mean colony population 
              to be 75,690 bees. 75,690 is a population of the population paramter
    
    We repeatedly sample 40 people at random from Seattle and 
            measure their heart rate,then calculate the mean of each sample. 
            We call the plot of this collection of statistics
            the sample distribution.
            
    There are exactly 58 Javan Rhino's left in the wild. 
            Their mean length has been measured accurately at 5 feet.
            This mean length is considered a population point estimate. 
            
    If we plot a histogram of individual pistil lengths 
          measured on 50 hibiscus flowers, we would be plotting the distribution 
          of an attribute of our statistic of hibiscus flowers. 
            
    Since every restaurant in Chicago is required by law to register
            with the city, we can accurately count the number of active pizza restaurants
            actively operating right now.  This group represents the parameter of actively 
            operating, registered pizza restaurants in Chicago.
        
    The mean number of hourly hits to Jelle's Marble Racing website 
                randomly sampled across a seven day period represents a sample
                sampling.
            



```python
#__SOLUTION__
# Word Exercise

var_1 = 'population'
var_2 = 'sample'
var_3 = 'point estimate'
var_4 = 'statistic'
var_5 = 'parameter'
var_6 = 'sampling'


print(f"""We sampled 40 bee hives and calcuted the mean colony population 
          to be 75,690 bees. 75,690 is a {var_3} of the population paramter\n""")

print(f"""We repeatedly sample 40 people at random from Seattle and 
        measure their heart rate,then calculate the mean of each sample. 
        We call the plot of this collection of statistics
        the {var_6} distribution.
        """)

print(f"""There are exactly 58 Javan Rhino's left in the wild. 
        Their mean length has been measured accurately at 5 feet.
        This mean length is considered a population {var_5}. 
        """)

print(f"""If we plot a histogram of individual pistil lengths 
      measured on 50 hibiscus flowers, we would be plotting the distribution 
      of an attribute of our {var_2} of hibiscus flowers. 
        """)

print(f"""Since every restaurant in Chicago is required by law to register
        with the city, we can accurately count the number of active pizza restaurants
        actively operating right now.  This group represents the {var_1} of actively 
        operating, registered pizza restaurants in Chicago.
    """)

print(f"""The mean number of hourly hits to Jelle's Marble Racing website 
            randomly sampled across a seven day period represents a sample
            {var_4}.
        """)
```

    We sampled 40 bee hives and calcuted the mean colony population 
              to be 75,690 bees. 75,690 is a point estimate of the population paramter
    
    We repeatedly sample 40 people at random from Seattle and 
            measure their heart rate,then calculate the mean of each sample. 
            We call the plot of this collection of statistics
            the sampling distribution.
            
    There are exactly 58 Javan Rhino's left in the wild. 
            Their mean length has been measured accurately at 5 feet.
            This mean length is considered a population parameter. 
            
    If we plot a histogram of individual pistil lengths 
          measured on 50 hibiscus flowers, we would be plotting the distribution 
          of an attribute of our sample of hibiscus flowers. 
            
    Since every restaurant in Chicago is required by law to register
            with the city, we can accurately count the number of active pizza restaurants
            actively operating right now.  This group represents the population of actively 
            operating, registered pizza restaurants in Chicago.
        
    The mean number of hourly hits to Jelle's Marble Racing website 
                randomly sampled across a seven day period represents a sample
                statistic.
            



# 2. Use numpy to randomly sample a distribution





## Group Exercise

Below, we have four different sample scenarios.  Each group will code out the following: 

You are given a "population" to sample from based on the type of distribution.

1. Take a random sample of size n, where n > 30, from the population and calculate the mean of that population.

2. Repeat the sample n numbers of times (n = 1000). 

3. Plot the sampling distribution


```python
mccalister = ['Adam', 'Amanda','Chum', 'Dann', 
 'Jacob', 'Jason', 'Johnhoy', 'Karim', 
'Leana','Luluva', 'Matt', 'Maximilian' ]

for n in range(1,4):
    group = np.random.choice(mccalister, 4, replace=False)
    print(f'group {n}', group)
    for name in list(group):
        mccalister.remove(name)

```

    group 1 ['Maximilian' 'Luluva' 'Dann' 'Matt']
    group 2 ['Karim' 'Chum' 'Leana' 'Jacob']
    group 3 ['Jason' 'Amanda' 'Adam' 'Johnhoy']


## Group 1:

A bowler on the PBA rolls a strike 60% of the time. The population strikes of all games ever bowled is stored in in the population variable below.



```python
population = np.random.binomial(12, .6, 10000)
fig, ax = plt.subplots()
ax.bar(range(0,12), np.unique(population, return_counts=True)[1])
ax.set_title('Strikes Per Game')
```




    Text(0.5, 1.0, 'Strikes Per Game')




![png](index_files/index_35_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 20)
```




    (array([  5.,   7.,  13.,  24.,  45.,  76.,  75., 148., 108., 133., 117.,
             91.,  66.,  36.,  35.,  15.,   3.,   0.,   2.,   1.]),
     array([6.46 , 6.541, 6.622, 6.703, 6.784, 6.865, 6.946, 7.027, 7.108,
            7.189, 7.27 , 7.351, 7.432, 7.513, 7.594, 7.675, 7.756, 7.837,
            7.918, 7.999, 8.08 ]),
     <a list of 20 Patch objects>)




![png](index_files/index_36_1.png)


## Group 2:

Stored in the variable below is the number of pieces of mail that arrive per week at your door for each of the 4500 weeks in your life.  


```python
mail_population = np.random.poisson(3, 4500)
counts = np.unique(mail_population, return_counts=True)

fig, ax = plt.subplots()
ax.bar(np.unique(counts[0]), counts[1])
ax.set_title('Distribution of Pieces of Mail/Week')
ax.set_xlabel("Pieces of Mail")
```




    Text(0.5, 0, 'Pieces of Mail')




![png](index_files/index_38_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(mail_population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 30)
ax.set_title('Sample Means of Pieces of Mail\n Arriving at your door')
ax.set_xlabel('Number of pieces of mail')
```




    Text(0.5, 0, 'Number of pieces of mail')




![png](index_files/index_39_1.png)


# Group 3 

The population data for the number of minutes between customers arriving in a Piggly Wiggly is stored in the variable piggly_population.


```python
# on average, 20 customers enter per hour
piggly_population = np.random.exponential(1/(20/60), size=10000)
fig, ax = plt.subplots()
ax.hist(piggly_population, bins = 50, normed=True)
ax.set_title('Sample Means of Time Between Piggle Wiggly Customers')
ax.set_xlabel('Minutes');
```

    /Users/johnmaxbarry/.local/lib/python3.7/site-packages/ipykernel_launcher.py:4: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      after removing the cwd from sys.path.



![png](index_files/index_41_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(piggly_population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 30);
ax.set_title("""Sample means of number of minutes\n between people entering a Piggly Wiggly""")
ax.set_xlabel("Number of minutes between customers")
```




    Text(0.5, 0, 'Number of minutes between customers')




![png](index_files/index_42_1.png)


# 3. Central Limit Theorem

If we take repeated samples of a population, the sampling distribution of sample means will approximate to a normal distribution, no matter the underlying distribution!

## $E(\bar{x_{n}}) = \mu$

as n --> "large"

[good D3 example](https://seeing-theory.brown.edu/probability-distributions/index.html)

[good video demonstration](https://www.youtube.com/watch?v=jvoxEYmQHNM)


Let's look at an example taken from the ubiquitous Iris dataset. This histogram represents the distributions of sepal length:


![probgif](./img/probability-basics.gif)

https://www.kaggle.com/tentotheminus9/central-limit-theorem-animation

As we will see in hypothesis testing, pairing this theorem with the Empirical rule will be very powerful.

![empirical](img/empirical_rule.png)



Knowing that any sampling distribtion, no matter the underlying population distribution, will approach normality, we will be able to judge, given the empirical rule, how rare a given sample statistic is.  

## Bike Example
A bicycle advocacy group has come to us to see if it would make sense to increase the length of time users of Capital Bike Share have to ride on their bikes before they have to return them. Let's analyze a collection of Capital Bike Share data to determine if we should lengthen the time people have with their bikes.

Let's head over [here](https://s3.amazonaws.com/capitalbikeshare-data/index.html) for some DC bike data!


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
%matplotlib inline

```


```python
! wget 'https://s3.amazonaws.com/capitalbikeshare-data/202003-capitalbikeshare-tripdata.zip' -O temp.zip
! unzip temp.zip
```

    --2020-05-30 17:59:40--  https://s3.amazonaws.com/capitalbikeshare-data/202003-capitalbikeshare-tripdata.zip
    Resolving s3.amazonaws.com... 52.216.85.109
    Connecting to s3.amazonaws.com|52.216.85.109|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3835293 (3.7M) [application/zip]
    Saving to: 'temp.zip'
    
    temp.zip            100%[===================>]   3.66M  2.57MB/s    in 1.4s    
    
    2020-05-30 17:59:41 (2.57 MB/s) - 'temp.zip' saved [3835293/3835293]
    
    Archive:  temp.zip
      inflating: 202003-capitalbikeshare-tripdata.csv  



```python
df = pd.read_csv('202003-capitalbikeshare-tripdata.csv')
```


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Duration</th>
      <th>Start date</th>
      <th>End date</th>
      <th>Start station number</th>
      <th>Start station</th>
      <th>End station number</th>
      <th>End station</th>
      <th>Bike number</th>
      <th>Member type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>162525</th>
      <td>673</td>
      <td>2020-03-31 23:17:58</td>
      <td>2020-03-31 23:29:12</td>
      <td>31235</td>
      <td>19th St &amp; Constitution Ave NW</td>
      <td>31265</td>
      <td>5th St &amp; Massachusetts Ave NW</td>
      <td>W22920</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>162526</th>
      <td>514</td>
      <td>2020-03-31 23:18:13</td>
      <td>2020-03-31 23:26:47</td>
      <td>31203</td>
      <td>14th &amp; Rhode Island Ave NW</td>
      <td>31324</td>
      <td>18th &amp; New Hampshire Ave NW</td>
      <td>21054</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>162527</th>
      <td>1524</td>
      <td>2020-03-31 23:29:00</td>
      <td>2020-03-31 23:54:25</td>
      <td>31110</td>
      <td>20th St &amp; Florida Ave NW</td>
      <td>31403</td>
      <td>5th &amp; Kennedy St NW</td>
      <td>W24341</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>162528</th>
      <td>565</td>
      <td>2020-03-31 23:41:17</td>
      <td>2020-03-31 23:50:43</td>
      <td>31603</td>
      <td>1st &amp; M St NE</td>
      <td>31256</td>
      <td>10th &amp; E St NW</td>
      <td>W22691</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>162529</th>
      <td>1054</td>
      <td>2020-03-31 23:45:05</td>
      <td>2020-04-01 00:02:40</td>
      <td>31325</td>
      <td>Reservoir Rd &amp; 38th St NW</td>
      <td>31214</td>
      <td>17th &amp; Corcoran St NW</td>
      <td>W24051</td>
      <td>Member</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert duration from seconds to minutes
trip_durations = df['Duration'] / 60
trip_durations_2hr = trip_durations[trip_durations < 2*60]
```

#### The length of a bike ride

What shape do you expect the distribution of trip durations to have?

#### Get population statistics


```python
trip_durations_2hr.hist(bins=100);
```


![png](index_files/index_57_0.png)



```python
trip_durations_2hr.median()
```




    12.083333333333334



Let's treat the whole dataset as our population.


```python
pop_mean = trip_durations.mean()
pop_std = trip_durations.std()
print(f'pop_mean is {pop_mean} \npop_std is {pop_std}')
```

    pop_mean is 18.978877745646678 
    pop_std is 36.47173596361402



```python
def one_sample_mean(population):
    sample = np.random.choice(population, size=200, replace=True)
    return sample.mean()
```


```python
one_sample_mean(trip_durations_2hr)
```




    17.846333333333334



### When we take multiple samples from the distribution,and plot the means of each sample, the shape of the curve shifts


```python
d = [one_sample_mean(trip_durations) for i in range(1000)]
plt.hist(d, bins=50)
plt.title('''.
'''
);
```


![png](index_files/index_64_0.png)



```python
import seaborn as sns

def central_limit_theorem_plotter(distribution, sample_size, num_samples):
    sample_means = np.zeros(num_samples)
    for idx, num in enumerate(range(num_samples)):
        sample = np.random.choice(distribution, size=sample_size, replace=True)
        sample_means[idx] = sample.mean()
    sns.distplot(sample_means, bins=80, kde=True)
    title = f'Sample Distribution n = {sample_size} and number of samples = {num_samples},\
    std error = {pop_std / num_samples}'
    print(f'mean = {sample_means.mean()}')
    plt.title(title)
```

### The number of samples drives the shape of the curve more than the sample size itself



```python
central_limit_theorem_plotter(trip_durations, 1000, 500);
```

    mean = 18.900280133333332



![png](index_files/index_67_1.png)


### Larger sample size, Fewer samples


```python
central_limit_theorem_plotter(trip_durations, 5000, 50);
```

    mean = 18.941702066666668



![png](index_files/index_69_1.png)


* What happens as we increase the sample size?
* How does the height of the distribution change? Why does it change?

### Show with exponential


```python
exponential = np.random.exponential(scale= 1, size=1000)
```


```python
plt.hist(exponential, bins=50);
```


![png](index_files/index_73_0.png)



```python
central_limit_theorem_plotter(exponential, 4000, 10000)
```

    mean = 1.0161536000655402



![png](index_files/index_74_1.png)


### Standard Error of the Mean

The standard error of the mean is the standard deviation of the sampling distribution.
The issue is that a sample is not an exact replica of the population. We need to account for that fact in order to make our estimate of the $\mu$ value possible. Let's break it down:

**Population sigma** <br/>

$\large\sigma _{x} = \frac{\sigma }{\sqrt{n}}$

* $ \sigma _{x}$ = standard error of $\bar{x} $
* $ \sigma $ = standard deviation of population

**What if we do not know the population sigma?**<br>
If we do not know the population standard deviation, we can approximate it by using the sample standard deviation.

$\large\sigma _{x} ≈ \frac{s}{\sqrt{n}}$

* s = sample standard deviation

**Sample size impact on standard error of mean**<br>

How should sample size influence standard error of the mean?

It will get *smaller* as sample size *increases*

![error](./img/diminishing_error.png)  
Important implication: The Standard Error of the mean remains the same as long as the population standard deviation is known and sample size remains the same.

