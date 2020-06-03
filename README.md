
# Sampling and the Central Limit Theorem

![sample](https://media.giphy.com/media/OsOP6zRwxrnji/giphy.gif)

# Agenda 

1. Differentiate terms: discriptive/inferential, statistics population/sample, paramater/statistic, sample distribution/sampling distribution
2. Define and calculate standard error
3. Use Numpy to randomly sample a distribution
4. Describe the central limit theorem and connect it to our knowledge of distributions and sampling.
5. Capital Bikeshare Example

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

The first way of sampling is considered a convenience sample.
You are going about collection in a non-random manner

# Sample Conditions

1. The sampled observations must be independent
    - The sampling method must be random  


2. Sample size distribution:
    - The more skewed the sample the larger samples we need. 
    - n > 30 is considered a large enough sample unless there is extreme skew




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




![png](index_files/index_15_1.png)


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




![png](index_files/index_18_1.png)


We can then calculate the sample statistics:


```python
print(f'Sample mean: {sample.mean()}')
print(f'Sample standard deviation: {sample.std()}')
print(f'Sample median: {np.median(sample)}')
```

    Sample mean: 115.4907660259872
    Sample standard deviation: 8.634151702239585
    Sample median: 115.4998911419724


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


![png](index_files/index_24_0.png)


An interesting property of this sampling distribution:
    
As we continue to sample, the mean of the sampling distribution gets closer and closer to the population mean.

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



```python
def standard_error(distribution, largest_sample_size, population_std=None):
    
    '''
    Calculate the standard errors for a range of sample sizes
    to demonstrate how standard error decreases when sample 
    size increases.
    '''
 
    std_errors = {}
    
    for sample_size in range(50,largest_sample_size+1):
        sample = np.random.choice(distribution, size=sample_size, replace=True)
        # Standard error with sample distribution standard deviation 
        # in place of population
        if population_std == None:
            std_err = np.std(sample)/np.sqrt(sample_size)
            std_errors[sample_size] = std_err
        
        else:
            std_err = population_std/np.sqrt(sample_size)
            std_errors[sample_size] = std_err
        
    return std_errors
    
```


```python
std_errors = standard_error(sys_pop, 1000)

fig, ax = plt.subplots()

sns.scatterplot(list(std_errors.keys()), list(std_errors.values()))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a242ffa20>




![png](index_files/index_31_1.png)



```python
std_errors = standard_error(sys_pop, 1000, population_std=114)

fig, ax = plt.subplots()

sns.scatterplot(list(std_errors.keys()), list(std_errors.values()))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a23d56908>




![png](index_files/index_32_1.png)


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
         operating right now.  This group represents the {var_5} of actively 
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
             operating right now.  This group represents the parameter of actively 
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

    group 1 ['Adam' 'Jason' 'Matt' 'Dann']
    group 2 ['Maximilian' 'Johnhoy' 'Chum' 'Jacob']
    group 3 ['Leana' 'Luluva' 'Amanda' 'Karim']


## Group 1:

A bowler on the PBA rolls a strike 60% of the time. The population strikes of all games ever bowled is stored in in the population variable below.



```python
population = np.random.binomial(12, .6, 10000)
fig, ax = plt.subplots()
ax.bar(range(0,12), np.unique(population, return_counts=True)[1])
ax.set_title('Strikes Per Game')
```




    Text(0.5, 1.0, 'Strikes Per Game')




![png](index_files/index_42_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 20)
```




    (array([  1.,   0.,   1.,   1.,   5.,  28.,  34.,  68.,  83., 100., 115.,
            167., 111., 121.,  67.,  51.,  24.,  12.,   6.,   5.]),
     array([6.26 , 6.343, 6.426, 6.509, 6.592, 6.675, 6.758, 6.841, 6.924,
            7.007, 7.09 , 7.173, 7.256, 7.339, 7.422, 7.505, 7.588, 7.671,
            7.754, 7.837, 7.92 ]),
     <a list of 20 Patch objects>)




![png](index_files/index_43_1.png)


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




![png](index_files/index_45_1.png)



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




![png](index_files/index_46_1.png)


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



![png](index_files/index_48_1.png)



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




![png](index_files/index_49_1.png)


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
Capital bike share is trying to figure out their pricing for members versus non-members. The first step in their analysis is to see if members vs non-members ride for different amounts of time per ride.

Let's head over [here](https://s3.amazonaws.com/capitalbikeshare-data/index.html) for some DC bike data!


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
%matplotlib inline

```


```python
! wget 'https://s3.amazonaws.com/capitalbikeshare-data/201905-capitalbikeshare-tripdata.zip' -O temp.zip
! unzip temp.zip
```

    --2020-06-03 08:43:31--  https://s3.amazonaws.com/capitalbikeshare-data/201905-capitalbikeshare-tripdata.zip
    Resolving s3.amazonaws.com... 52.216.160.245
    Connecting to s3.amazonaws.com|52.216.160.245|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 7837266 (7.5M) [application/zip]
    Saving to: 'temp.zip'
    
    temp.zip             50%[=========>          ]   3.81M  1.80MB/s               ^C
    Archive:  temp.zip
      End-of-central-directory signature not found.  Either this file is not
      a zipfile, or it constitutes one disk of a multi-part archive.  In the
      latter case the central directory and zipfile comment will be found on
      the last disk(s) of this archive.
    unzip:  cannot find zipfile directory in one of temp.zip or
            temp.zip.zip, and cannot find temp.zip.ZIP, period.



```python
df = pd.read_csv('201905-capitalbikeshare-tripdata.csv')
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
      <th>337699</th>
      <td>434</td>
      <td>2019-05-31 23:59:15</td>
      <td>2019-06-01 00:06:30</td>
      <td>31281</td>
      <td>8th &amp; O St NW</td>
      <td>31627</td>
      <td>3rd &amp; M St NE</td>
      <td>W23767</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>337700</th>
      <td>227</td>
      <td>2019-05-31 23:59:45</td>
      <td>2019-06-01 00:03:32</td>
      <td>31201</td>
      <td>15th &amp; P St NW</td>
      <td>31229</td>
      <td>New Hampshire Ave &amp; T St NW</td>
      <td>W23691</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>337701</th>
      <td>1638</td>
      <td>2019-05-31 23:59:45</td>
      <td>2019-06-01 00:27:03</td>
      <td>31261</td>
      <td>21st St &amp; Constitution Ave NW</td>
      <td>31247</td>
      <td>Jefferson Dr &amp; 14th St SW</td>
      <td>W20810</td>
      <td>Member</td>
    </tr>
    <tr>
      <th>337702</th>
      <td>1621</td>
      <td>2019-05-31 23:59:51</td>
      <td>2019-06-01 00:26:53</td>
      <td>31261</td>
      <td>21st St &amp; Constitution Ave NW</td>
      <td>31247</td>
      <td>Jefferson Dr &amp; 14th St SW</td>
      <td>W21714</td>
      <td>Casual</td>
    </tr>
    <tr>
      <th>337703</th>
      <td>373</td>
      <td>2019-05-31 23:59:51</td>
      <td>2019-06-01 00:06:04</td>
      <td>31281</td>
      <td>8th &amp; O St NW</td>
      <td>31201</td>
      <td>15th &amp; P St NW</td>
      <td>W21769</td>
      <td>Member</td>
    </tr>
  </tbody>
</table>
</div>



### Let's take a look at the shape of our dataset


```python
import seaborn as sns
from scipy import stats

fig, ax = plt.subplots(2,1, figsize=(10,10))
sns.distplot(df.Duration, bins = 20, ax=ax[0])
sns.boxplot(df.Duration, ax=ax[1])

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a35f11cc0>




![png](index_files/index_62_1.png)


The shape is difficult to see because of the outliers. Let's remove some to get a better sense of the shape


```python
pop_no_fliers = df[np.abs(stats.zscore(df.Duration) > 3)]

fig, ax = plt.subplots()
sns.distplot(pop_no_fliers.Duration)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a267166d8>




![png](index_files/index_64_1.png)



```python
member_df = df[df['Member type'] == 'Member']
casual_df = df[df['Member type'] == 'Casual']
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-2ed6639d768d> in <module>
    ----> 1 member_df = df[df['Member type'] == 'Member']
          2 casual_df = df[df['Member type'] == 'Casual']


    NameError: name 'df' is not defined



```python
plt.boxplot(member_df['Duration']);

```


![png](index_files/index_66_0.png)



```python
from scipy import stats
print(member_df.shape)
sum(stats.zscore(member_df.Duration)>3)
```

    (286079, 9)





    1899




```python
member_df_nofliers = member_df[np.abs(stats.zscore(member_df.Duration)) < 3]
member_df_nofliers.shape
```




    (284180, 9)




```python
import seaborn as sns
fig, ax = plt.subplots()
sns.distplot(member_df_nofliers.Duration, bins = 20, ax=ax);
```


![png](index_files/index_69_0.png)



```python
fig, ax = plt.subplots()
sns.boxplot(casual_df.Duration)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2be4b630>




![png](index_files/index_70_1.png)



```python
casual_df_nofliers = casual_df[np.abs(stats.zscore(casual_df.Duration)) < 3]
```


```python
fig, ax = plt.subplots()
sns.distplot(casual_df_nofliers.Duration, bins=20, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a264c89b0>




![png](index_files/index_72_1.png)



```python
fig, ax = plt.subplots()
sns.distplot(member_df_nofliers.Duration, bins = 20, ax=ax, color='blue');
sns.distplot(casual_df_nofliers.Duration, bins=20, ax=ax, color='green')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2d81c5c0>




![png](index_files/index_73_1.png)


#### Get population statistics


```python
df.median()
```




    Duration                  724.0
    Start station number    31266.0
    End station number      31264.0
    dtype: float64



Let's treat the whole dataset as our population.


```python
pop_mean = df.Duration.mean()
pop_std = df.Duration.std()
print(f'pop_mean is {pop_mean} \npop_std is {pop_std}')
```

    pop_mean is 1138.3041865065263 
    pop_std is 2282.5139701034595



```python
def one_sample_mean(population):
    sample = np.random.choice(population, size=200, replace=True)
    return sample.mean()
```


```python
one_sample_mean(df.Duration)
```




    1237.33



### When we take multiple samples from the distribution,and plot the means of each sample, the shape of the curve shifts


```python
d = [one_sample_mean(df.Duration) for i in range(1000)]
plt.hist(d, bins=50)

```




    (array([ 3.,  5.,  8., 26., 25., 50., 50., 75., 72., 68., 68., 73., 58.,
            60., 48., 55., 33., 20., 31., 27., 17., 12., 16., 18., 12.,  9.,
            10.,  4.,  4.,  7., 13.,  2.,  6.,  0.,  2.,  1.,  0.,  1.,  4.,
             0.,  0.,  2.,  1.,  0.,  0.,  0.,  0.,  0.,  2.,  2.]),
     array([ 829.1   ,  852.8163,  876.5326,  900.2489,  923.9652,  947.6815,
             971.3978,  995.1141, 1018.8304, 1042.5467, 1066.263 , 1089.9793,
            1113.6956, 1137.4119, 1161.1282, 1184.8445, 1208.5608, 1232.2771,
            1255.9934, 1279.7097, 1303.426 , 1327.1423, 1350.8586, 1374.5749,
            1398.2912, 1422.0075, 1445.7238, 1469.4401, 1493.1564, 1516.8727,
            1540.589 , 1564.3053, 1588.0216, 1611.7379, 1635.4542, 1659.1705,
            1682.8868, 1706.6031, 1730.3194, 1754.0357, 1777.752 , 1801.4683,
            1825.1846, 1848.9009, 1872.6172, 1896.3335, 1920.0498, 1943.7661,
            1967.4824, 1991.1987, 2014.915 ]),
     <a list of 50 Patch objects>)




![png](index_files/index_81_1.png)



```python
import seaborn as sns

def central_limit_theorem_plotter(distribution, sample_size, num_samples, color='blue'):
    sample_means = np.zeros(num_samples)
    for idx, num in enumerate(range(num_samples)):
        sample = np.random.choice(distribution, size=sample_size, replace=True)
        sample_means[idx] = sample.mean()
    sns.distplot(sample_means, bins=80, kde=True,  color=color)
    title = f'Sample Distribution n = {sample_size} and number of samples = {num_samples},\
    std error = {pop_std / num_samples}'
    print(f'mean = {sample_means.mean()}')
    plt.title(title)
```

### The number of samples drives the shape of the curve more than the sample size itself



```python
central_limit_theorem_plotter(df.Duration, 1000, 500);
```

    mean = 1136.515084



![png](index_files/index_84_1.png)


### Larger sample size, Fewer samples


```python
central_limit_theorem_plotter(df.Duration, 5000, 50);
```

    mean = 1132.6324960000002



![png](index_files/index_86_1.png)


* What happens as we increase the sample size?
* How does the height of the distribution change? Why does it change?


```python
central_limit_theorem_plotter(member_df.Duration, 1000, 500, 'blue')
central_limit_theorem_plotter(casual_df.Duration, 1000, 500, 'green')
```

    mean = 885.1660820000001
    mean = 2523.241936



![png](index_files/index_88_1.png)

