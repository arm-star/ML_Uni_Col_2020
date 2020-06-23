# ML_Uni_Col_2020

Lets stat this exciting course.
https://www.youtube.com/watch?v=d79mzijMAw0&list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM
Course materials at: https://www.cs.columbia.edu/~amueller/comsw4995s20/schedule/

# Lecture 1: Introduction

### Types of Machine Learning
#### Supervised
Collected samples are i.i.d. independent identity distributed.
For example in case of time series the samples are not i.i.d.

Examples of Supervised Learning

> spam detection, medical diagnosis, ad click prediction

#### Unsupervised
Goal is to learn about the distribution.

Examples of Unsupervised Learning

>Outlier detection, clustering, dimensionality reduction

#### Reinforcement
Learn to drive, play games.

# Lecture 2: Visualization and Matplotlib

twinx, twiny

````
ax1 = plt.gca()
line1, = ax1.plot(years, phds)
ax2 = ax1.twinx()
line2, = ax2.plot(years, revenue, c='r')
ax1.set_ylabel("Math PhDs awarded")
ax2.set_ylabel("revenue by arcades")
ax2.legend((line1, line2),
           ("math PhDs awarded", "revenue by arcades"))
````       









