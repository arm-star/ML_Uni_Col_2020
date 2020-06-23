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
heatmaps
````
fig, ax = plt.subplots(2, 2)
im1 = ax[0, 0].imshow(arr)
ax[0, 1].imshow(arr, interpolation='bilinear')
im3 = ax[1, 0].imshow(arr, cmap='gray')
im4 = ax[1, 1].imshow(arr, cmap='bwr',
                      vmin=-1.5, vmax=1.5)
plt.colorbar(im1, ax=ax[0, 0])
plt.colorbar(im3, ax=ax[1, 0])
plt.colorbar(im4, ax=ax[1, 1])
````

plot
````
fig, ax = plt.subplots(2, 4, figsize=(10, 5))
ax[0, 0].plot(sin)
ax[0, 1].plot(range(100), sin)  # same as above
ax[0, 2].plot(np.linspace(-4, 4, 100), sin)
ax[0, 3].plot(sin[::10], 'o')
ax[1, 0].plot(sin, c='r')
ax[1, 1].plot(sin, '--')
ax[1, 2].plot(sin, lw=3)
ax[1, 3].plot(sin[::10], '--o')
plt.tight_layout() # makes stuff fit - usually works
````
hexgrids
````
plt.figure()
plt.hexbin(x, y, bins='log', extent=(100, 160, -45, -10))
plt.colorbar()
plt.axis("off")
````
- pandas plotting - convenience
- seaborn - ready-made stats plots
- bokeh - alternative to matplotlib for in-browser
- several ggplot translations / interfaces
- bqplot
- plotly
- altair (the cool new kid)
- yellowbrick (plotting for sklearn)

# Lecture 3: Introduction to Supervised Learning







