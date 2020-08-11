# ML_Uni_Col_2020

Lets start this exciting course.
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

#### Nierest Neigbhours

Threeforld Split for Hyper-Parameter

````
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)
val_scores = []
neighbors = np.arange(1, 15, 2)
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    val_scores.append(knn.score(X_val, y_val))
print(f"best validation score: {np.max(val_scores):.3}")
best_n_neighbors = neighbors[np.argmax(val_scores)]
print("best n_neighbors:", best_n_neighbors)
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_trainval, y_trainval)
print(f"test-set score: {knn.score(X_test, y_test):.3f}")

````

The more general approach which is more roubust is Cross-validation

````
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y)
cross_val_scores = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X_train, y_train, cv=10)
    cross_val_scores.append(np.mean(scores))
print(f"best cross-validation score: {np.max(cross_val_scores):.3}")
best_n_neighbors = neighbors[np.argmax(cross_val_scores)]
print(f"best n_neighbors: {best_n_neighbors}")
knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)
print(f"test-set score: {knn.score(X_test, y_test):.3f}")
````

#### GridSearchCV
````
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
param_grid = {'n_neighbors':  np.arange(1, 30, 2)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=10,
                   return_train_score=True)
grid.fit(X_train, y_train)
print(f"best mean cross-validation score: {grid.best_score_}")
print(f"best parameters: {grid.best_params_}")
print(f"test-set score: {grid.score(X_test, y_test):.3f}")
````

#### GridSearchCV Results (this can be used to plot the graphs)

````
import pandas as pd
results = pd.DataFrame(grid.cv_results_)
results.columns
````

````
results.params
````

#### Cross-Validation Strategies

- KFold Cross-Validation: make sure to SUFFLE the data if not time series
- StratifiedKFold : Used for imbalanced data , Ensure relative class frequencies in each fold reflect relative class 
frequencies on the whole dataset

Importance of Stratification

````
y.value_counts()
0    60
1    40

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.dummy import DummyClassifier
dc = DummyClassifier('most_frequent')
skf = StratifiedKFold(n_splits=5, shuffle=True)
res = cross_val_score(dc, X, y, cv=skf)
np.mean(res), res.std()
(0.6, 0.0)

kf = KFold(n_splits=5, shuffle=True)
res = cross_val_score(dc, X, y, cv=kf)
np.mean(res), res.std()
(0.6, 0.063)

````

#### Better: ShuffleSplit (also known as Monte Carlo)
Repeatedly sample a test set with replacement

#### Even Better: RepeatedKFold.
Apply KFold or StratifiedKFold multiple times with shuffled data.

### Defaults in scikit-learn
- 5-fold in 0.22 (used to be 3 fold)
- For classification cross-validation is stratified
- train_test_split has stratify option: train_test_split(X, y, stratify=y)
- No shuffle by default!

## Cross-Validation with non-iid data

Grouped Data
Assume have data (medical, product, user...) from 5 cities
- New York, San Francisco, Los Angeles, Chicago, Houston.
We can assume data within a city is more correlated then between cities.

Usage Scenarios
- Assume all future users will be in one of these cities: i.i.d.
- Assume we want to generalize to predict for a new city: not i.i.d.

####  Correlations in time (and/or space)

TimeSeriesSplit(5, max_train_size=20)

````
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, RepeatedStratifiedKFold
kfold = KFold(n_splits=5)
skfold = StratifiedKFold(n_splits=5, shuffle=True)
ss = ShuffleSplit(n_splits=20, train_size=.4, test_size=.3)
rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

print("KFold:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=kfold))

print("StratifiedKFold:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=skfold))

print("ShuffleSplit:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=ss))

print("RepeatedStratifiedKFold:")
print(cross_val_score(KNeighborsClassifier(), X, y, cv=rs))

KFold:
[0.93 0.96 0.96 0.98 0.96]
StratifiedKFold:
[0.98 0.96 0.96 0.97 0.96]
ShuffleSplit:
[0.98 0.96 0.96 0.98 0.94 0.96 0.95 0.98 0.97 0.92 0.94 0.97 0.95 0.92
 0.98 0.98 0.97 0.94 0.97 0.95]
RepeatedStratifiedKFold:
[0.99 0.96 0.97 0.97 0.95 0.98 0.97 0.98 0.97 0.96 0.97 0.99 0.94 0.96
 0.96 0.98 0.97 0.96 0.96 0.97 0.97 0.96 0.96 0.96 0.98 0.96 0.97 0.97
 0.97 0.96 0.96 0.95 0.96 0.99 0.98 0.93 0.96 0.98 0.98 0.96 0.96 0.95
 0.97 0.97 0.96 0.97 0.97 0.97 0.96 0.96]
````

####  cross_validate function

````
from sklearn.model_selection import cross_validate
res = cross_validate(KNeighborsClassifier(), X, y, return_train_score=True,
                     scoring=["accuracy", "roc_auc"])
res_df = pd.DataFrame(res)
````

gives you more parameters wr cross_validate_score

````
fit_time    score_time    test_accuracy    test_roc_auc    train_accuracy    train_roc_auc
0.000839    0.010204    0.965217        0.996609        0.980176        0.997654
0.000870    0.014424    0.956522        0.983689        0.975771        0.998650
0.000603    0.009298    0.982301        0.999329        0.971491        0.996977
0.000698    0.006670    0.955752        0.984071        0.978070        0.997820
0.000611    0.006559    0.964602        0.994634        0.978070        0.998026
````

# Lecture 4: Preprocessing and Feature Transformations

## Scaling

- StandartScaler (mean and standart deviation)
- MinMaxScaler (for features that have clear boundaries) 
- RobustScaler (median, good if you have outliers in the data)
- Normalizer (each row devided by euqleadian distance, most used in NLP)

### Sparse data
only scale do not center

###  Standard Scaler Example
````
from sklearn.linear_model import Ridge
# Back to King Country house prices
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
ridge = Ridge().fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
ridge.score(X_test_scaled, y_test)
````
- Never call fit on test dataset
- KNN is sensitive to scaling

## Pipelines

````
from sklearn.linear_model import Ridge
X, y = df, target
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
ridge = Ridge().fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
ridge.score(X_test_scaled, y_test)
0.684

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
0.684
````

- you can put the pipeline in cross-validation

#### Pipeline and GridSearchCV
````
from sklearn.model_selection import GridSearchCV
knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
param_grid = {'kneighborsregressor__n_neighbors': range(1, 10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))
````
#### Going wild with Pipelines
````
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=0)
from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge())
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid=param_grid,
                    n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)
````

#### Going wildest with Pipelines
you can use grid in param grid and make algorithm to choose which regressor and which scaler works best
````
from sklearn.tree import DecisionTreeRegressor
pipe = Pipeline([('scaler', StandardScaler()),
                 ('regressor', Ridge())])
# check out searchgrid for more convenience
param_grid = [{'regressor': [DecisionTreeRegressor()],
               'regressor__max_depth': [2, 3, 4],
               'scaler': ['passthrough']},
              {'regressor': [Ridge()],
               'regressor__alpha': [0.1, 1],
               'scaler': [StandardScaler(), MinMaxScaler(),
                          'passthrough']}
             ]
grid = GridSearchCV(pipe, param_grid)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)
````

## Categorical Variables

#### OneHotEncoder + ColumnTransformer
you can use sk leanr to make one hot encoding of the columns which are categorical
basicly assuming that everything which is string time in columns are categorical and 
the others like integers are continious variables
That is realised in a sort of pipeline
- Column Transformer
````
categorical = df.dtypes == object
preprocess = make_column_transformer(
    (StandardScaler(), ~categorical),
    (OneHotEncoder(), categorical))
model = make_pipeline(preprocess, LogisticRegression())
````

#### Dummy variables and colinearity
- One-hot is redundant (last one is 1 – sum of others)
- Can introduce co-linearity
- Can drop one
- Choice which one matters for penalized models
- Keeping all can make the model more interpretable

````
X = data.frame.drop(['date', 'price', 'zipcode'], axis=1)
scores = cross_val_score(Ridge(), X, target)
np.mean(scores)

0.69

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
X = data.frame.drop(['date', 'price'], axis=1)
ct = make_column_transformer((OneHotEncoder(), ['zipcode']), remainder='passthrough')
pipe_ohe = make_pipeline(ct, Ridge())
scores = cross_val_score(pipe_ohe, X, target)
np.mean(scores)

0.52

from category_encoders import TargetEncoder
X = data.frame.drop(['date', 'price'], axis=1)
pipe_target = make_pipeline(TargetEncoder(cols='zipcode'), Ridge())
scores = cross_val_score(pipe_target, X, target)
np.mean(scores)

0.78

````