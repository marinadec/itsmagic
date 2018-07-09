# itsmagic

### Intro

This repository has two main notebooks: preprocess_data.ipynb, train.ipynb.

The first one contains data preprocessing, the second one contains the models I train. 

Running the code can be done simply by running the whole notebook. All the models, data preprocessing and the outputs are listed with all the comments as we go step by step in each notebook. Data notebook generates some tables which are used for training in the train notebook. No other specific remarks required.

### ML algorithms used
- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost
- SVM
- Ensemble of Logistic Regression, Decision Tree, XGBoost
- MLP (keras)
- RNN with LSTM cell (tensorflow)


### Libraries used

Preprocessing:
- sklearn
- numpy
- pandas

Training:
- sklearn
- xgboost
- keras
- tensorflow

Displaying:
- tqdm
- matplotlib (pyplot)
- seaborn

Other:
- sys, re, os
- pickle
- bunch

### Data preprocessing

Data preprocessing includes some magic on the data side such as: 
- aggregating a client with his orders to one row data to be fed to standart ML algorithm
- checking some data statistics on different columns (histograms, average, quantile)
- finding outliers
- creating dummy columns for some categorical columns (which has a reasonable number of categories - not 13k as restaurants)
- investigating columns correlation and dropping some of the highly correlated ones
- data normalization

The notebook generates two file: 

- Xy_norm.feather, normalized table with aggregated orders per client, to be fed to standart ML algo
- order_data_lbl.feather, table with all client orders, a sequence of client orders to be fed to RNN algorithm (Tensorflow) 

#### Conclusions

- We have enough of clients to learn: 245455.
- Column "customer_order_rank" is mostly missing so it's removed from further learning.
- Aggregation of orders for a client can be mean, sum, std, quantile, count. E.g. for some client and his 10 orders there will be one row generated with number different payment methods he/she used in total, mean order cost, number of failed orders, sum of vouchers.
- Dummy columns are created for columns (payment_id, platform_id, transmission_id) as they have a reasonable number of categories (5, 14, 10) each. For columns (restaurant_id, city_id) there are (13569, 3749) categories respectively. So to avoid plenty of extra features I don't create dummies for these.
- Column value distributions are checked.
- The outliers for some columns are present, e.g. for column "orders_num" there is a person who has ~400 orders. I tried to train ML algorithm discussed below without some obvious outliers for each column but it didn't bring any improvement for the models, so I'll just leave it.
- Removing highly correlated columns helps improves the models.
- Data is a bit imbalanced (rate of neg/pos labels=0.25) but this is not crucial. Especially, it's not a problem for the models like Neural net (problematic imbalanced data is if rate=0.01 for instance) or ... . So I will not apply any balancing techniques here.
- First mentioned table above is normalized. I don't normalized the second one which is fed to RNN as the values are not significantly high and also because there are many dummy columns, so whatever at this step.


### Training

There is no need to precisely look through the commiting history to find which models I trained as all of them are listed in the notebooks with some metrics no matter if it's a good model or not, thus we can compare the models via scrolling through one notebook. The results are discussed below.

Data is split to train and validation sets. 

Two baselines are computed: zero output and random output.








### Possible future modifications





