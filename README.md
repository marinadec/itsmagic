# itsmagic

### Intro

This repository has two main notebooks: preprocess_data.ipynb, train.ipynb.

The first one contains data preprocessing, the second one contains the models trained. 

Running the code can be done simply by running the whole notebook. All the models, data preprocessing and the outputs are listed with all the comments as we go step by step in each notebook. Data notebook generates some tables which are used for training in the train notebook. No other specific remarks required.

I additionally train MLP model on Keras and Sequence RNN model on Tensorflow.

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
- skip poorly represented columns (e.g. "customer_order_rank")
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

There is no need to precisely look through the commiting history to find which models I trained as all of them are listed in the notebooks with some metrics no matter if it's a good model or not, thus we can compare the models via scrolling through one notebook. The results are discussed below and in the notebooks themselves.

Data is split to train and validation sets. 

Two baselines are computed: zero output and random output.

The models are compared with the baselines and with each other by metrics such as accuracy, AUC, APS, precision, recall, f1 score. I also check the final output probabilities distribution. For every model I vary it's parameters e.g. learning rate, tree depth, number of neural net layers, hidden vector size, threshold for probabilities to be picked etc. Sometimes I try reach overfitting to justify the model, but in general I try to avoid it by changing the model or picking the best epoch (in case of MLP and RNN) as there shouldn't be any overfitting.

See the notebook training.ipynb for the metrics and outputs, supported with the comments as we go.

For RNN model I picked 2 layers with LSTM cell. For a single client a sequence is his orders in time, so recurrent network accumulates client's orders one by one and the last output is kept, then passed to a dense layer with sigmoid. Metrics are loss and accuracy - accumulatively computed batch by batch using tf.metrics.

So far I would say that the best models are XGBoost, Ensemble and MLP. Some of the metrics are the best for one of them. Sequence RNN model might seem a bit useless (is it?) as the majority of people has one order only, however, it shows the result similar to other models.

Follow to the notebook for the implementation details.

Note 1: SVM runs too long, I'll just leave the code.

Note 2: Carefully compare RNN model to others as validatino set is the same for the models except RNN.

Note 3: For zero baseline there's no presicion, recall, f1. For RNN I use just accuracy and AUC and don't add extra accumularive metrics to the model to save the space in the code, as it would look a bit messy for a potential viewer.

### The metrics on validation set

Note: 'thr' stands for 'threshold'.

|           | Zero baseline| Random baseline| Logistic Reg.| Random forest| Decision tree| XGBoost|Ensemble| MLP    | RNN    | 
| --------- | ------------ | -------------- | ------------ | ------------ | ------------ | ------ | ------ | ------ | ------ |
| Accuracy  | .77          | .72 (thr=.9)   | .808         | .806         | .808         |**.813**|**.812**|**.814**|**.81** |
| AUC       | .5           | .5             | .753         | .76          | .753         |**.768**|**.767**|**.768**|**.766**|
| APS       | .23          | .23            | .543         | .538         | .513         |**.562**|**.561**|**.562**| -      |
| Precision | -            | .23            | **.7**       | **.694**     | .689         | .678   |**.7**  | .676   | -      |
| Recall.   | -            |**.5** (thr=.5) | .262         | .367         | .386         | .331   | .298   | .34    | -      |
| F1        | -            | .31 (thr=.5)   | .381         | **.437**     | .411         |**.444**| .418   |**.452**| -      |


### Possible improvements

- For MLP in Keras try to keep the categorical columns as dummies but add embedding vectors right after them (https://arxiv.org/pdf/1604.06737.pdf).
- Tune Sequence RNN model.
- Different models for customers with the different behaviour like very high activity (which is many orders).
- Use other column combination and check their correlations not to create "similar” multiple columns.
- Explore carefully the impact of the data outliers.
- Create more samples by splitting the sequence of orders to subsequences according to the order’s time (label will be 1 for all of the subsequences except the last one, the last is already defined). This also creates more positive labels to fight imbalance.
- Create more data with data augmentation techniques, like adding some noise etc.
- Apply data preprocessing (from orders to one client row) in parallel client-wise as it is slow.
- Try not/normalized data. For instance, shouldn't matter for the trees, but matter for logistic regression.
- Tune model params, e.g. with cross-validation, vary model complexity etc.
- More feature engineering.





