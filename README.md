# machine-predicted-lithology
FORCE Competition :  Create a machine learning model that has the highest accuracy in prediction lithology from a suite of wireline logs

The provided datasets (train.csv, test.csv) contain well logs and interpreted lithofacies.
This competition is well suited for people with some background in geosciences. I don't have any background in geosciences so my approach is fully data driven.

To have a full overview of the problem and data, please visit the competition website at https://xeek.ai/challenges/force-well-logs/overview

## REPOSITORY FILES DESCRIPTION
 * data_analysis.ipynb : details on some analysis and vizualisation I did on the data.
 * data_processing.ipynb : details on how data have been processed (imputation, scaling and encoding)
 * classification.ipynb: details on how I trained my classifier.

## Data Analysis (data_analysis.ipynb)
First of all, we started by looking at the provided data to get a better understanding and extract some insights on how features are linked to target (lithofacies labels). All detailed steps are available inside "data_analysis.ipynb" notebook.

Some of the insights we got/verify are:

* There are some spikes in data. They can be due to data acquisition errors or rock layers transition. We will threshold those values during data processing since we won't try to capture layers transition.
* Figure below shows that there are a large amount of null values. Some logs like ROPA, RMIC, DTS and SGR even have more than 80% of null values. Therefore one of the main challenges will be to find the most discriminative logs for our classification task. Also we should find the best way to fill those empty values.
![alt text](images/null_values_per_log.png)
* The only log with no missing value is GR
*  Our targets (lithofacies) have the similar amount of null values except of 88000 and 99000 with more than 60% of null data.
![alt text](images/null_values_per_target.png)
* Logs correlation
![alt text](images/logs_correlation.png)
Correlation matrix above shows that many logs are higly correlated (RHOB, NPHI, DTC, DTS,...). This can be very useful to fill null values.
* Also, some logs are very discriminative for lithofacies classification:
    * RSHA, RMED,RDEP bring out very clearly lithofacies 86000, 88000 and 93000. 
    * Same thing for BS and 99000 
    * Same thing for DTS,RXO and 86000. Thus we will make sure to keep those logs during modelisation.

## Data processing (data_processing.ipynb)
Many processing steps have been applied to our data
* **Data splitting**: We split our data in 3 sets train/val/test. The splitting is done per well to make sure samples of the same well fall within the same set. This way, our training is well-independent and thus we can expect a better generalization during testing and production. We keep 2 wells for validation and 2 wells for test set. The remaning wells are used for training the classifier.
* **Features selection**: There is a lot of features in our dataset. According to our previous data analysis, we realize that some of them are not quite useful for our classification task and others like SGR or RMIC have more than 80% of null values. Therefore we decided to remove those features from our data. We've also trained a RandomForestClassifier to get a sense about the most "important" features. After that, we decided to keep the following logs: 'RMED','RDEP','RHOB','GR','NPHI','DTC','RXO','RSHA',"X_LOC","Y_LOC","Z_LOC",'DEPTH_MD', 'GROUP' and 'FORMATION'.
* **Processing pipeline**: This pipeline involves data imputation, numeric features scaling and categorical features encoding.  To impute numeric features, we used an iterative imputer (from sklearn) which iteratively predict features null values using others features. This is done with a Bayesian approach. For more information on this method, please visit : https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html.
Finally , to impute GROUP and FORMATION, we trained 2 classifiers using numeric features as input. We think this method yields a better result than just filling those features with the mode. All this processing has been trained on our train set only. We only applied it on val/test sets to make sure we are not leaking information before the training phase.
* We saved the processed data and the processing pipeline for the next step. More details on this processing step are available in data_processing.ipynb 

## Modelisation (classification.ipynb)
After cleaning the data, we tried many type of classification models:
- XGBOOST classifier: Very efficient Gradient Boosting Model. 
- Catboost : Also a GBM. One of the main difference with XGboost is that it handles categorical features and is easy to train on GPU
- Deep Learning Sequence to sequence model using LSTM. Since we know that there is a structure in  Lithofacies alignment, We think that Sequential models like LSTM will be able to capture that structure and yield better results than classifier applied independently to each sample point.

For this competition, we only had time to push results yielded by Catboost as we arrived when the competition was about to close. Therefore we did a simle grid search to find a classifier with acceptable results.

- First we trained a baseline model. It gives us 
![alt text](images/baseline_metrics.png)
Above plots show the model is overfitting since it works well on train/val but not on test set.

- Thus we tried a grid search on some hyperparameters to find a better model. 
![alt text](images/gs_metrics.png)
Above plots show that our classifier is generalizing more (better test scores). This is done after some grid search iterations. Better models can be found after longer hyperparameters tuning using fancier methods like Bayesian search.

- Features importance
![alt text](images/features_importance.png)
Our current best model features importance show that GR, GROUP and FORMATION are the most used features for trees construction. Therefore it was a good idea to fill GROUP and FEATURE with classifiers since there are very important for this task.

- Confusion matrix
![alt text](images/confusion_matrix.png)
Our current model is not doing very well in terms of Recall and precision. Many lithofacies are predicted as 65000(most frequent class in our dataset) This is due to the fact that we have a highly unbalanced classes. There are many ways we can try to solve this issue.


## HOW CAN WE IMPROVE THOSE RESULTS?
- Firstly we should deal with unbalanced classes problem. This can be done by weighting samples according to classes frequency. If we give higher weights to rare classes, maybe the model will be able to classifier them better. Also we can try upsampling/downsampling method to train our model on a more balanced dataset.
- Then we should improve hyperparameters tuning. This can be done by running more iterations on our current grid search or by using bayesian search which usually gives better results.
- Also, to improve our results, we can try other type of models. For example we can use LSTM architecture to classify all samples of the same well at the same time. This way our model will use information of surroundings to classify a sample. Also this type of model could be able to capture lithofacies structure underground. This is thus a promising way to go.

## COMPETITION RESULTS
- With my first approach (data filling with iterative imputer and Catboost)I was ranked  55/329 on the public leaderboard. 
https://xeek.ai/challenges/force-well-logs/leaderboard
![alt text](images/rank.PNG)
- The winner on the private (final) leaderboard was 24/329 on the public leaderboard. It means that top scorers on the public leaderboard were overfitting a lot on data used for the public leaderboard.
- I didn't get a chance to run my model on the private leaderboard dataset. So we will never know what my final ranking would have been ;). 
- Even with a simple modelisation (Catboost + quick grid search) and most of my time working on data vizualisation/analysis/processing, I managed to get a good ranking and I had a lot of fun working on this competition.