# machine-predicted-lithology
FORCE Competition :  Create a machine learning model that has the highest accuracy in prediction lithology from a suite of wireline logs

The provided datasets (train.csv, test.csv) contain well logs and interpreted lithofacies.
This competition is well suited for people with some background in geosciences. I don't have any background in geosciences so my approach is fully data driven.

To have a full overview of the problem and data, please visit the competition website at https://xeek.ai/challenges/force-well-logs/overview

## REPOSITORY FILES DESCRIPTION
 * data_analysis.ipynb : details on some analysis and vizualisation I did on the data.

## Data Analysis (data_analysis.ipynb)
First of all we started by looking at the provided data to get a better understanding and extract some insights on how features are linked to target (lithofacies labels). All detailed steps are available inside "data_analysis.ipynb" notebook.

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
