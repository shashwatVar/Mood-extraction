# Mood-extraction
Mood-extraction is meant for the prediction of mood of the mobile user based on the phone app usage data and statistics. For the data RMF approach is used for feature definition.
The Recency-Frequency-Monetary value segmentation has been around for a while now and provides a pretty simple but effective way to segment.  
An RFM model can be used in conjunction with certain predictive models to gain even further insight into user behavior.

the approach used can be summarized as follows: 
* Calculate R, F and M parameters
* Apply k-means clustering algorithm on these parameters to group similar content.
* Apply classification algorithms such as Logistic Regression and Decision Trees to predict future customer behavior.
* Finally apply recommendation algorithms such as collaborative or content based filtering and Association Rules

here random forest classification is used with the K-Means clustering at n = 2, at which silhouette coefficient was the highest


![alt text][score]

[score]: https://raw.githubusercontent.com/geforce6t/Mood-extraction/master/models/scores.png "score"

2 cluster: 
![alt text][2cluster]

[2cluster]: https://raw.githubusercontent.com/geforce6t/Mood-extraction/master/models/inference1.png "2 cluster mean graph"

3 cluster: 
![alt text][3cluster]

[3cluster]: https://raw.githubusercontent.com/geforce6t/Mood-extraction/master/models/inference2.png "3 cluster mean graph"
