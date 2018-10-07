import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

#Use KMeans Clustering to cluster Universities into to two groups, Private and Public.

def rateValueCheck(currentRate):#Make sure that the rates of the data are between 0 to 100.
    if currentRate >= 100:
        return 100
    elif currentRate <= 0:
        return 0
    else:
        return currentRate

def convertPredictionIntoClusterNumber(cluster):#Convert private column into number of cluster , 1 - Private school, 0 - Public school
    if cluster=='Yes':
        return 1
    else:
        return 0

universitiesDataFrame = pd.read_csv('College_Data',index_col=0)#Read the csv file with the university name as the first column.

universitiesDataFrame['Grad.Rate'] = universitiesDataFrame['Grad.Rate'].apply(rateValueCheck)#Make sure that the Grade rates of the data are between 0 to 100.
universitiesDataFrame['TrueCluster'] = universitiesDataFrame['Private'].apply(convertPredictionIntoClusterNumber)#Convert labels into number of cluster

kMeansClustering = KMeans(n_clusters=2)# Create an instance of a K Means model with 2 clusters.
kMeansClustering.fit(universitiesDataFrame.drop('Private',axis=1))#Fit the model to the data except for the Private label.

#Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.
print(confusion_matrix(universitiesDataFrame['TrueCluster'],kMeansClustering.labels_))
print('\n')
print(classification_report(universitiesDataFrame['TrueCluster'],kMeansClustering.labels_))

# Show the true labeled data in a graph.
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=universitiesDataFrame, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
plt.title('True Data Clustering')
plt.show()



