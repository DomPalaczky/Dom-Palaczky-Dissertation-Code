from .KmeansCluster import kmeansCluster
from .SOMCluster import SelfOrganisingMap
from .visual import averageBarchart, boxplot, pltHistwithMeans, pltNormalDistwithMeans, radarClus, parallelClus
from .validateCluster import clusterSummary, featureImportanceStats, featureImportanceDT, treevis, tsneScatter2D, pcaScatter, plotFeatureImportance, plotFeatureImportanceByClass
from.timeVisuals import createTimeSeriesGraph
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler 

def runClusterAnalysis(data, saveimagepath, 
                       labelcolumn = None, labelnames = None, 
                       timeseriesdata = None, mergetimeserieson = None, timeseriesdates= None, timeseriesamounts = None,
                       savedatapath=None, savemodelpath = None, 
                       algorithm = 'SOM', validationthreshold = 0.95, maxclusters = 10, SOMParams = None):
    #create save directories if dont exist
    try:
        os.mkdir(saveimagepath)
    except:
        pass
    try:
        os.mkdir(savedatapath)
    except:
        pass
    try:
        os.mkdir(savemodelpath)
    except:
        pass
    
    #tidy and seperate data  
    columnnames = list(data.columns)
    
    if labelcolumn is not None:
        columnnames.remove(labelcolumn)
        labels = data[labelcolumn].to_list()
    else:
        labels = None
        
    if mergetimeserieson is not None: #assume merge column is to be removed from training
        columnnames.remove(mergetimeserieson)
        
    X = data[columnnames].to_numpy()
    X = MinMaxScaler().fit_transform(X)
    
    t1 = time.time()
    if algorithm.lower() == 'kmeans': #run kmeans algorithm
        km = kmeansCluster()

        inertias, kneedle = km.findKMeansClusters(X, saveimagepath) #generates elbow plot, uses k needle to calculate

        clusterLabels, centers, kmeans = km.labelKMeans(X, maxclusters)
        
        #give stats on clusters
        labelandcounts = np.unique(clusterLabels, return_counts = True)

        print('There are ' + str(max(clusterLabels)+1) + ' clusters')
        print('The size of clusters are ', np.transpose(labelandcounts))
        
        data['cluster'] = clusterLabels
        if savedatapath is not None:
            data.to_csv(savedatapath  + '/labelledDataKmeans.csv', index = False)
        print(time.time()-t1)
        
    elif algorithm.lower() == 'som': #run SOM algorithm
        SOM = SelfOrganisingMap() 
        results, best, som = SOM.train_som(X, params = SOMParams) #caluclate best SOM usin gridsearch, then visualise
        SOM.makeDistanceMap(som, saveimagepath)
        
        if labels is not None:
            SOM.makeLabelledSOM(som, X, labelnames, labels, best, title = 'category',path = saveimagepath) #if additional labels, caluclate SOM
            
        SOM.makeFrequencyMap(som,X) #visualise
        unique_winners, winnerscat, big_neurons = SOM.return_unique_winners(som, X, return_cat = True, return_big_neurons = True)
        
        SOMlabels = SOM.SOMKmeans(som, returnLabelsShaped= True, path = saveimagepath) #label using SOM
        
        clusterLabels = SOM.SOMKmeansWinners(som, X, SOMlabels) #Cluster labels using Kmeans
        
        SOM.makeLabelledSOM(som, X, list(range(max(clusterLabels)+1)), clusterLabels, best, title = 'clusters', path = saveimagepath) #visualise
        
        #give stats on clusters
        labelandcounts = np.unique(clusterLabels, return_counts = True)

        print('There are ' + str(max(clusterLabels)+1) + ' clusters')
        print('The size of clusters are ', np.transpose(labelandcounts))
        
        data['cluster'] = clusterLabels
        if savedatapath is not None:
            data.to_csv(savedatapath  + '/labelledDataSOM.csv', index = False)
        print(time.time()-t1)
            
    #validate clustering
    featureImportance, score, clas = featureImportanceDT(X,clusterLabels, returnScore = True)
    treevis(clas,X,clusterLabels,columnnames, saveimagepath)
    
    print('Decision Tree Validation score is ' + str(score[-1]))
    
    if float(score[-1]) < float(validationthreshold):
        print('Clusters are not valid')
        return
    
    clustersummarydf = clusterSummary(X,columnnames, clusterLabels)
    if savedatapath is not None:
        data.to_csv(savedatapath  + '/clustersummary.csv', index = False)
    
    stats,impval = featureImportanceStats(clustersummarydf, max(clusterLabels)+1, X)
    #print(stats)
    
    plotFeatureImportance(columnnames, featureImportance, saveimagepath)
    
    plotFeatureImportanceByClass(columnnames, featureImportance, X, clusterLabels, clas, saveimagepath)
    
    tsneScatter2D(X,clusterLabels, saveimagepath)
    pcaScatter(X,clusterLabels, saveimagepath)
    
    #Aggregate variables and visualise
    averageBarchart(X, clusterLabels, columnnames, saveimagepath, show = False)
    boxplot(X,clusterLabels, columnnames, saveimagepath)
    
    pltHistwithMeans(clustersummarydf, X, saveimagepath)
    pltNormalDistwithMeans(clustersummarydf, X, saveimagepath)
    
    
    radarClus(X, clusterLabels, columnnames, saveimagepath, show = False)
    parallelClus(X, clusterLabels, columnnames, saveimagepath, show = False)
    #if labelling problem, produce same figs
    if labels is not None:
        saveimagepathlabels = saveimagepath + 'labelled'
        
        try:
            os.mkdir(saveimagepathlabels)
        except:
            pass
        
        averageBarchart(X, labels, columnnames, saveimagepathlabels, show = False)
        boxplot(X,labels, columnnames, saveimagepathlabels)
        
        labelsummarydf = clusterSummary(X,columnnames, labels)
        pltHistwithMeans(labelsummarydf, X, saveimagepathlabels)
        pltNormalDistwithMeans(labelsummarydf, X, saveimagepathlabels)


        radarClus(X, labels, columnnames, saveimagepathlabels, show = False)
        parallelClus(X, labels, columnnames, saveimagepathlabels, show = False)
    
    
    if timeseriesdata is not None: #if time series data exists, visualsie
        timeseriesdatalabelled = pd.merge(timeseriesdata,data, on = mergetimeserieson)
        
        
        dates = np.array(timeseriesdata[timeseriesdates])
        amounts = np.array(timeseriesdata[timeseriesamounts])
        
        title = timeseriesamounts + ' by cluster'
        createTimeSeriesGraph(dates, amounts, title, np.array(clusterLabels), saveimagepath)
        
        #if data is labelled, plot that too
        if labels is not None:
            dates = np.array(timeseriesdata[timeseriesdates])
            amounts = np.array(timeseriesdata[timeseriesamounts])
            title = timeseriesamounts + ' by ' + labelcolumn
            createTimeSeriesGraph(dates, amounts, title, np.array(labels), saveimagepathlabels)