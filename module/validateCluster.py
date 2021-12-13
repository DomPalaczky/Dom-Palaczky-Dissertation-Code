import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from dtreeviz.trees import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import colorsys


def clusterSummary(X, columnnames, labels):
    
    #add cluster as index
    cn = columnnames + ['cluster']
    clusterData = pd.DataFrame(np.c_[X, labels], columns = cn)
    print(clusterData.head())
    clusterData = clusterData.set_index('cluster')
    
    #find average for each feature per cluster
    avgs = []
    
    for i in range(0,max(labels)+1):
        avg = clusterData.loc[i]
        avgs.append(list(avg.describe().loc['mean']))
    
    #save as dataframe
    avgDF = pd.DataFrame(avgs)
    avgDF.columns = columnnames
    return avgDF

def featureImportanceStats(clustersummarydf, numofclusters, X):
    
    importance = {key: [] for key in range(numofclusters)}
    impvals = []
    i=0
    for c in clustersummarydf.columns: 
    
        l = clustersummarydf[c]
        x = X[:,i]

        for j in range(len(l)):
            if l[j] > np.mean(x) + (np.std(x)): # test representativeness
                impvals.append([c,j,np.mean(x), 'high']) #column, cluster, mean value, high or low
            elif l[j] < np.mean(x) - (np.std(x)): # test representativeness
                impvals.append([c,j,np.mean(x), 'low'])
            
        out = ([ i for i in range(len(l)) if l[i] > np.mean(x) + (np.std(x)) or l[i] < np.mean(x) - (np.std(x))])

        for clus in out:
            importance[clus].append(c)
        
        i+=1
    # for key in list(importance.keys()):
    #     print('Cluster ' + str(key) + ' has important features: ' + str(importance[key]))
    
    impvals.sort(key=lambda x: x[1])
    print('Statistically')
    for v in impvals: #print important clusters
        print('Cluster ' + str(v[1]) + ' has relatively ' + str(v[3]) + ' feature ' + str(v[0]) + ' with mean value of ' + str(v[2]))
        
    
    return importance, impvals

def featureImportanceDT(X,labels, returnScore=False):
    clf = DecisionTreeClassifier(random_state=0, max_depth = 5)
    clf.fit(X, labels)
    if returnScore == True:
        score = cross_val_score(clf, X, labels, cv=5)
        return clf.feature_importances_, score, clf
    else:
        return clf.feature_importances_, clf
    
def plotFeatureImportance(columnnames, featureImportance, path, overall = True, clusnum = None): 
    #create directory
    try:
        os.mkdir(path + '/feature_importance')
    except:
        pass
    
    if overall == True:
        fig1, ax1 = plt.subplots()
        ax1.bar(columnnames[:], featureImportance)
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.title('Overall Feature Importance')
        fig1.savefig(path +'/feature_importance/feature importance.png', transparent=False, facecolor='white')
        plt.clf()
    else:
        plt.bar(x = columnnames, height = featureImportance)
        plt.title('Feature importance for cluster ' + str(clusnum))
        lo, la = plt.xticks() 
        plt.setp(la, rotation=30, horizontalalignment='right')
        plt.savefig(path +'/feature_importance/feature importance cluster ' +str(clusnum) +'.png', transparent=False, facecolor='white')
        
        plt.clf()
        
def plotFeatureImportanceByClass(columnnames, featureImportance, X, labels, clas, path): #uses dtreeviz
    #create directory
    try:
        os.mkdir(path + '/feature_importance')
    except:
        pass
    
    plt.cla()
    plt.close('all')
    for i in range(max(labels)+1):
        idx =  np.where(np.array(labels) == i)[0]
        heights = []
        idx = np.random.choice( idx.ravel(),100,replace=False) # choose 100 values
        x = X[idx]
        for j in x:
            
            ax = explain_prediction_path(clas, j, feature_names=columnnames[:], explanation_type="sklearn_default")
            
            con = ax.containers 
            for bars in con:
                heights.append([b.get_height() for b in bars])
            plt.close('all')
            plt.cla()
            
        featureImportance = np.array(heights).mean(axis=0)
        
        plotFeatureImportance(columnnames, featureImportance, path, overall = False, clusnum = i)

def treevis(clas,X,labels,columnnames, path):
   
    viz = dtreeviz(clas, 
                   X,
                   np.array(labels),
                   target_name='cluster',
                   feature_names=columnnames[:],  
                   histtype= 'bar')

    viz.save(path + '/tree.svg')
    
def tsneScatter2D(X,labels,path, title=None):
    tsne = TSNE(n_components=2, random_state = 1)
    tsneX = tsne.fit_transform(X)

    fig, ax = plt.subplots()

    scatter = ax.scatter(tsneX[:,0],tsneX[:,1], c = labels,alpha = 0.5)
    ax.set(title = 'TSNE Visulation')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    legend = ax.legend(*scatter.legend_elements(),loc="right", title="Cluster")
    ax.add_artist(legend)
    fig.savefig(path + '/TSNE Visulation.png', transparent=False, facecolor='white')
    plt.clf()
    
def pcaScatter(X, labels, path, title = None):
    pca = PCA(n_components=2)
    pcaX = pca.fit_transform(X)

    fig, ax = plt.subplots()

    scatter = ax.scatter(pcaX[:,0],pcaX[:,1], c = labels,alpha = 0.5)
    ax.set(title = 'PCA Visulation')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    legend = ax.legend(*scatter.legend_elements(),loc="right", title="Cluster")
    ax.add_artist(legend)
    fig.savefig(path + '/PCA Visulation.png', transparent=False, facecolor='white')
    plt.clf()