import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler 

def averageBarchart(X, labels, columnnames, path, show = False):
    try:
        os.mkdir(path + '/aggregation')
    except:
        pass
    
    x = MinMaxScaler().fit_transform(X) #normalise
    
    gap = .8 /( max(labels)+1)
    i=-((max(labels)+1)/2)

    idx = np.asarray([i for i in range(len(columnnames))])

    fig, ax = plt.subplots(figsize=(15, 5))
    
    #calulcate mean per column name
    
    
    for l in range(max(labels)+1):
        
        x1 = x[np.where(np.array(labels) == l)]

        xmeans = [np.mean(x2) for x2 in np.array(x1).transpose()]

        ax.bar(idx + (i *gap),
               xmeans,
               width = gap,
               label = l)
        i+=1
        
    ax.set_xticks(idx)
    ax.set_xticklabels(list(columnnames))
    ax.legend(loc='best')
    
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    fig.savefig(path +'/aggregation/barchart.png', transparent=False, facecolor='white')
    
    if show == True:
        plt.show()
    
    plt.clf()
    
def boxplot(x, y, columns,path, normalise = True, show = False):
    try:
        os.mkdir('./vis/aggregation')
    except:
        pass
    
    Xdf = pd.DataFrame(x,columns = list(columns))

    if normalise == True:
        Xdf=(Xdf-Xdf.min())/(Xdf.max()-Xdf.min())

    X = Xdf
    Y = y    

    for column in range(len(Xdf.iloc[0])):
        # Plot the orbital period with horizontal boxes
        fig, ax = plt.subplots()
        sns.boxplot(y=X.iloc[:,column], x=Y,
                    whis=[0, 100], width=.6, palette="vlag")

        # Add in points to show each observation
        sns.stripplot(y=X.iloc[:,column], x=Y,
                      size=4, color=".3", linewidth=0, alpha =0.3)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        sns.despine(trim=True, left=True)
        ax.set_title(columns[column])
        fig.savefig(path + '/aggregation/boxplot of ' + str(columns[column]) + '.png', transparent=False, facecolor='white')
        
        if show == True:
            plt.show()
        
        plt.clf()

def pltNormalDistwithMeans(clustersummarydf, X, path, show=False):
    try:
        os.mkdir(path + '/aggregation/normalDistrubution')
    except:
        pass
    i = 0
    for c in clustersummarydf.columns:
        avg = np.mean(X[:,i])
        std = np.std(X[:,i])
        x = np.linspace(avg - 5*std, avg + 5*std, 100)

        plt.plot(x, stats.norm.pdf(x, avg, std))

        y=[min(stats.norm.pdf(x, avg, std)),max(stats.norm.pdf(x, avg, std))]
        x1 = [avg-std,avg-std]
        plt.plot(x1,y,alpha=0.15,c='r')

        x1 = [avg+std,avg+std]
        plt.plot(x1,y,alpha=0.15,c='r')
        j=0
        for x in clustersummarydf[c]:
            y = stats.norm.pdf(x, avg, std)
            plt.plot(x,y,marker = 'x', label=str(j),  markersize = 10)
            j+=1
        plt.legend()
        plt.title(c)
        plt.savefig(path + '/aggregation/normalDistrubution/' + str(c) + '.png', transparent=False, facecolor='white')
        
        if show == True:
            plt.show()
        
        plt.clf()
        i += 1
        
def pltHistwithMeans(clustersummarydf, X, path, show = False):
    try:
        os.mkdir(path + '/aggregation/histogramDistrubutionwithMeans')
    except:
        pass
    i = 0
    for c in clustersummarydf.columns:
        x = np.linspace(min(X[:,i]),max(X[:,i]), 100)
        gaus = stats.gaussian_kde(X[:,i])
        plt.plot(x,gaus(x))
        
        y1 = [min(gaus(x)),max(gaus(x))]
        x1 = [np.percentile(X[:,i],68),np.percentile(X[:,i],68)]
        plt.plot(x1,y1,alpha=0.15,c='r')
        x1 = [np.percentile(X[:,i],32),np.percentile(X[:,i],32)]
        plt.plot(x1,y1,alpha=0.15,c='r')
        
        j=0
        for x in clustersummarydf[c]:
            y = gaus(x)
            plt.plot(x,y,marker = 'x', label=str(j),  markersize = 10)
            j+=1
        plt.legend()
        plt.title(c)
        plt.savefig(path + '/aggregation/histogramDistrubutionwithMeans/' + str(c) + '.png', transparent=False, facecolor='white')
        if show == True:
            plt.show()
        plt.clf()
        i += 1
        
def pltHistWithCluster(X, labels,path, show = False):
    try:
        os.mkdir(path + '/aggregation/histogramDistrubution')
    except:
        pass
    i = 0
    for x in X.transpose():
        xnew = np.linspace(min(x),max(x), 100)
        j=0
        for l in range(max(labels)+1):

            xl = x[np.where(np.array(labels)==l)]
            xlnew = np.linspace(min(xl),max(xl), 100)
            gaus = stats.gaussian_kde(xl)
            plt.plot(xlnew,gaus(xlnew), label = j)
            j+=1
        plt.legend()
        plt.title(columnnames[i])
        plt.show()
        plt.savefig(path + '/aggregation/histogramDistrubution/' + str(c) + '.png', transparent=False, facecolor='white')
        
        if show == True:
            plt.show()
            
        plt.clf()    
        i += 1   
        
def radarClus(X, labels, columnnames, path, show = False):
    
    labelloc = np.linspace(start=0, stop=2 * np.pi, num=len(columnnames))

    x = MinMaxScaler().fit_transform(X)    
    
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    for i in range(max(labels)+1):
        x1 = x[np.where(np.array(labels) == i)]
        xmeans = [np.mean(col) for col in x1.transpose()]
        plt.plot(labelloc, xmeans, label='Cluster' +str(i), linewidth = 5, alpha = .75, linestyle= 'dashdot')
    lines, labels = plt.thetagrids(np.degrees(labelloc), labels=columnnames)
    plt.legend()
    
    plt.savefig(path + '/aggregation/radar.png', transparent=False, facecolor='white')
    
    if show == True:
        plt.show()
            
    plt.clf() 

def parallelClus(X, labels, columnnames, path, show = False):
    
    x = MinMaxScaler().fit_transform(X)

    i = 0

    plt.figure(figsize=(15, 5))

    for i in range(max(labels)+1):
        x1 = x[np.where(np.array(labels) == i)]
        xmedian = np.array([np.median(col) for col in x1.transpose()])
                  
        plt.plot(xmedian, label = 'Cluster ' +str(i), linewidth = 3, alpha = 0.75)

        xvar = np.array([np.var(col) for col in x1.transpose()])
        plt.fill_between(np.arange(len(xmedian)),xmedian-xvar,xmedian+xvar, alpha = 0.25)
        i+=1

    plt.legend()
    plt.grid(axis = 'x')
    plt.xticks(np.arange(0,len(x[1])),labels = columnnames)
    plt.savefig(path + '/aggregation/parallel.png', transparent=False, facecolor='white')
    
    if show == True:
        plt.show()
        
    plt.clf() 