from sklearn.cluster import KMeans
import kneed
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class kmeansCluster:
    
    def Norm(self,x):
        #calculate max and min
        maxi = np.max(x)
        mini = np.min(x)
        
        #normalise between 0 and 1
        funcNorm = lambda x: (x-mini)/(maxi-mini)
        
        return(funcNorm(x))

    def plotInertia(self, clusters, intertias, path, show = False):
        
        #smooth points, for visual reasons
        clustersNew = np.linspace(np.min(clusters), np.max(clusters), 300)  
        inertiasSmooth = interp1d(clusters, intertias, kind='cubic')
        
        #create plot
        fig, ax = plt.subplots()
        ax.plot(clustersNew, inertiasSmooth(clustersNew), label='Inertia')
        ax.set(title = 'Elbow Method Visulisation', 
               xlabel = 'Number of Clusters',
               ylabel = 'Score')
        ax.legend(loc='best')
        
        fig.savefig(path + '/kmeansinertia.png', transparent=False, facecolor='white')
        
        if show == True:
            plt.show()
        
        return ax
        
    def findKMeansClusters(self, x, path, minClusters=2, maxClusters=50):
        
        clusters = []
        #scores = []
        inertias = []
    
        #nomrmalise x
        x = MinMaxScaler().fit_transform(x)
        
        #run kmeans on cluster sizes
        for i in range(minClusters, maxClusters):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(x)
            clusters.append(i)
            #scores.append(kmeans.score(x))
            inertias.append(kmeans.inertia_)
        
        #normalise    
        intertiasNorm = self.Norm(inertias)
        
        #plot graph
        ax = self.plotInertia(clusters, intertiasNorm, path)
        
        kneedle = kneed.KneeLocator(np.arange(np.array(intertiasNorm).size), intertiasNorm, curve = "convex", direction="decreasing")
        
        self.knee = kneedle.knee
        
        return inertias, kneedle
    
    def labelKMeans(self,x, maxclusters = 10, clustersNum = 5, useKneedle = True):
        
        if maxclusters < self.knee:
            useKneedle = False
            clustersNum = maxclusters
            print('Warning: too many kneedle clusters, reverting to max clusters')
        
        if useKneedle == True:
            clustersNum = self.knee
            
        x = MinMaxScaler().fit_transform(x) #scale data
        
        kmeans = KMeans(n_clusters=clustersNum, random_state=0).fit(x) #run Kmeans
        
        #find labels
        labels = kmeans.predict(x)
        centers = kmeans.cluster_centers_
        
        return labels, centers, kmeans