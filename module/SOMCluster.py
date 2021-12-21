# Heavy use of MiniSOM package https://github.com/JustGlowing/minisom
import itertools as it
from minisom import MiniSom
import numpy as np
from .KmeansCluster import kmeansCluster
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class SelfOrganisingMap:
    
    def somTrain(self, X, param):
        #given param dict train a SOM
        som = MiniSom(x = param[0], y = param[0], sigma = param[1], learning_rate = param[2], neighborhood_function = param[4], input_len = X.shape[1], random_seed = 1)
        som.train(X, num_iteration = param[3])

        return som.quantization_error(X), som.topographic_error(X)
    
    def train_som(self, X, params = None, returnBestTrained=True):
        #perform a gird search of given parameters, then returns lowest error
        if params is None:
            params = {'xy': [5,8,10,12,15,20,25],
                      'sigma' : [0.5,0.75,1,2,3,4],
                      'learning_rate' : [0.2,0.5,0.75,1,2,5],
                      'iterations' : [10000],
                      'neighborhood_function' : ['gaussian']}

        paramsCombined = list(it.product(*(params[p] for p in params))) # create a combinaation of all parameters

        results = {p:self.somTrain(X, p) for p in paramsCombined} # dictionary of results of all params

        best = list(sorted(results.items(), key=lambda item: item[1]))[0][0] #lowest quat error
        if returnBestTrained == True:
            som = MiniSom(x = best[0], y = best[0], sigma = best[1], learning_rate = best[2], neighborhood_function = best[4], input_len = X.shape[1], random_seed = 1)
            som.train(X, num_iteration = best[3])
            print("The best parameters are: " + str(best))
            return results, best, som
        else:
            return results, best
        
    def makeDistanceMap(self, som, path, show = False):
        plt.figure()
        plt.pcolor(som.distance_map().T, cmap='bone_r')
        plt.colorbar()
        plt.title('Neuron Distances')  
        
        plt.savefig(path + '/SOMdistancemap.png', transparent=False, facecolor='white')
        
        if show == True:
            plt.show()
    
    def makeLabelledSOM(self, som, X, label_names, labels, best, title, path, show = False):
        #creates a visual of the SOM with piecharts for neurons of proportions within, taken from minisom documentation 
        
        labels_map = som.labels_map(X, [label_names[t] for t in labels])

        fig = plt.figure(figsize=(10,10))
        the_grid = gridspec.GridSpec(best[0], best[0], fig)
        for position in labels_map.keys():
            label_fracs = [labels_map[position][l] for l in label_names]
            plt.subplot(the_grid[best[0]-1-position[1],
                                 position[0]], aspect=1)
            patches, texts = plt.pie(label_fracs)

        plt.legend(patches, label_names, loc = 'best', bbox_to_anchor=(2.05, 0.5)) 
        
        plt.savefig(path + '/SOMlabelledmap' + str(title) + '.png', transparent=False, facecolor='white')
        
        if show == True:
            plt.show()
        
        return plt
    
    def makeFrequencyMap(self, som, X, show=False):
        #taken from minisom documentation, shows neuron activation freqs
        plt.figure()
        frequencies = som.activation_response(X)
        plt.pcolor(frequencies.T, cmap='Blues') 
        plt.colorbar()
        plt.title('Neuron Activation Frequency')
    
        if show == True:
            plt.show()
        
        return plt
    
    def return_unique_winners(self, som, X, return_cat = False, return_big_neurons = False):
        #returns a list of unique winning neurons, not all neurons win datapoints
        winners = [list(som.winner(x)) for x in X]
        Xlen = len(X)
        unique_winners = [[w[0],w[1]] for w in list(np.unique(winners, axis = 0, return_counts=True))]
        unique_winners = [[list(np.unique(winners, axis = 0, return_counts=True)[0][i]), np.unique(winners, axis = 0, return_counts=True)[1][i]] for i in range(len(np.unique(winners, axis = 0)))]

        if return_cat == True:
            unique_winners_dict = {str(unique_winners[i][0]): i for i in range(len(unique_winners)) }
            winnersstr = [str(w) for w in winners]
            winnerscat = [unique_winners_dict[w] for w in winnersstr]
            if return_big_neurons == True:
                big_neurons = [unique_winners_dict[str(w[0])] for w in unique_winners if w[1] > Xlen*.25]
                return unique_winners, winnerscat, big_neurons
            else:
                return unique_winners, winnerscat

        else:
            return unique_winners
        
    def progressiveStatistical(self, big_neurons, data):
        #reduces big neurons into smaller
        n = big_neurons[0]

        datanew = data[data['cluster'] == n]
        Xnew = datanew[['valueLog', 'moveFreq', 'premsFreq', 'premsMedian','traceability']].to_numpy()

        resultsNew, newBest, newSOM = self.train_som(Xnew)

        winners = [list(newSOM.winner(x)) for x in Xnew]

        if len(np.unique(winners,axis=0)) > 1:
            print('reducing big SOM')
            unique_winners, winnerscat, big_neurons = return_unique_winners(winners, Xlen = len(Xnew), return_cat=True, return_big_neurons=True)
            return unique_winners, winnerscat, big_neurons
        else: 
            print('SOM cannot be reduced with this method')
            pass
        
    def SOMKmeans(self, som, path, returnLabelsShaped = False):
        # use K-means to aggregate clusters

        weights = som.get_weights()
        weightsshape = np.shape(weights)
        weightsreshape = weights.reshape(-1, weights.shape[-1])

        km = kmeansCluster()
        inertias, kneedle = km.findKMeansClusters(weightsreshape, path = path)

        labels, centers, kmeans = km.labelKMeans(weightsreshape)
        
        if returnLabelsShaped == True:
            labels = labels.reshape(weightsshape[0],weightsshape[1])
            
        return labels
    
    def SOMKmeansWinners(self, som, X, labels):
        #creates Kmeans clusters based on SOM labelling
        winners = [list(som.winner(x)) for x in X]

        kmeanslabels = [labels[w[0],w[1]] for w in winners]
        
        #some neurons dont win, meaning some k means cluster are empty, tidy this.
        labeldict={}
        i=0
        for l in np.unique(kmeanslabels):
            labeldict[int(l)] = i
            i += 1
        
        kmeanslabels = [labeldict[l] for l in kmeanslabels]
        
        return kmeanslabels