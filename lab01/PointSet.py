from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes], min_split_points: int = 1):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """

        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.min_split_points = min_split_points

        self.best_partition = None
        self.best_feature = -1

    

    def calculate_partitions(self, feature):
        partitions = []
        unique_values = np.unique(feature)

        for f in unique_values:
                partition = [[f],list(set(unique_values)-set([f]))]
                if len(partition[0])>0 and len(partition[1])>0:
                    partitions.append(partition)

        return partitions


    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        #Get the frequency for every unique value in the classes
        _ , count_labels = np.unique(self.labels, return_counts=True)
        
        gini = 1 - np.sum((count_labels/len(self.labels))**2)

        return gini     

    def get_partition_gini(self,node) -> float :
        #Get the frequency for every unique value in the classes
        _ , count_labels = np.unique(node, return_counts=True)
        gini = 1 - np.sum((count_labels/len(node))**2)

        return gini

    def get_gini_continuous(self, nodes) -> float :
        """Compute the gini of a continuos feature
        ----------
        nodes : List[List[int]]
            The nodes matrix is the actual matrix to compute the gini with a threshold
        
        Returns
        -------
        float
            The threshold that gives the best splitting
        """


        gini0 = 1 - np.sum((nodes[0]/np.sum(nodes[0]))**2) if np.sum(nodes[0])>0 else 0
        gini1 = 1 - np.sum((nodes[1]/np.sum(nodes[1]))**2) if np.sum(nodes[1])>0 else 0

        
        #Calculate the gini Split
        gini_split = (np.sum(nodes[0])*gini0 + np.sum(nodes[1])*gini1 )/len(self.labels)

        #Calculate the gini gain
        gini_gain = self.get_gini() - gini_split
        return gini_gain

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        
        features_transpose = np.transpose(self.features)
        best_gini_gain = -9999
        feature_best_gini_gain = -1
        best_partition = None

        original_gini = self.get_gini()

        for index,feature in enumerate(features_transpose):

                
                if self.types[index] == FeaturesTypes.BOOLEAN or self.types[index] == FeaturesTypes.CLASSES: 
                #if self.types[index] == FeaturesTypes.BOOLEAN or self.types[index] == FeaturesTypes.CLASSES or self.types[index] == FeaturesTypes.REAL: 
            
                    partitions =  [0] if self.types[index] ==FeaturesTypes.BOOLEAN else np.unique(feature)


                    for partition in partitions:
                        node0 = []
                        node1 = []
                        for i in range(len(feature)):
                            if feature[i] == partition and (self.types[index] == FeaturesTypes.BOOLEAN or self.types[index] == FeaturesTypes.CLASSES):
                                node0.append(self.labels[i])
                            elif feature[i] < partition and self.types[index] == FeaturesTypes.REAL:
                                node0.append(self.labels[i])
                            else:
                                node1.append(self.labels[i])

                        #Calculate gini for every node in the partition
                        gini0 = self.get_partition_gini(node0)
                        gini1 = self.get_partition_gini(node1)

                        #Calculate the gini Split
                        gini_split = (len(node0)*gini0 + len(node1)*gini1 )/len(self.labels)

                        #Calculate the gini gain
                        gini_gain = original_gini - gini_split 

                        if gini_gain > best_gini_gain and len(node0)>=self.min_split_points and len(node1)>=self.min_split_points :
                            best_gini_gain = gini_gain
                            feature_best_gini_gain = index
                            best_partition = partition

                else :
                    partitions =  [feature, self.labels]
                    partitions = [{"feature":item[0], "label":item[1]} for item in np.transpose(partitions)]
                    partitions = sorted(partitions, key=lambda x:x['feature'])

                    _,count_values_node1 = np.unique(self.labels, return_counts=True)
                    changeMatrix = [[0,0],count_values_node1]


                    for i,partition in enumerate(partitions):
                        
                        changeMatrix[0][int(partition['label'])] += 1
                        changeMatrix[1][int(partition['label'])] -= 1

                        
                        if i<len(partitions)-1 and partition['feature'] != partitions[i+1]['feature'] :
                            gini_gain = self.get_gini_continuous(changeMatrix)
                            

                            if gini_gain > best_gini_gain and  np.sum(changeMatrix[0])>=self.min_split_points and  np.sum(changeMatrix[1])>=self.min_split_points :
                                best_gini_gain = gini_gain
                                feature_best_gini_gain = index
                                
                                best_partition =  partition['feature']
                    

        self.best_partition = best_partition
        self.best_feature = feature_best_gini_gain
        return (feature_best_gini_gain,best_gini_gain)
        #raise NotImplementedError('Please implement this methode for Question 2')

    def get_best_threshold(self) -> float :
        """Compute the threshold along which splitting provides the best gain

        Returns
        -------
        float
            The threshold that gives the best splitting
        """
        if self.best_feature != -1:
            if self.types[self.best_feature] == FeaturesTypes.BOOLEAN:
                return None
            elif self.types[self.best_feature] == FeaturesTypes.CLASSES:
                return self.best_partition
            elif self.types[self.best_feature] == FeaturesTypes.REAL:
                
                auxiliar_features = np.transpose(self.features)
                sorted_feature = np.sort(auxiliar_features[self.best_feature])
                actual_threshlod = self.best_partition
                
                less_threshold = [value for value in sorted_feature if value <= actual_threshlod]
                geq_threshold = [value for value in sorted_feature if value > actual_threshlod]
                maxL = np.max(less_threshold) if len(less_threshold)>0 else 0
                minR = np.min(geq_threshold) if len(geq_threshold)>0 else 0
                    
                return (maxL+minR)/2
        else:
            raise Exception('Not Gini Called')
        
    def addElement(self, features, label):
        self.features = np.append(self.features, features)
        self.labels = np.append(self.labels, label)
    
    def deleteElement(self, features, label):
        self.features = np.setdiff1d(self.features, features )
        self.labels = np.setdiff1d(self.labels, label)