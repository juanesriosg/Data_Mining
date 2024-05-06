from typing import List

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1,
                 beta: float = 0):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """

        self.points = PointSet(features, labels, types, min_split_points)
        self.h = h
        self.min_split_points = min_split_points
        self.beta = beta
        self.best_feature, gini = self.points.get_best_gain()

        if(self.best_feature != -1):
            self.partition = self.points.get_best_threshold()

        #root = feature with best gini gain
        self.root = self.points
        self.types = types

        self.build(features,labels, types, h, min_split_points)

        

    def build(self, features, labels, types, h, min_split_points) -> None :  
        self.leave_left = []
        self.leave_right = []

        #Split to get the new trees or leafs
        labels_leaves =[[],[]]
        features_leaves = [[],[]]

        for i, f in enumerate(labels):
            if types[self.best_feature] == FeaturesTypes.CLASSES and features[i][self.best_feature] == self.partition :
                labels_leaves[0].append(labels[i])
                features_leaves[0].append(features[i])  
            elif types[self.best_feature] == FeaturesTypes.BOOLEAN and features[i][self.best_feature] == 0:
                labels_leaves[0].append(labels[i])
                features_leaves[0].append(features[i])
            elif types[self.best_feature] == FeaturesTypes.REAL and features[i][self.best_feature] < self.partition:
                labels_leaves[0].append(labels[i])
                features_leaves[0].append(features[i])
            else:
                labels_leaves[1].append(labels[i])
                features_leaves[1].append(features[i])
            

        #Add validation of h or stop if the label is the same or if there is non a split that allows a minimun points
        if(h == 1):
            self.leave_left = PointSet(features_leaves[0],labels_leaves[0],types,min_split_points)
            self.leave_right = PointSet(features_leaves[1],labels_leaves[1],types,min_split_points)
        else:
            if(self.stop(labels_leaves[0]) or len(labels_leaves[0]) < 2 * min_split_points ):
                self.leave_left = PointSet(features_leaves[0],labels_leaves[0],types)
            elif(PointSet(features_leaves[0],labels_leaves[0],types,min_split_points).get_best_gain()[0] == -1):
                self.leave_left = PointSet(features_leaves[0],labels_leaves[0],types,min_split_points)
            else:
                self.leave_left = Tree(features_leaves[0], labels_leaves[0], types, h-1, min_split_points)

            if(self.stop(labels_leaves[1]) or len(labels_leaves[1]) < 2 * min_split_points ):
                self.leave_right = PointSet(features_leaves[1],labels_leaves[1],types,min_split_points)
            elif(PointSet(features_leaves[1],labels_leaves[1],types,min_split_points).get_best_gain()[0] == -1):
                self.leave_right = PointSet(features_leaves[1],labels_leaves[1],types,min_split_points)
            else:
                self.leave_right = Tree(features_leaves[1], labels_leaves[1], types, h-1, min_split_points)

    def stop(self, labels):
        
        count0 =  0
        count1 =  0

        for label in labels:
            if label == 0:
                count0 += 1
            elif label == 1:
                count1 += 1

        return True if count0==len(labels) or count1==len(labels) else False

    def leaf_class(self,labels):
        count0 =  (labels == 0).sum()
        count1 =  (labels == 1).sum()

        return 0 if count0>count1 else 1
    
    def print_tree(self):
        """Give the structure of the tree """

        print(f"Tree: {self}")
        print(f"Height: {self.h}")
        print(f"Best Feature: {self.best_feature}")
        print(f"Best Feature Type: {self.types[self.best_feature]}")
        print(f"Partition: {self.partition}")
        
        if(type(self.leave_left) == Tree):
            print(f"Left Leaf Tree")
            self.leave_left.print_tree()
        else:
            print("Left Leaf")
            print(self.leave_left.labels)
            print(f"Decision: {bool(self.leaf_class(self.leave_left.labels))}")

        if(type(self.leave_right) == Tree):
            print("Right Leaf Tree")
            self.leave_right.print_tree()
        else:
            print("Right Leaf")
            print(self.leave_right.labels)
            print(f"Decision: {bool(self.leaf_class(self.leave_right.labels))}")

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        
        leaf = features[self.best_feature]
        
        if self.types[self.best_feature] == FeaturesTypes.CLASSES:
            if leaf == self.partition:
                
                if(type(self.leave_left) == Tree):
                    return self.leave_left.decide(features)
                else :
                    return bool(self.leaf_class(self.leave_left.labels))
            else:
                if(type(self.leave_right) == Tree):
                    return self.leave_right.decide(features)
                else :
                    return bool(self.leaf_class(self.leave_right.labels)) 
        if self.types[self.best_feature] == FeaturesTypes.BOOLEAN:
            if leaf == 0:                
                if(type(self.leave_left) == Tree):
                    return self.leave_left.decide(features)
                else :
                    return bool(self.leaf_class(self.leave_left.labels))
            else:
                if(type(self.leave_right) == Tree):
                    return self.leave_right.decide(features)
                else :
                    return bool(self.leaf_class(self.leave_right.labels)) 
        if self.types[self.best_feature] == FeaturesTypes.REAL:
            if leaf < self.partition:
                
                if(type(self.leave_left) == Tree):
                    return self.leave_left.decide(features)
                else :
                    return bool(self.leaf_class(self.leave_left.labels)) 
            else:
                if(type(self.leave_right) == Tree):
                    return self.leave_right.decide(features)
                else :
                    return bool(self.leaf_class(self.leave_right.labels)) 

    def add_training_point(self, features: List[float], label:bool)-> None :
        
        lookFor = self.lookFor(features, [])
        c = 0
        for vi in lookFor:
            if(type(vi) == Tree and vi.points != None):
                vi.points = vi.points.addElement(features, label)
            elif type(vi) == PointSet:
                vi = vi.addElement(features, label)
            c += 1
            
            if type(vi) == Tree and (vi.points != None and c >= self.beta * len(vi.points.labels)) :
                c = 0
                self = Tree(vi.points.features, vi.points.labels, self.h, self.min_split_points, self.beta)
    
    def del_training_point(self, features: List[float], label:bool)-> None :
        lookFor = self.lookFor(features, [])
        c = 0
        for index,vi in enumerate(lookFor):
            if(type(vi) == Tree  and vi.points != None):
                vi.points = vi.points.deleteElement(features, label)
            elif type(vi) == PointSet:
                vi = vi.deleteElement(features, label)
            c += 1
            
            if type(vi) == Tree and (vi.points != None and c >= self.beta * len(vi.points.labels)) :
                c = 0
                if vi.points == None:
                    j = index
                    while(type(lookFor[j-1]) != Tree ):
                        j -= 1

                    if(lookFor[j-1] == Tree): 
                        self = Tree(lookFor[j-1].points.features, lookFor[j-1].points.labels, self.h, self.min_split_points, self.beta)
                else:
                    self = Tree(vi.points.features, vi.points.labels, self.h, self.min_split_points, self.beta)
    
    def lookFor(self,features: List[float], path: List[PointSet]) -> List :
        
        leaf = features[self.best_feature]
        if (self.types[self.best_feature] == FeaturesTypes.CLASSES and leaf == self.partition):
            path.append(self.leave_left)
            if(type(self.leave_left) == Tree):
                return self.leave_left.lookFor(features, path)
            else :
                return path
            
        if (self.types[self.best_feature] == FeaturesTypes.BOOLEAN and leaf == 0):              
            path.append(self.leave_left)
            if(type(self.leave_left) == Tree):
                return self.leave_left.lookFor(features, path)
            else :
                return path
           
        if (self.types[self.best_feature] == FeaturesTypes.REAL and leaf < self.partition):
            path.append(self.leave_left)
            if(type(self.leave_left) == Tree):
                return self.leave_left.lookFor(features, path)
            else :
                return path
        else:
            path.append(self.leave_right)
            if(type(self.leave_right) == Tree):
                
                return self.leave_right.lookFor(features, path)
            else :
                return path