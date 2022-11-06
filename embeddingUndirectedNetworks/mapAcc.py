#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import math
import random as rand
import os
from scipy import stats



#A function for creating a dictionary containing the randomly selected set of node pairs and the shortest path length between them to be examined with respect to the mapping accuracy.
#G is the examined NetworkX Graph (connected component with only single edges and no self-loops)
#numOfNodePairsInTheSample can be set to an integer (e.g. 100) denoting the number of node pairs contained by the chosen set of node pairs. The default is the string 'all', meaning that we want to consider all the node pairs that are connected to each other by a path, therefore all such node pairs will be inputted to the functions that calculate the pairwise geometric measures. Change numOfNodePairsInTheSample from the default value for examining large networks, for which the total number of possible pairs of vertexes would be too large.
def createNodePairListForTest(G,numOfNodePairsInTheSample='all'):
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight=None)) #weight=None: every edge has weight 1 #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node=length of the shortest path from the target node to the source node
    nodeList = list(G.nodes())
    N = len(nodeList)
    nodePair_SPL_dict = {}
    #iterate over each undirected node pair once
    for sID in range(N-1):
        s = nodeList[sID]
        for tID in range(sID+1,N): #the pairing of each node with itself is disregarded
            t = nodeList[tID]
            nodePair_SPL_dict[(s,t)] = shortestPathLengthsDict[s][t]

    if numOfNodePairsInTheSample=='all': 
        return nodePair_SPL_dict

    else: #create a sample of the above dictionary, containing a random set of the embedded node pairs that are connected to each other by at least one path
        selectedNodePairs = rand.sample(nodePair_SPL_dict.keys(),numOfNodePairsInTheSample)
        nodePair_SPL_dict_sampled = {}
        for nodePair in selectedNodePairs:
            nodePair_SPL_dict_sampled[nodePair] = nodePair_SPL_dict[nodePair]
        return nodePair_SPL_dict_sampled





# hyperbolic embedding #################################################################################################################################

#This function calculates the hyperbolic distance h between two nodes of the network. The NumPy vectors coords1 and coords2 both contain d Cartesian coordinates, which describe the position of the given nodes in the native representation of the d-dimensional hyperbolic space of curvature K<0.
def hypDist(coords1,coords2,K=-1):
    zeta = math.sqrt(-K)
    r1 = np.linalg.norm(coords1) #the radial coordinate of node 2 in the native representation, i.e. the Euclidean length of the vector coords2
    r2 = np.linalg.norm(coords2) #the radial coordinate of node 2 in the native representation, i.e. the Euclidean length of the vector coords2
    if r1==0:
        h = r2
    elif r2==0:
        h = r1
    else:
        cos_angle = np.inner(coords1,coords2)/(r1*r2) #cosine of the angular distance between the two nodes
        if cos_angle==1: #the vectors coords1 and coords2 point in the same direction; in this case the hyperbolic distance between the two nodes is acosh(cosh(r1-r2))=|r1-r2|
            h = math.fabs(r1-r2)
        elif cos_angle==-1: #the vectors coords1 and coords2 point in the opposite direction
            h = r1+r2
        else:
            argument_of_acosh = math.cosh(zeta*r1)*math.cosh(zeta*r2)-math.sinh(zeta*r1)*math.sinh(zeta*r2)*cos_angle
            if argument_of_acosh<1: #a rounding error occurred, because the hyperbolic distance h is close to zero
                print("The argument of acosh is "+str(argument_of_acosh)+", less than 1.")
                h = 0 #acosh(1)=0
            else:
                h = math.acosh(argument_of_acosh)/zeta
    return h



#This function calculates the mapping accuracy in the case of hyperbolic embeddings: it returns the Spearman's correlation coefficient between the shortest path lengths and the pairwise hyperbolic distances between the network nodes.
#dictOfSPLs is a dictionary with tuples of node names corresponding to the examined node pairs as keys and the shortest path length between the two nodes as values
#Coord is a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the native representation of the d-dimensional hyperbolic space.
#K<0 is the curvature of the hyperbolic space
def mapAcc_hyp(dictOfSPLs,Coord,K=-1):
    SPLlist = []
    measureList = []
    for nodePair in dictOfSPLs.keys():
        SPLlist.append(dictOfSPLs[nodePair])
        geomMeasure = hypDist(Coord[nodePair[0]],Coord[nodePair[1]],K) #smaller hyperbolic distance=larger proximity=smaller shortest path length
        measureList.append(geomMeasure)
    MA = stats.spearmanr(SPLlist,measureList)[0] #the function stats.spearmanr returns two values: the correlation and the associated pvalue
    return MA



#  Euclidean embedding  ###################################################################################################################################

#This function calculates the mapping accuracy using inner products in Euclidean embeddings: it returns the Spearman's correlation coefficient between the shortest path lengths and the pairwise -inner products between the network nodes.
#dictOfSPLs is a dictionary with tuples of node names corresponding to the examined node pairs as keys and the shortest path length between the two nodes as values
#Coord is a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
def mapAcc_Euc_inner(dictOfSPLs,Coord):
    SPLlist = []
    measureList = []
    for nodePair in dictOfSPLs.keys():
        SPLlist.append(dictOfSPLs[nodePair])
        geomMeasure = -np.inner(Coord[nodePair[0]],Coord[nodePair[1]]) #larger inner product=larger proximity=smaller shortest path length
        measureList.append(geomMeasure)
    MA = stats.spearmanr(SPLlist,measureList)[0] #the function stats.spearmanr returns two values: the correlation and the associated pvalue
    return MA


#This function calculates the mapping accuracy using distances in Euclidean embeddings: it returns the Spearman's correlation coefficient between the shortest path lengths and the pairwise Euclidean distances between the network nodes.
#dictOfSPLs is a dictionary with tuples of node names corresponding to the examined node pairs as keys and the shortest path length between the two nodes as values
#Coord is a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
def mapAcc_Euc_dist(dictOfSPLs,Coord):
    SPLlist = []
    measureList = []
    for nodePair in dictOfSPLs.keys():
        SPLlist.append(dictOfSPLs[nodePair])
        geomMeasure = np.linalg.norm(Coord[nodePair[0]]-Coord[nodePair[1]]) #smaller Euclidean distance=larger proximity=smaller shortest path length
        measureList.append(geomMeasure)
    MA = stats.spearmanr(SPLlist,measureList)[0] #the function stats.spearmanr returns two values: the correlation and the associated pvalue
    return MA

