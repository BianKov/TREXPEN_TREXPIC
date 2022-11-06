#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
import operator
import random as rand



#A function for creating a dictionary containing the randomly selected set of node pairs to be examined with regard to the greedy routing as keys and the corresponding shortest path length as values. There is always a path of positive, finite length between the selected node pairs.
#G is the examined NetworkX Graph (connected component with only single edges and no self-loops)
#numOfNodePairsInTheSample can be set to an integer (e.g. 250000) denoting the number of node pairs in the chosen set. The default is the string 'all', meaning that we want to examine all the node pairs that are connected with a path of positive, finite length. Change numOfNodePairsInTheSample from the default value for testing greedy routing on large networks, for which the total number of possible pairs of vertexes would be too large to examine.
def createNodePairListForTest(G,numOfNodePairsInTheSample='all'):
    totalDictOfProperNodePairs = {} #key=(sourceNodeName,targetNodeName) tuple, value=hop-length of the connecting shortest path
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight=None)) #weight=None: every edge has weight 1
        #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node
    for s in G: #iterate over the nodes as sources
        dictForReachableTargets = shortestPathLengthsDict[s] #dictionary with names of the reachable target nodes as keys and their distance from node s as values
        for t in dictForReachableTargets.keys(): #iterate over the reachable target nodes
            if s!=t:
                totalDictOfProperNodePairs[(s,t)] = dictForReachableTargets[t]
    if numOfNodePairsInTheSample=='all':
        return totalDictOfProperNodePairs
    else: #sample the given number of node pairs
        if len(totalDictOfProperNodePairs)<numOfNodePairsInTheSample:
            print('\nERROR: The total number of proper node pairs is '+str(len(totalDictOfProperNodePairs))+', smaller than the required number of node pairs. All the proper node pairs are returned.\n')
            return totalDictOfProperNodePairs
        else:
            listOfExaminedNodePairs = rand.sample(totalDictOfProperNodePairs.keys(),numOfNodePairsInTheSample)
            examinedDictOfNodePairs = {nodePair:totalDictOfProperNodePairs[nodePair] for nodePair in listOfExaminedNodePairs}
            return examinedDictOfNodePairs



#This function returns the greedy routing score (average ratio between shortest path lengths and hop-length of greedy routing paths, where the hop-length of unsuccessful GR paths is infinite), the percentage of successful greedy routing paths and the average hop-length of the successful greedy routing paths calculated for an embedded network, and the average shortest path length between those nodes that are connected by a successful greedy routing path.
#G is the embedded NetworkX Graph
#examinedNodePairsDict is a dictionary containing (start,destination) tuples of node names corresponding to the the examined node pairs (each of them are connected by a path of positive, finite length) as keys and the corresponding shortest path lengths as values.
#Coord is a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in a d-dimensional space.
#usedMeasure is a string determining on what measure the greedy routing is based:
    #'innerProd' means that the greedy router always maximizes its position vector's inner product with the position vector of the destination
    #'EucDist' means that the greedy router always minimizes its position's Euclidean distance from the position of the destination
    #'hypDist' means that the greedy router always minimizes its position's hyperbolic distance from the position of the destination
#K<0 is the curvature of the hyperbolic space (this parameter is not used when usedMeasure is 'innerProd' or 'EucDist')
#Example for function call:
#   [GRscore_hyp,percentageOfSuccPaths_hyp,avgGRlength_hyp,avgSHlength_hyp]=GRcalculation.greedy_routing(G,Coord_hyp,'hypDist')
def greedy_routing(G,examinedNodePairsDict,Coord,usedMeasure,K=-1):
    N = len(G) #the number of nodes in graph G

    examinedEndPoints = set([nodePair[1] for nodePair in examinedNodePairsDict.keys()])

    measureDict = {} #initialize the dictionary of the pairwise node-node "distances", based on which the greedy routing will be performed
    closestDict = {} #initialize the dictionary "closest": for each node pair s->t, determine the name of the neighbour of s closest to t
    GRpathsDict = {} #initialize the dictionary "paths": key=(sourceNodeName,targetNodeName) tuple, value=hop-length of the greedy path from sourceNodeName to targetNodeName
                         #the greedy path length will be Inf at unsuccessful greedy paths (where shortest path length is not Inf); otherwise, it contains non-negative integers, corresponding to the hop-length of successful path
    for startNode in Coord.keys():
        for endNode in examinedEndPoints: #iterate over those nodes that are present in the examined paths as endpoints
            measureDict[(startNode,endNode)] = math.nan
            closestDict[(startNode,endNode)] = math.nan
            GRpathsDict[(startNode,endNode)] = math.nan
            #nan means that the given value has not been calculated (and has not been used)

    #calculate the greedy path lengths
    if usedMeasure=='innerProd':
        for (startNode,endNode) in examinedNodePairsDict.keys():
            #check whether we already know the greedy path length from node startNode to node endNode
            if math.isnan(GRpathsDict[(startNode,endNode)]): #we do not know yet
                greedy_routing_rec_EucInner(startNode, endNode, G, Coord, measureDict, closestDict, GRpathsDict)
    elif usedMeasure=='EucDist':
        for (startNode,endNode) in examinedNodePairsDict.keys():
            #check whether we already know the greedy path length from node startNode to node endNode
            if math.isnan(GRpathsDict[(startNode,endNode)]): #we do not know yet
                greedy_routing_rec_EucDist(startNode, endNode, G, Coord, measureDict, closestDict, GRpathsDict)
    elif usedMeasure=='hypDist':
        for (startNode,endNode) in examinedNodePairsDict.keys():
            #check whether we already know the greedy path length from node startNode to node endNode
            if math.isnan(GRpathsDict[(startNode,endNode)]): #we do not know yet
                greedy_routing_rec_hypDist(startNode, endNode, G, Coord, measureDict, closestDict, GRpathsDict, K)
    else:
        print("Error: Invalid setting of the parameter usedMeasure. Set usedMeasure to one of the strings 'innerProd', 'EucDist' or 'hypDist'!")

    numOfSuccessfulGRs = 0
    sumOfGRpathLengths = 0
    sumOfShPathLengths = 0
    sumOfShGRratios = 0
    for (startNode,endNode) in examinedNodePairsDict.keys():
        if not math.isinf(GRpathsDict[(startNode,endNode)]):
            numOfSuccessfulGRs = numOfSuccessfulGRs+1
            sumOfGRpathLengths = sumOfGRpathLengths+GRpathsDict[(startNode,endNode)]
            sumOfShPathLengths = sumOfShPathLengths+examinedNodePairsDict[(startNode,endNode)]
            sumOfShGRratios = sumOfShGRratios+(examinedNodePairsDict[(startNode,endNode)]/GRpathsDict[(startNode,endNode)])
    totalNumOfRoutes = len(examinedNodePairsDict)
    percentageOfSuccPaths = numOfSuccessfulGRs/totalNumOfRoutes #calculate the percentage of successful greedy routing paths (note: the diagonal is not considered)
    avgGRlength = sumOfGRpathLengths/numOfSuccessfulGRs #calculate the average hop-length of the successful greedy routing paths
    avgSHlength = sumOfShPathLengths/numOfSuccessfulGRs #calculate the average hop-length of the shortest path lengths where the greedy routing was successful
    GRscore = sumOfShGRratios/totalNumOfRoutes #calculate the GR-score

    return [GRscore,percentageOfSuccPaths,avgGRlength,avgSHlength]



def greedy_routing_rec_EucInner(i, j, G, coords, measures, closestNeighbors, GRpaths): #determine the length of the GR path from node i to node j
    GRpaths[(i,j)] = -1 #mark that the ith node (the starting node) has already been visited in the current path
    if math.isnan(closestNeighbors[(i,j)]): #we have not determined which neighbor of node i is the closest on to node j
        bestMeasure = math.inf
        for neighborNode in G.neighbors(i):
            measures[(neighborNode,j)] = -np.inner(coords[neighborNode],coords[j]) #greedy routing: always step to the node neighbour minimising this measure, i.e. to the one of which the position vector has the largest inner product with the position vector of the destination (larger inner product=larger proximity -> maximise the inner product)
            if measures[(neighborNode,j)]<bestMeasure:
                closestNeighbors[(i,j)] = neighborNode
                bestMeasure = measures[(neighborNode,j)]
            elif measures[(neighborNode,j)]==bestMeasure:
                if rand.random()<0.5: #sample a floating point number in the range [0.0, 1.0)
                    closestNeighbors[(i,j)] = neighborNode
        if math.isinf(bestMeasure):
            GRpaths[(i,j)] = math.inf #DURING the greedy routing we reached a node from which the endpoint is not reachable - the unsuccessfulness of the greedy routing is the fault of the greedy routing
    if closestNeighbors[(i,j)]==j: #the node following node i is the endpoint
        GRpaths[(i,j)] = 1
    elif GRpaths[(closestNeighbors[(i,j)],j)]==-1: #the node following the ith node has already been visited in the current path -> we are in a loop
        GRpaths[(i,j)] = math.inf
    else:
        if math.isnan(GRpaths[(closestNeighbors[(i,j)],j)]): #the node following node i has not been visited yet and we do not know the length of the path from it to the endpoint
            greedy_routing_rec_EucInner(closestNeighbors[(i,j)], j, G, coords, measures, closestNeighbors, GRpaths)

        if math.isinf(GRpaths[(closestNeighbors[(i,j)],j)]): #the endpoint can not be reached via greedy routing from the node following node i
            GRpaths[(i,j)] = math.inf
        else: #the endpoint can be reached via greedy routing from the node following node i
            GRpaths[(i,j)] = 1 + GRpaths[(closestNeighbors[(i,j)],j)]



def greedy_routing_rec_EucDist(i, j, G, coords, measures, closestNeighbors, GRpaths): #determine the length of the GR path from node i to node j
    GRpaths[(i,j)] = -1 #mark that the ith node (the starting node) has already been visited in the current path
    if math.isnan(closestNeighbors[(i,j)]): #we have not determined which neighbor of node i is the closest on to node j
        bestMeasure = math.inf
        for neighborNode in G.neighbors(i):
            measures[(neighborNode,j)] = np.linalg.norm(coords[neighborNode]-coords[j]) #greedy routing: always step to the node neighbour minimising this measure, i.e. to the one that has the smallest Euclidean distance from the destination
            if measures[(neighborNode,j)]<bestMeasure:
                closestNeighbors[(i,j)] = neighborNode
                bestMeasure = measures[(neighborNode,j)]
            elif measures[(neighborNode,j)]==bestMeasure:
                if rand.random()<0.5: #sample a floating point number in the range [0.0, 1.0)
                    closestNeighbors[(i,j)] = neighborNode
        if math.isinf(bestMeasure):
            GRpaths[(i,j)] = math.inf #DURING the greedy routing we reached a node from which the endpoint is not reachable - the unsuccessfulness of the greedy routing is the fault of the greedy routing
    if closestNeighbors[(i,j)]==j: #the node following node i is the endpoint
        GRpaths[(i,j)] = 1
    elif GRpaths[(closestNeighbors[(i,j)],j)]==-1: #the node following the ith node has already been visited in the current path -> we are in a loop
        GRpaths[(i,j)] = math.inf
    else:
        if math.isnan(GRpaths[(closestNeighbors[(i,j)],j)]): #the node following node i has not been visited yet and we do not know the length of the path from it to the endpoint
            greedy_routing_rec_EucDist(closestNeighbors[(i,j)], j, G, coords, measures, closestNeighbors, GRpaths)

        if math.isinf(GRpaths[(closestNeighbors[(i,j)],j)]): #the endpoint can not be reached via greedy routing from the node following node i
            GRpaths[(i,j)] = math.inf
        else: #the endpoint can be reached via greedy routing from the node following node i
            GRpaths[(i,j)] = 1 + GRpaths[(closestNeighbors[(i,j)],j)]



def greedy_routing_rec_hypDist(i, j, G, coords, measures, closestNeighbors, GRpaths, K=-1): #determine the length of the GR path from node i to node j
    GRpaths[(i,j)] = -1 #mark that the ith node (the starting node) has already been visited in the current path
    if math.isnan(closestNeighbors[(i,j)]): #we have not determined which neighbor of node i is the closest on to node j
        bestMeasure = math.inf
        for neighborNode in G.neighbors(i):
            measures[(neighborNode,j)] = hypDist(coords[neighborNode],coords[j],K) #greedy routing: always step to the node neighbour minimising this measure, i.e. to the one that is the hyperbolically closest to the destination
            if measures[(neighborNode,j)]<bestMeasure:
                closestNeighbors[(i,j)] = neighborNode
                bestMeasure = measures[(neighborNode,j)]
            elif measures[(neighborNode,j)]==bestMeasure:
                if rand.random()<0.5: #sample a floating point number in the range [0.0, 1.0)
                    closestNeighbors[(i,j)] = neighborNode
        if math.isinf(bestMeasure):
            GRpaths[(i,j)] = math.inf #DURING the greedy routing we reached a node from which the endpoint is not reachable - the unsuccessfulness of the greedy routing is the fault of the greedy routing
    if closestNeighbors[(i,j)]==j: #the node following node i is the endpoint
        GRpaths[(i,j)] = 1
    elif GRpaths[(closestNeighbors[(i,j)],j)]==-1: #the node following the ith node has already been visited in the current path -> we are in a loop
        GRpaths[(i,j)] = math.inf
    else:
        if math.isnan(GRpaths[(closestNeighbors[(i,j)],j)]): #the node following node i has not been visited yet and we do not know the length of the path from it to the endpoint
            greedy_routing_rec_hypDist(closestNeighbors[(i,j)], j, G, coords, measures, closestNeighbors, GRpaths, K)

        if math.isinf(GRpaths[(closestNeighbors[(i,j)],j)]): #the endpoint can not be reached via greedy routing from the node following node i
            GRpaths[(i,j)] = math.inf
        else: #the endpoint can be reached via greedy routing from the node following node i
            GRpaths[(i,j)] = 1 + GRpaths[(closestNeighbors[(i,j)],j)]


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
