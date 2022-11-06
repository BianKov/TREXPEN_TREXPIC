#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import math
import random as rand
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc #for computing Area Under the Curve (AUC) using the trapezoidal rule



#Testing link prediction: remove some links -> embed the resulted graph -> predict the removed links based on the node coordinates
#Testing graph reconstruction: embed the whole known graph -> remove all links -> predict the removed links based on the node coordinates



#A function for creating a list containing the randomly selected set of node pairs to be examined during the graph reconstruction.
#G is the NetworkX DiGraph (weakly connected component with only single edges and no self-loops) to be reconstructed
#numOfLinksInTheSample can be set to an integer (e.g. 100) denoting the number of actual links in the chosen set of node pairs. The default is the string 'all', meaning that we want to reconstruct all the links, therefore all node pairs will be inputted to the function that calculates the connection probabilities. Change numOfLinksInTheSample from the default value for testing graph reconstruction on large networks, for which the total number of possible pairs of vertexes would be too large to examine.
def createNodePairListForTest(G,numOfLinksInTheSample='all'):
    edgeList = list(G.edges)
    if numOfLinksInTheSample=='all': #create a list containing all the possible node pairs (thus all the connected node pairs too)
        examinedNodePairs = [] #a list consisting of (source,target) tuples of node names corresponding to the the node pairs to be examined
        for s in G:
            if G.out_degree(s)!=0: #If node s has 0 links as a source node, then no link can be predicted that point outwards from this node.
                for t in G:
                    if G.in_degree(t)!=0 and s!=t: #If node t has 0 links as a target node, then no link can be predicted that point towards this node, and self-loops are disregarded during the embedding and the graph reconstruction.
                        examinedNodePairs.append((s,t))
        listOfLinksInTheGivenSetOfNodePairs = edgeList #we want to reconstruct all the links in the network

    else: #create a list containing a random set of the node pairs containing a given number of randomly chosen links
        totalNumOfLinks = len(edgeList)
        if totalNumOfLinks<numOfLinksInTheSample:
            print('Error: The required number of links to be reconstructed ('+str(numOfLinksInTheSample)+') is larger than the total number of links ('+str(totalNumOfLinks)+').')
        else:
            #sample the given number of links to be reconstructed:
            listOfLinksInTheGivenSetOfNodePairs = rand.sample(edgeList,numOfLinksInTheSample)

            #sample non-connected node pairs:
            #the proportion of actually non-connected node pairs in the chosen set of node pairs has to be the same as in the whole set of node pairs -> denote this proportion with connectionRate
            listOfSourceNodes = []
            listOfTargetNodes = []
            numOfValidSources = 0
            numOfValidTargets = 0
            numOfNodesWithNonzeroInAndOutDegree = 0
            for nodeName in G.nodes:
                outDeg = G.out_degree(nodeName)
                inDeg = G.in_degree(nodeName)
                if outDeg!=0: #the node named nodeName has been placed as a source node in the embedding
                    numOfValidSources = numOfValidSources+1
                    listOfSourceNodes.append(nodeName)
                    if inDeg!=0:
                        numOfNodesWithNonzeroInAndOutDegree = numOfNodesWithNonzeroInAndOutDegree+1
                if inDeg!=0: #the node named nodeName has been placed as a target node in the embedding
                    numOfValidTargets = numOfValidTargets+1
                    listOfTargetNodes.append(nodeName)
            totalNumOfNodePairs = (numOfValidSources*numOfValidTargets)-numOfNodesWithNonzeroInAndOutDegree #the number of pairs of EMBEDDED nodes, without self-loops
            connectionRate = totalNumOfLinks/totalNumOfNodePairs
            numOfUnconnectionsInTheSample = int((1-connectionRate)*numOfLinksInTheSample/connectionRate) #the number of non-connected node pairs to be selected randomly
            whichUnconnectedNodePairs = rand.sample(range(totalNumOfNodePairs-totalNumOfLinks), numOfUnconnectionsInTheSample) #random integers from the interval [0 , number of non-connected node pairs] without repetition
            whichUnconnectedNodePairs.sort() #sort in increasing order
            print('Number of unconnected node pairs in the sample: '+str(len(whichUnconnectedNodePairs)))

            linkSet = set(edgeList) #use this SET in the for loop below, because 'in' is faster for sets than for lists
            randUnconnections = []
            unconnectedNodePairID = 0
            orderID = 0
            wantedNodePairID = whichUnconnectedNodePairs[orderID]
            for s in listOfSourceNodes: #If node s has 0 links as a source node, then it is not placed in the embedding as a source node, and thus no link can be predicted that point outwards from this node->do not include such nodes in the iteration over sources
                for t in listOfTargetNodes: #If node t has 0 links as a target node, then it is not placed in the embedding as a target node, and thus no link can be predicted that point towards this node->do not include such nodes in the iteration over sources
                    if s!=t and (s,t) not in linkSet: #not a self-loop and not a connected node pair
                        if unconnectedNodePairID==wantedNodePairID:
                            randUnconnections.append((s,t))
                            orderID = orderID+1
                            if orderID<numOfUnconnectionsInTheSample:
                                wantedNodePairID = whichUnconnectedNodePairs[orderID]
                            else:
                                break #jump to line 83
                        unconnectedNodePairID = unconnectedNodePairID+1
                if orderID>=numOfUnconnectionsInTheSample:
                    break #jump to line 85
            examinedNodePairs = listOfLinksInTheGivenSetOfNodePairs+randUnconnections
            rand.shuffle(examinedNodePairs)

    return [examinedNodePairs,listOfLinksInTheGivenSetOfNodePairs]





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



#This function creates the input for the graph reconstruction function in the case of hyperbolic embeddings: it returns a dictionary with the examined (source node,target node) tuples as keys and the proximity between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#Coord_source is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources in the native representation of the d-dimensional hyperbolic space to the node names.
#Coord_target is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as targets in the native representation of the d-dimensional hyperbolic space to the node names.
#K<0 is the curvature of the hyperbolic space
def linkProbability_hyp(examinedNodePairs_name,Coord_source,Coord_target,K=-1):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=1/(1+hypDist(Coord_source[sourceNodeName],Coord_target[targetNodeName],K)) #smaller hyperbolic distance=larger proximity=more probable missing link
    return linkProbabilityDict



#  Euclidean embedding  ###################################################################################################################################

#This function creates the input for the graph reconstruction function using inner products in Euclidean embeddings: it returns a dictionary with the examined (source node,target node) tuples as keys and the proximity between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#Coord_source is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources in the d-dimensional Euclidean space to the node names.
#Coord_target is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as targets in the d-dimensional Euclidean space to the node names.
def linkProbability_Euc_inner(examinedNodePairs_name,Coord_source,Coord_target):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (proximity between the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=np.inner(Coord_source[sourceNodeName],Coord_target[targetNodeName]) #larger inner product=larger proximity=more probable missing link
    return linkProbabilityDict


#This function creates the input for the graph reconstruction function using distances in Euclidean embeddings: it returns a dictionary with the examined (source node,target node) tuples as keys and the proximity between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#Coord_source is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources in the d-dimensional Euclidean space to the node names.
#Coord_target is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as targets in the d-dimensional Euclidean space to the node names.
def linkProbability_Euc_dist(examinedNodePairs_name,Coord_source,Coord_target):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (proximity between the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=1/(1+np.linalg.norm(Coord_source[sourceNodeName]-Coord_target[targetNodeName])) #smaller Euclidean distance=larger proximity=more probable missing link
    return linkProbabilityDict



#  Local methods  ###################################################################################################################################

#This function creates the input for the graph reconstruction function based on the number of common neighbors: it returns a dictionary with the examined (source node,target node) tuples as keys and the "proximity" between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#G is the examined NetworkX DiGraph
def linkProbability_commonNeighbors(examinedNodePairs_name,G):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (number of common neigbors of the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        CNset = set(G.successors(sourceNodeName)).intersection(set(G.predecessors(targetNodeName))) #set of the names of those nodes, through which node targetNodeName is reachable from node sourceNodeName in 2 steps
            #a successor of node sourceNodeName is a node succ_sourceNodeName such that there exists a directed edge from sourceNodeName to succ_sourceNodeName
            #a predecessor of node targetNodeName is a node pred_targetNodeName such that there exists a directed edge from pred_targetNodeName to targetNodeName
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=len(CNset) #the larger the number of direct paths from node sourceNodeName to node targetNodeName, the more probable that there is a link from sourceNodeName to targetNodeName
    return linkProbabilityDict


#This function creates the input for the graph reconstruction function based on the product of node degrees: it returns a dictionary with the examined (source node,target node) tuples as keys and the "proximity" between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#G is the examined NetworkX DiGraph
def linkProbability_configModel(examinedNodePairs_name,G):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (number of common neigbors of the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        linkProbabilityDict[(sourceNodeName,targetNodeName)] = G.out_degree(sourceNodeName)*G.in_degree(targetNodeName) #the larger the out-degree of the node sourceNodeName and the in-degree of the node targetNodeName, the more probable that there is a link from sourceNodeName to targetNodeName
    return linkProbabilityDict


#This function creates the input for the graph reconstruction function based on the resource allocation index written up using the out-degree of the common neighbours: it returns a dictionary with the examined (source node,target node) tuples as keys and the "proximity" between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#G is the examined NetworkX DiGraph
def linkProbability_resourceAllocation_outDeg(examinedNodePairs_name,G):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (resource allocation index for the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        CNset = set(G.successors(sourceNodeName)).intersection(set(G.predecessors(targetNodeName))) #set of the names of those nodes, through which node targetNodeName is reachable from node sourceNodeName in 2 steps
            #a successor of node sourceNodeName is a node succ_sourceNodeName such that there exists a directed edge from sourceNodeName to succ_sourceNodeName
            #a predecessor of node targetNodeName is a node pred_targetNodeName such that there exists a directed edge from pred_targetNodeName to targetNodeName
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=0 #initialization
        for CN in CNset: #iterate over the common neighbors
            linkProbabilityDict[(sourceNodeName,targetNodeName)] = linkProbabilityDict[(sourceNodeName,targetNodeName)] + (1/G.out_degree(CN))
    return linkProbabilityDict


#This function creates the input for the graph reconstruction function based on the resource allocation index written up using the in-degree of the common neighbours: it returns a dictionary with the examined (source node,target node) tuples as keys and the "proximity" between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#G is the examined NetworkX DiGraph
def linkProbability_resourceAllocation_inDeg(examinedNodePairs_name,G):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (resource allocation index for the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        CNset = set(G.successors(sourceNodeName)).intersection(set(G.predecessors(targetNodeName))) #set of the names of those nodes, through which node targetNodeName is reachable from node sourceNodeName in 2 steps
            #a successor of node sourceNodeName is a node succ_sourceNodeName such that there exists a directed edge from sourceNodeName to succ_sourceNodeName
            #a predecessor of node targetNodeName is a node pred_targetNodeName such that there exists a directed edge from pred_targetNodeName to targetNodeName
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=0 #initialization
        for CN in CNset: #iterate over the common neighbors
            linkProbabilityDict[(sourceNodeName,targetNodeName)] = linkProbabilityDict[(sourceNodeName,targetNodeName)] + (1/G.in_degree(CN))
    return linkProbabilityDict


#This function creates the input for the graph reconstruction function based on the resource allocation index written up using the degree (in+out) of the common neighbours: it returns a dictionary with the examined (source node,target node) tuples as keys and the "proximity" between the concerned nodes as values.
#examinedNodePairs_name is a list consisting of (source,target) tuples of node names corresponding to the the examined node pairs
#G is the examined NetworkX DiGraph
def linkProbability_resourceAllocation_totalDeg(examinedNodePairs_name,G):
    linkProbabilityDict={} #key=possible directed link (a (source,target) tuple), value="probability" of the given link (resource allocation index for the given two nodes)
    numOfPossibleLinks = len(examinedNodePairs_name)
    for i in range(numOfPossibleLinks):
        sourceNodeName = examinedNodePairs_name[i][0]
        targetNodeName = examinedNodePairs_name[i][1]
        CNset = set(G.successors(sourceNodeName)).intersection(set(G.predecessors(targetNodeName))) #set of the names of those nodes, through which node targetNodeName is reachable from node sourceNodeName in 2 steps
            #a successor of node sourceNodeName is a node succ_sourceNodeName such that there exists a directed edge from sourceNodeName to succ_sourceNodeName
            #a predecessor of node targetNodeName is a node pred_targetNodeName such that there exists a directed edge from pred_targetNodeName to targetNodeName
        linkProbabilityDict[(sourceNodeName,targetNodeName)]=0 #initialization
        for CN in CNset: #iterate over the common neighbors
            linkProbabilityDict[(sourceNodeName,targetNodeName)] = linkProbabilityDict[(sourceNodeName,targetNodeName)] + (1/G.degree(CN))
    return linkProbabilityDict





#  Evaluation of graph reconstruction test ################################################################################################################

#This function calculates the data needed for plotting the precision@k curve, the precision-recall curve and the receiver operating characteristic (ROC) curve for a given set of links to be reconstructed. The function also returns the number of hits when the same number of links is reconstructed as the total number of links to be reconstructed, the average precision (AP) and the area under the ROC curve (AUROC).
#linksToBeReconstructed: a list or a set of the actual links ((source node name,target node name) tuples) in the set of examined node pairs
#linkProbabilityDict: a dictionary containing the connection "probability" for each examined node pair; key=possible directed link (a (source node name,target node name) tuple), value="probability" of the given link (proximity between the given two nodes)
def precisionATk_precisionRecallCurve_ROCcurve(linksToBeReconstructed,linkProbabilityDict):
    linksInDescendingOrderOfProbability = [link for link,prob in sorted(linkProbabilityDict.items(), reverse=True, key=lambda item: item[1])]
    kList = list(range(1,len(linksInDescendingOrderOfProbability)+1))
    precisionATk = []
    truePos = 0
    for k in kList:
        if linksInDescendingOrderOfProbability[k-1] in linksToBeReconstructed:
            truePos = truePos+1
        precisionATk.append(truePos/k)
    numOfLinksToBeReconstructed = len(linksToBeReconstructed)
    precisionATnumOfLinksToBeReconstructed = precisionATk[numOfLinksToBeReconstructed-1]
    numOfHitsATnumOfLinksToBeReconstructed = int(precisionATnumOfLinksToBeReconstructed*numOfLinksToBeReconstructed)

    isConnectedOrNotList = [] #0/1 for all the examined node pairs
    proximityList = [] #proximity value for all the examined node pairs
    for link,prob in linkProbabilityDict.items():
        if link in linksToBeReconstructed:
            isConnectedOrNotList.append(1) #1=True; the currently examined node pair is connected
        else:
            isConnectedOrNotList.append(0) #0=False
        proximityList.append(prob)
    precision, recall, thresholds_pr = precision_recall_curve(isConnectedOrNotList,proximityList)
    AP = auc(recall,precision) #Computing the area under the precision-recall curve with the trapezoidal rule (using linear interpolation).
    falsePositiveRate, truePositiveRate, thresholds_ROC = roc_curve(isConnectedOrNotList,proximityList,drop_intermediate=False)
    AUROC = auc(falsePositiveRate,truePositiveRate) #Computing the area under the ROC curve with the trapezoidal rule (using linear interpolation).
    return [kList,precisionATk,precisionATnumOfLinksToBeReconstructed,numOfHitsATnumOfLinksToBeReconstructed,recall,precision,AP,falsePositiveRate,truePositiveRate,AUROC]

