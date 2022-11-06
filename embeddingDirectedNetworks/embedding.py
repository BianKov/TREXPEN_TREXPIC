#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
from scipy.sparse.linalg import svds
from numpy import linalg as LA #for calculating the spectral radius of the adjacency matrix in the variants of HOPE



#A function for loading the directed edge list of the network to be embedded. Edge weights are disregarded, self-loops are removed, multi-edges are converted to single edges, and only the largest weakly connected component is returned as a NetworkX DiGraph.
#path is a string with the path of the text file containing the edge list to be loaded
    #In the edge list each line has to correspond to one connected node pair with the identifier of the edge's source node in the first column and the identifier of the edge's target node in the second column. Additional columns will be disregarded, each edge will have weight 1.
#skipRows is the number of lines to be skipped at the beginning of the text file containing the edge list; the default is 0
#delimiter is the string used to separate the columns in the text file to be loaded; the default is "\t"
#Example for function call:
#   G=embedding.loadGraph(os.getcwd()+"/"+directoryName+"/edgeList.txt",1,"\t")
def loadGraph(path,skipRows=0,delimiter="\t"):
    edgeList = [] #initialize the list of the (source node identifier,target node identifier) edge tuples
    fileHandler = open(path,"r")
    for l in range(skipRows):
        line = fileHandler.readline()
    while True:
        line = fileHandler.readline() #get the next line from file
        if not line: #line is empty (end of the file)
            break;
        listOfWords = line.split(delimiter)
        sourceNodeID = listOfWords[0] #string from the first column as the identifier of the source node
        if listOfWords[1][-1]=="\n": #the second column is the last in the currently loaded line
            targetNodeID = listOfWords[1][:-1] #string from the second column without "\n" as the identifier of the target node
        else: #there are more than two columns in the currently loaded line
            targetNodeID = listOfWords[1] #string from the second column as the identifier of the target node
        if sourceNodeID != targetNodeID: #the self-loops are disregarded
            edgeList.append((sourceNodeID,targetNodeID))
    fileHandler.close()

    G_total = nx.DiGraph()
    G_total.add_edges_from(edgeList) #multi-edges are automatically converted to single edges
    #extract the largest weakly connected component:
    G=max([G_total.subgraph(comp).copy() for comp in nx.weakly_connected_components(G_total)],key=len) #.copy(): create a subgraph with its own copy of the edge/node attributes -> changes to attributes in the subgraph are NOT reflected in the original graph; without copy the subgraph is a frozen graph for which edges can not be added or removed

    return G





#At https://www.nature.com/articles/s41467-017-01825-5 different pre-weighting strategies are described for undirected networks that facilitated the estimation of the spatial arrangement of the nodes. This function performs the analogous link weighting on directed network. Larger weight corresponds to less similarity.
#G is the NetworkX DiGraph to be pre-weighted. Note that the weights in G do not change, the function returns a weighted copy of G.
#weightingType is a string corresponding to the name of the pre-weighting rule to be used. Can be set to 'RA1', 'RA2', 'RA3', 'RA4' or 'EBC'.
#Example for function call:
#   G_weighted = embedding.preWeighting(G,'RA1')
def preWeighting(G,weightingType):
    G_weighted = nx.DiGraph()
    G_weighted.add_nodes_from(G.nodes) #keep the node order of the original graph
    if weightingType=='RA1':
        for (i,j) in G.edges(): #(i,j)=a tuple, which denotes the edge from node i to node j
            CNset = set(G.successors(i)).intersection(set(G.predecessors(j))) #set of the identifiers of those nodes, through which node j is reachable from node i in 2 steps
                #a successor of node i is a node succ_i such that there exists a directed edge from i to succ_i
                #a predecessor of node j is a node pred_j such that there exists a directed edge from pred_j to j
            CN = len(CNset) #number of direct paths from node i to node j
            #assign a weight to the i->j edge:
            w = (G.out_degree(i) + G.in_degree(j) + G.out_degree(i) * G.in_degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA2':
        for (i,j) in G.edges(): #(i,j)=a tuple, which denotes the edge from node i to node j
            CNset = set(G.successors(i)).intersection(set(G.predecessors(j))) #set of the identifiers of those nodes, through which node j is reachable from node i in 2 steps
            CN = len(CNset) #number of direct paths from node i to node j

            #ei=the external degree of the node i with respect to node j,
            #i.e. the number of links from i to neither j nor the common neighbors with j,
            #i.e. the number of i's successors without node j and the common neighbors with j,
            #where the common neighbors are those nodes, which belong to the successors of node i and the predecessors of node j at the same time
            ei = len(set(G.successors(i)) - {j} - CNset)

            #ej=the external degree of the node j with respect to node i,
            #i.e. the number of links to j from neither i nor the common neighbors with i,
            #i.e. the number of j's predecessors without node i and the common neighbors with i,
            #where the common neighbors are those nodes, which belong to the successors of node i and the predecessors of node j at the same time
            ej = len(set(G.predecessors(j)) - {i} - CNset)

            #assign a weight to the i->j edge:
            w = (1 + ei + ej + ei * ej) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA3':
        for (i,j) in G.edges(): #(i,j)=a tuple, which denotes the edge from node i to node j
            CNset = set(G.successors(i)).intersection(set(G.predecessors(j))) #set of the identifiers of those nodes, through which node j is reachable from node i in 2 steps
            CN = len(CNset) #number of direct paths from node i to node j
            #assign a weight to the i->j edge:
            w = (G.out_degree(i) + G.in_degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA4':
        for (i,j) in G.edges(): #(i,j)=a tuple, which denotes the edge from node i to node j
            CNset = set(G.successors(i)).intersection(set(G.predecessors(j))) #set of the identifiers of those nodes, through which node j is reachable from node i in 2 steps
            CN = len(CNset) #number of direct paths from node i to node j
            #assign a weight to the i->j edge:
            w = (G.out_degree(i) * G.in_degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='EBC': #use the edge betweenness centrality
        edgeWeightDict = {(i,j):0 for (i,j) in G.edges()} #initialization
        for s in G:
            if G.out_degree(s)!=0:
                for t in G:
                    if t!=s and G.in_degree(t)!=0:
                        try:
                            SPlist = [p for p in nx.all_shortest_paths(G,source=s,target=t)] #default: weight=None, i.e. every edge has weight/distance/cost 1
                        except nx.NetworkXNoPath:
                            continue #move on to the next target node
                        totalNumOfSPLs_st = len(SPlist) #number of shortest paths from node s to node t
                        for p in SPlist: #iterate over the list of shortest paths from node s to node t (each path is a list of nodes following each other in the path)
                            currentNode = p[0]
                            for nextNode in p[1:]:
                                edgeWeightDict[(currentNode,nextNode)]=edgeWeightDict[(currentNode,nextNode)]+1/totalNumOfSPLs_st
                                currentNode = nextNode
        for (i,j),w_ij in edgeWeightDict.items():
            G_weighted.add_edge(i,j,weight=w_ij)
    else:
        print('False parameter: weightingType\n')
    return G_weighted







#A function for creating a d-dimensional Euclidean embedding with TREXPEN.
#G is the (pre-weighted) NetworkX DiGraph to be embedded
#d is the dimension of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (e.g. q=0.001 keeps all the proximity values close to 1 (except the 0 values, which correspond to infinitely large distances along the graph), while q=1000 keeps all the proximity values close to 0 (except the 1 values, which correspond to paths of 0 length))
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.TREXPEN(G,d)
def TREXPEN(G,d,q=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN can not be larger than the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight='weight'))
        #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node

    if q==None: #use the default setting of the multiplying factor
        listOfAllSPLs = []
        for sour in shortestPathLengthsDict.keys():
            for targ in shortestPathLengthsDict[sour].keys():
                listOfAllSPLs.append(shortestPathLengthsDict[sour][targ])
        maxSPL = max(listOfAllSPLs)
        qmin = math.log(1.0/0.9)/maxSPL
        qmax = math.log(math.pow(10,50))/maxSPL
        q = math.exp((math.log(qmin)+math.log(qmax))/2)
        print('The default q multiplying factor is '+str(q)+'.')

    P = np.zeros((N,N)) #a matrix with the node-node proximities; P[i,j] is the proximity calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to G.nodes
    rowID = 0
    for s in G: #iterate over the nodes as sources
        colID = 0
        for t in G: #iterate over all the nodes as targets
            try:
                matrixElement = math.exp(-q*shortestPathLengthsDict[s][t])
            except KeyError: #there is no path from node s to node t -> set the proximity to the possible lowest value (SPL=infinity -> exp(-q*SPL)=0)
                matrixElement = 0.0
            P[rowID,colID] = matrixElement
            colID = colID+1
        rowID = rowID+1

    #dimension reduction
    if d == N: #there is no dimension reduction
        U,S,VT=np.linalg.svd(P) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N: only the first d singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(P,d,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN can not be larger than the number of positive singular values of the proximity matrix, i.e. '+str(numOfPositiveSingularValues)+'.\n\n')
    Ssqrt = np.diag(np.sqrt(S))

    #create the matrixes from which the node coordinates will be obtained
    CoordMatrix_source=np.matmul(U,Ssqrt) #the jth column contains the jth source coordinate of all nodes in the reduced space
    if isTheAdjMatrixSymmetric:
        CoordMatrix_target=CoordMatrix_source #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
    else:
        CoordMatrix_target=np.transpose(np.matmul(Ssqrt,VT)) #the jth column contains the jth target coordinate of all nodes in the reduced space

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            Coord_source[nodeName] = CoordMatrix_source[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a source node in the d-dimensional Euclidean space
        if G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            Coord_target[nodeName] = CoordMatrix_target[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a target node in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1
    
    return [Coord_source,Coord_target]





#A function for creating a d-dimensional Euclidean embedding with TREXPEN-R.
#G is the (pre-weighted) NetworkX DiGraph to be embedded
#d is the dimension of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (e.g. q=0.001 keeps all the proximity values close to 1 (except the 0 values, which correspond to infinitely large distances along the graph), while q=1000 keeps all the proximity values close to 0 (except the 1 values, which correspond to paths of 0 length))
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.TREXPEN_R(G,d)
def TREXPEN_R(G,d,q=None):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN_R can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight='weight'))
        #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node

    if q==None: #use the default setting of the multiplying factor
        listOfAllSPLs = []
        for sour in shortestPathLengthsDict.keys():
            for targ in shortestPathLengthsDict[sour].keys():
                listOfAllSPLs.append(shortestPathLengthsDict[sour][targ])
        maxSPL = max(listOfAllSPLs)
        qmin = math.log(1.0/0.9)/maxSPL
        qmax = math.log(math.pow(10,50))/maxSPL
        q = math.exp((math.log(qmin)+math.log(qmax))/2)
        print('The default q multiplying factor is '+str(q)+'.')

    P = np.zeros((N,N)) #a matrix with the node-node proximities; P[i,j] is the proximity calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to G.nodes
    rowID = 0
    for s in G: #iterate over the nodes as sources
        colID = 0
        for t in G: #iterate over all the nodes as targets
            try:
                matrixElement = math.exp(-q*shortestPathLengthsDict[s][t])
            except KeyError: #there is no path from node s to node t -> set the proximity to the possible lowest value (SPL=infinity -> exp(-q*SPL)=0)
                matrixElement = 0.0
            P[rowID,colID] = matrixElement
            colID = colID+1
        rowID = rowID+1

    #dimension reduction
    if d == N-1:
        U,S,VT=np.linalg.svd(P) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N-1: only the first d+1 singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(P,d+1,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN_R can not be larger than the number of positive singular values of the proximity matrix-1, i.e. '+str(numOfPositiveSingularValues-1)+'.\n\n')
    Ssqrt = np.diag(np.sqrt(S))
    
    #create the matrixes from which the node coordinates will be obtained
    CoordMatrix_source=np.matmul(U,Ssqrt) #the jth column contains the j-1th source coordinate of all nodes in the reduced space; the 0th column will be discarded in the absence of centering
    if isTheAdjMatrixSymmetric:
        CoordMatrix_target=CoordMatrix_source #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
    else:
        CoordMatrix_target=np.transpose(np.matmul(Ssqrt,VT)) #the jth column contains the j-1th target coordinate of all nodes in the reduced space; the 0th column is discarded in the absence of centering

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            Coord_source[nodeName] = CoordMatrix_source[nodeIndex,1:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a source node in the d-dimensional Euclidean space
        if G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            Coord_target[nodeName] = CoordMatrix_target[nodeIndex,1:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a target node in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1
    
    return [Coord_source,Coord_target]






#A function for creating a d-dimensional Euclidean embedding with TREXPEN-S.
#G is the (pre-weighted) NetworkX DiGraph to be embedded
#d is the dimension of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (e.g. q=0.001 keeps all the proximity values close to 1 (except the 0 values, which correspond to infinitely large distances along the graph), while q=1000 keeps all the proximity values close to 0 (except the 1 values, which correspond to paths of 0 length))
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.TREXPEN_S(G,d)
def TREXPEN_S(G,d,q=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN_S can not be larger than the number of nodes the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight='weight'))
        #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node

    if q==None: #use the default setting of the multiplying factor
        listOfAllSPLs = []
        for sour in shortestPathLengthsDict.keys():
            for targ in shortestPathLengthsDict[sour].keys():
                listOfAllSPLs.append(shortestPathLengthsDict[sour][targ])
        maxSPL = max(listOfAllSPLs)
        qmin = math.log(1.0/0.9)/maxSPL
        qmax = math.log(math.pow(10,50))/maxSPL
        q = math.exp((math.log(qmin)+math.log(qmax))/2)
        print('The default q multiplying factor is '+str(q)+'.')

    P = np.zeros((N,N)) #a matrix with the node-node proximities; P[i,j] is the proximity calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to G.nodes
    rowID = 0
    for s in G: #iterate over the nodes as sources
        colID = 0
        for t in G: #iterate over all the nodes as targets
            try:
                matrixElement = math.exp(-q*shortestPathLengthsDict[s][t])
            except KeyError: #there is no path from node s to node t -> set the proximity to the possible lowest value (SPL=infinity -> exp(-q*SPL)=0)
                matrixElement = 0.0
            P[rowID,colID] = matrixElement
            colID = colID+1
        rowID = rowID+1
    P = P-np.mean(P) #set the average of all the matrix elements to 0 ("centering")

    #dimension reduction
    if d == N: #there is no dimension reduction
        U,S,VT=np.linalg.svd(P) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N: only the first d singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(P,d,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN can not be larger than the number of positive singular values of the proximity matrix, i.e. '+str(numOfPositiveSingularValues)+'.\n\n')
    Ssqrt = np.diag(np.sqrt(S))

    #create the matrixes from which the node coordinates will be obtained
    CoordMatrix_source=np.matmul(U,Ssqrt) #the jth column contains the jth source coordinate of all nodes in the reduced space
    if isTheAdjMatrixSymmetric:
        CoordMatrix_target=CoordMatrix_source #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
    else:
        CoordMatrix_target=np.transpose(np.matmul(Ssqrt,VT)) #the jth column contains the jth target coordinate of all nodes in the reduced space

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            Coord_source[nodeName] = CoordMatrix_source[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a source node in the d-dimensional Euclidean space
        if G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            Coord_target[nodeName] = CoordMatrix_target[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a target node in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return [Coord_source,Coord_target]






#A function for performing High-Order Proximity preserved Embedding (HOPE) (SVD of Katz proximity matrix).
#G is the NetworkX DiGraph to be embedded (all the link weights will be considered to be 1; the here-applied formula of the Katz matrix is not valid for weighted networks)
#d is the number of dimensions of the Euclidean embedding space
#alpha is the decaying constant; the default is None, in this case alpha will be set to 0.5/spectral radius (its value will be printed)
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.    
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.HOPE(G,d)
def HOPE(G,d,alpha=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE can not be larger than the number of nodes the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight=None) #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes().
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    if alpha==None: #use the default setting of the decay parameter
        spectralRadius = np.amax(np.abs(LA.eigvals(A_np))) #the spectral radius of a square matrix is the largest absolute value of its eigenvalues
        alpha = 0.5/spectralRadius
        print('The default alpha decay parameter is '+str(alpha)+'.')
    #Katz proximity matrix = (I - alpha*A)^-1 * alpha*A = (I - alpha*A)^-1 * (I - (I -alpha*A)) = (I - alpha*A)^-1 * I  -  (I - alpha*A)^-1 * (I -alpha*A) = (I - alpha*A)^-1 - I for unweighted networks
    P = np.linalg.inv(np.eye(N)-alpha*A_np) - np.eye(N)

    #dimension reduction
    if d == N: #there is no dimension reduction
        U,S,VT=np.linalg.svd(P) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N: only the first d singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(P,d,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE can not be larger than the number of positive singular values of the proximity matrix, i.e. '+str(numOfPositiveSingularValues)+'.\n\n')
    Ssqrt = np.diag(np.sqrt(S))

    #create the matrixes from which the node coordinates will be obtained
    CoordMatrix_source=np.matmul(U,Ssqrt) #the jth column contains the jth source coordinate of all nodes in the reduced space
    if isTheAdjMatrixSymmetric:
        CoordMatrix_target=CoordMatrix_source #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
    else:
        CoordMatrix_target=np.transpose(np.matmul(Ssqrt,VT)) #the jth column contains the jth target coordinate of all nodes in the reduced space

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            Coord_source[nodeName] = CoordMatrix_source[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a source node in the d-dimensional Euclidean space
        if G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            Coord_target[nodeName] = CoordMatrix_target[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a target node in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return [Coord_source,Coord_target]





#A function for performing High-Order Proximity preserved Embedding (HOPE) (SVD of Katz proximity matrix) with the neglection of the first dimension of the embedding.
#G is the NetworkX DiGraph to be embedded (all the link weights will be considered to be 1; the here-applied formula of the Katz matrix is not valid for weighted networks)
#d is the number of dimensions of the Euclidean embedding space
#alpha is the decaying constant; the default is None, in this case alpha will be set to 0.5/spectral radius (its value will be printed)
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.    
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.HOPE_R(G,d)
def HOPE_R(G,d,alpha=None):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE_R can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight=None) #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes().
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    if alpha==None: #use the default setting of the decay parameter
        spectralRadius = np.amax(np.abs(LA.eigvals(A_np))) #the spectral radius of a square matrix is the largest absolute value of its eigenvalues
        alpha = 0.5/spectralRadius
        print('The default alpha decay parameter is '+str(alpha)+'.')
    #Katz proximity matrix = (I - alpha*A)^-1 * alpha*A = (I - alpha*A)^-1 * (I - (I -alpha*A)) = (I - alpha*A)^-1 * I  -  (I - alpha*A)^-1 * (I -alpha*A) = (I - alpha*A)^-1 - I for unweighted networks
    P = np.linalg.inv(np.eye(N)-alpha*A_np) - np.eye(N)

    #dimension reduction
    if d == N-1:
        U,S,VT=np.linalg.svd(P) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N-1: only the first d+1 singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(P,d+1,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE_R can not be larger than the number of positive singular values of the proximity matrix-1, i.e. '+str(numOfPositiveSingularValues-1)+'.\n\n')
    Ssqrt = np.diag(np.sqrt(S))
    
    #create the matrixes from which the node coordinates will be obtained
    CoordMatrix_source=np.matmul(U,Ssqrt) #the jth column contains the j-1th source coordinate of all nodes in the reduced space; the 0th column will be discarded in the absence of centering
    if isTheAdjMatrixSymmetric:
        CoordMatrix_target=CoordMatrix_source #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
    else:
        CoordMatrix_target=np.transpose(np.matmul(Ssqrt,VT)) #the jth column contains the j-1th target coordinate of all nodes in the reduced space; the 0th column is discarded in the absence of centering

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            Coord_source[nodeName] = CoordMatrix_source[nodeIndex,1:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a source node in the d-dimensional Euclidean space
        if G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            Coord_target[nodeName] = CoordMatrix_target[nodeIndex,1:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a target node in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return [Coord_source,Coord_target]





#A function for performing High-Order Proximity preserved Embedding (HOPE) (SVD of Katz proximity matrix) with shifting the mean of all matrix elements to 0.
#G is the NetworkX DiGraph to be embedded (all the link weights will be considered to be 1; the here-applied formula of the Katz matrix is not valid for weighted networks)
#d is the number of dimensions of the Euclidean embedding space
#alpha is the decaying constant; the default is None, in this case alpha will be set to 0.5/spectral radius (its value will be printed)
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.    
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.HOPE_S(G,d)
def HOPE_S(G,d,alpha=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE_S can not be larger than the number of nodes the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight=None) #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes().
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    if alpha==None: #use the default setting of the decay parameter
        spectralRadius = np.amax(np.abs(LA.eigvals(A_np))) #the spectral radius of a square matrix is the largest absolute value of its eigenvalues
        alpha = 0.5/spectralRadius
        print('The default alpha decay parameter is '+str(alpha)+'.')
    #Katz proximity matrix = (I - alpha*A)^-1 * alpha*A = (I - alpha*A)^-1 * (I - (I -alpha*A)) = (I - alpha*A)^-1 * I  -  (I - alpha*A)^-1 * (I -alpha*A) = (I - alpha*A)^-1 - I for unweighted networks
    P = np.linalg.inv(np.eye(N)-alpha*A_np) - np.eye(N)

    P = P-np.mean(P) #set the average of all the matrix elements to 0 ("centering")

    #dimension reduction
    if d == N: #there is no dimension reduction
        U,S,VT=np.linalg.svd(P) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N: only the first d singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(P,d,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE_S can not be larger than the number of positive singular values of the proximity matrix, i.e. '+str(numOfPositiveSingularValues)+'.\n\n')
    Ssqrt = np.diag(np.sqrt(S))

    #create the matrixes from which the node coordinates will be obtained
    CoordMatrix_source=np.matmul(U,Ssqrt) #the jth column contains the jth source coordinate of all nodes in the reduced space
    if isTheAdjMatrixSymmetric:
        CoordMatrix_target=CoordMatrix_source #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
    else:
        CoordMatrix_target=np.transpose(np.matmul(Ssqrt,VT)) #the jth column contains the jth target coordinate of all nodes in the reduced space

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            Coord_source[nodeName] = CoordMatrix_source[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a source node in the d-dimensional Euclidean space
        if G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            Coord_target[nodeName] = CoordMatrix_target[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) as a target node in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return [Coord_source,Coord_target]











#A function for shifting the center of mass of a d-dimensional Euclidean embedding to the origin.
#Coord_source_Euc_original is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources in the d-dimensional Euclidean space to the node names.
#Coord_target_Euc_original is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as targets in the d-dimensional Euclidean space to the node names.
#d is the number of dimensions of the space to which the network is embedded
#The function returns the dictionaries Coord_source_Euc_shifted and Coord_target_Euc_shifted assigning NumPy arrays of d elements containing the shifted Cartesian coordinates of the network nodes as sources and targets in the d-dimensional Euclidean space to the node names.    
def shiftCOMtoOrigin(Coord_source_Euc_original,Coord_target_Euc_original,d):
    #determine the position of the center of mass
    meanOfCoords=np.zeros(d)
    for nodeName in Coord_source_Euc_original.keys():
        meanOfCoords = meanOfCoords+Coord_source_Euc_original[nodeName]
    for nodeName in Coord_target_Euc_original.keys():
        meanOfCoords = meanOfCoords+Coord_target_Euc_original[nodeName]
    meanOfCoords = meanOfCoords/(len(Coord_source_Euc_original)+len(Coord_target_Euc_original))
    #create the dictionaries of the SHIFTED source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as a source/as a target
    Coord_source_Euc_shifted = {}
    for nodeName in Coord_source_Euc_original.keys():
        Coord_source_Euc_shifted[nodeName] = Coord_source_Euc_original[nodeName]-meanOfCoords
    Coord_target_Euc_shifted = {}
    for nodeName in Coord_target_Euc_original.keys():
        Coord_target_Euc_shifted[nodeName] = Coord_target_Euc_original[nodeName]-meanOfCoords
    return [Coord_source_Euc_shifted,Coord_target_Euc_shifted]







#A function for performing the model-independent conversion MIC, i.e. for converting a d-dimensional Euclidean embedding of a graph to a hyperbolic embedding by changing its radial node arrangement.
#N is the number of nodes in the embedded graph
#Coord_source_Euc is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources in the d-dimensional Euclidean space to the node names.
#Coord_target_Euc is a dictionary assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as targets in the d-dimensional Euclidean space to the node names.
#d is the number of dimensions of the space to which the network is embedded
#C is a multiplying factor in the desired largest hyperbolic radial coordinate rhmax_source=rhmax_target=(C/zeta)*ln(N), where zeta=sqrt(-K). The volume of the hyperbolic sphere inside the outermost node will be proportional to N^(C*(d-1)) for not too small networks.
#K<0 is the curvature of the hyperbolic space
#The function returns the dictionaries Coord_source_hyp and Coord_target_hyp assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the native representation of the d-dimensional hyperbolic space to the node names.
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source_hyp and nodes with 0 out-degree appear only in the key list of Coord_target_hyp.
#Example for function call:
#   [Coord_source_hyp,Coord_target_hyp]=embedding.convertEucToHyp(G,Coord_source_Euc,Coord_target_Euc,d)
def convertEucToHyp(N,Coord_source_Euc,Coord_target_Euc,d,C=2,K=-1):
    zeta = math.sqrt(-K)
     
    #identify the radial intervals of the Euclidean embedding
    norm_s_Euc_dict = {nodeName:np.linalg.norm(Coord_source_Euc[nodeName]) for nodeName in Coord_source_Euc.keys()} #dictionary with the lengths of the Euclidean source position vectors
    r_E_min_s = min(norm_s_Euc_dict.values())
    r_E_max_s = max(norm_s_Euc_dict.values())
    norm_t_Euc_dict = {nodeName:np.linalg.norm(Coord_target_Euc[nodeName]) for nodeName in Coord_target_Euc.keys()} #dictionary with the lengths of the Euclidean target position vectors
    r_E_min_t = min(norm_t_Euc_dict.values())
    r_E_max_t = max(norm_t_Euc_dict.values())
    
    #calculate the hyperbolic source positions
    Coord_source_hyp = {}
    if r_E_min_s==0: #the Euclidean embedding method placed at least one source node in the origin, which is a position that does not have an obvious hyperbolic equivalent
        increasingEucSourceRadCoords = sorted(norm_s_Euc_dict.values()) #list of source radial coordinates sorted in an increasing order
        i = 0
        while increasingEucSourceRadCoords[i]==0:
            i = i+1
        r_E_nonZeroMin_s = increasingEucSourceRadCoords[i] #the occurring smallest non-zero Euclidean source radial coordinate in the embedded graph
        r_h_max_s = C*math.log(N)/zeta #calculate the largest hyperbolic radial coordinate obtained among the source nodes that were not in the origin in the Euclidean case
        numOfEucZeros_s = 0
        for nodeName in Coord_source_Euc.keys(): #iterate over those nodes that have a position as a source
            if norm_s_Euc_dict[nodeName]==0: #the node named nodeName was placed in the origin in the Euclidean embedding -> estimate its hyperbolic position
                numOfEucZeros_s = numOfEucZeros_s+1
                Coord_source_hyp[nodeName] = np.zeros(d) #initialization
                while np.linalg.norm(Coord_source_hyp[nodeName])<0.0001: #in order to avoid problems with floating point precision when normalizing the array
                    Coord_source_hyp[nodeName] = np.random.standard_normal(d) #the hyperbolic angular coordinates of nodes that were placed in the origin in the Euclidean embedding are set to random values
                Coord_source_hyp[nodeName] = 10*r_h_max_s*Coord_source_hyp[nodeName]/np.linalg.norm(Coord_source_hyp[nodeName]) #the hyperbolic radial coordinate of nodes that were placed in the origin in the Euclidean embedding is set to an extremely large value, namely 10*the largest hyperbolic r obtained among the source nodes that were not in the origin in the Euclidean case
            else: #the node named nodeName was placed out of the origin in the Euclidean embedding -> calculate its hyperbolic position as usual
                try:
                    r_source_hyp = math.log(1+(math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName],d))/(zeta*(d-1))
                    if r_source_hyp==0: #a rounding error has occurred, because the term (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName],d) is too small compared to the +1 in the logarithm
                        r_source_hyp = (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                except OverflowError: #math.pow(N,C*(d-1)) is too large
                    try:
                        r_source_hyp = math.log(1+math.pow(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName],d))/(zeta*(d-1))
                        if r_source_hyp==0: #a rounding error has occurred
                            r_source_hyp = math.pow(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                    except OverflowError: #the dth power is still too large
                        r_source_hyp = math.log(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName])*d/(zeta*(d-1))
                        #this can not become 0
                Coord_source_hyp[nodeName] = r_source_hyp*Coord_source_Euc[nodeName]/norm_s_Euc_dict[nodeName] #the Cartesian coordinates of the node named nodeName as a source node in the native representation of the d-dimensional hyperbolic space
        print('The Euclidean embedding method placed '+str(numOfEucZeros_s)+' source nodes in the origin. In the hyperbolic embedding, the angular coordinates of these nodes are set to random values, and their hyperbolic radial coordinate is set to an extremely large value, namely 10*the largest hyperbolic r obtained among the source nodes that were not in the origin in the Euclidean case. Consider changing the q or alpha parameter of the Euclidean embedding!')
    else: #the Euclidean embedding placed all the source nodes out of the origin
        for nodeName in Coord_source_Euc.keys(): #iterate over those nodes that have a position as a source
            try:
                r_source_hyp = math.log(1+(math.pow(N,C*(d-1))-1)*math.pow(r_E_min_s/norm_s_Euc_dict[nodeName],d))/(zeta*(d-1))
                if r_source_hyp==0: #a rounding error has occurred, because the term (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_s/norm_s_Euc_dict[nodeName],d) is too small compared to the +1 in the logarithm
                    r_source_hyp = (math.pow(N,C*(d-1))-1)*math.pow(r_E_min_s/norm_s_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
            except OverflowError: #math.pow(N,C*(d-1)) is too large
                try:
                    r_source_hyp = math.log(1+math.pow(math.pow(N,C*(d-1)/d)*r_E_min_s/norm_s_Euc_dict[nodeName],d))/(zeta*(d-1))
                    if r_source_hyp==0: #a rounding error has occurred
                        r_source_hyp = math.pow(math.pow(N,C*(d-1)/d)*r_E_min_s/norm_s_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                except OverflowError: #the dth power is still too large
                    r_source_hyp = math.log(math.pow(N,C*(d-1)/d)*r_E_min_s/norm_s_Euc_dict[nodeName])*d/(zeta*(d-1))
                    #this can not become 0
            Coord_source_hyp[nodeName] = r_source_hyp*Coord_source_Euc[nodeName]/norm_s_Euc_dict[nodeName] #the Cartesian coordinates of the node named nodeName as a source node in the native representation of the d-dimensional hyperbolic space

    #calculate the hyperbolic target positions
    Coord_target_hyp = {}
    if r_E_min_t==0: #the Euclidean embedding method placed at least one target node in the origin, which is a position that does not have an obvious hyperbolic equivalent
        increasingEucTargetRadCoords = sorted(norm_t_Euc_dict.values()) #list of target radial coordinates sorted in an increasing order
        i = 0
        while increasingEucTargetRadCoords[i]==0:
            i = i+1
        r_E_nonZeroMin_t = increasingEucTargetRadCoords[i] #the occurring smallest non-zero Euclidean target radial coordinate in the embedded graph
        r_h_max_t = C*math.log(N)/zeta #calculate the largest hyperbolic radial coordinate obtained among the target nodes that were not in the origin in the Euclidean case
        numOfEucZeros_t = 0
        for nodeName in Coord_target_Euc.keys(): #iterate over those nodes that have a position as a target
            if norm_t_Euc_dict[nodeName]==0: #the node named nodeName was placed in the origin in the Euclidean embedding -> estimate its hyperbolic position
                numOfEucZeros_t = numOfEucZeros_t+1
                Coord_target_hyp[nodeName] = np.zeros(d) #initialization
                while np.linalg.norm(Coord_target_hyp[nodeName])<0.0001: #in order to avoid problems with floating point precision when normalizing the array
                    Coord_target_hyp[nodeName] = np.random.standard_normal(d) #the hyperbolic angular coordinates of nodes that were placed in the origin in the Euclidean embedding are set to random values
                Coord_target_hyp[nodeName] = 10*r_h_max_t*Coord_target_hyp[nodeName]/np.linalg.norm(Coord_target_hyp[nodeName]) #the hyperbolic radial coordinate of nodes that were placed in the origin in the Euclidean embedding is set to an extremely large value, namely 10*the largest hyperbolic r obtained among the target nodes that were not in the origin in the Euclidean case
            else: #the node named nodeName was placed out of the origin in the Euclidean embedding -> calculate its hyperbolic position as usual
                try:
                    r_target_hyp = math.log(1+(math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName],d))/(zeta*(d-1))
                    if r_target_hyp==0: #a rounding error has occurred, because the term (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName],d) is too small compared to the +1 in the logarithm
                        r_target_hyp = (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                except OverflowError: #math.pow(N,C*(d-1)) is too large
                    try:
                        r_target_hyp = math.log(1+math.pow(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName],d))/(zeta*(d-1))
                        if r_target_hyp==0: #a rounding error has occurred
                            r_target_hyp = math.pow(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                    except OverflowError: #the dth power is still too large
                        r_target_hyp = math.log(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName])*d/(zeta*(d-1))
                        #this can not become 0
                Coord_target_hyp[nodeName] = r_target_hyp*Coord_target_Euc[nodeName]/norm_t_Euc_dict[nodeName] #the Cartesian coordinates of the node named nodeName as a target node in the native representation of the d-dimensional hyperbolic space
        print('The Euclidean embedding method placed '+str(numOfEucZeros_t)+' target nodes in the origin. In the hyperbolic embedding, the angular coordinates of these nodes are set to random values, and their hyperbolic radial coordinate is set to an extremely large value, namely 10*the largest hyperbolic r obtained among the target nodes that were not in the origin in the Euclidean case. Consider changing the q or alpha parameter of the Euclidean embedding!')
    else: #the Euclidean embedding placed all the target nodes out of the origin
        for nodeName in Coord_target_Euc.keys(): #iterate over those nodes that have a position as a target
            try:
                r_target_hyp = math.log(1+(math.pow(N,C*(d-1))-1)*math.pow(r_E_min_t/norm_t_Euc_dict[nodeName],d))/(zeta*(d-1))
                if r_target_hyp==0: #a rounding error has occurred, because the term (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin_t/norm_t_Euc_dict[nodeName],d) is too small compared to the +1 in the logarithm
                    r_target_hyp = (math.pow(N,C*(d-1))-1)*math.pow(r_E_min_t/norm_t_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
            except OverflowError: #math.pow(N,C*(d-1)) is too large
                try:
                    r_target_hyp = math.log(1+math.pow(math.pow(N,C*(d-1)/d)*r_E_min_t/norm_t_Euc_dict[nodeName],d))/(zeta*(d-1))
                    if r_target_hyp==0: #a rounding error has occurred
                        r_target_hyp = math.pow(math.pow(N,C*(d-1)/d)*r_E_min_t/norm_t_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                except OverflowError: #the dth power is still too large
                    r_target_hyp = math.log(math.pow(N,C*(d-1)/d)*r_E_min_t/norm_t_Euc_dict[nodeName])*d/(zeta*(d-1))
                    #this can not become 0
            Coord_target_hyp[nodeName] = r_target_hyp*Coord_target_Euc[nodeName]/norm_t_Euc_dict[nodeName] #the Cartesian coordinates of the node named nodeName as a target node in the native representation of the d-dimensional hyperbolic space

    return [Coord_source_hyp,Coord_target_hyp]
















#A function for creating a d-dimensional hyperbolic embedding with TREXPIC.
#G is the (pre-weighted) NetworkX DiGraph to be embedded
#d is the number of dimensions of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0)
#K<0 is the curvature of the hyperbolic space
#The function returns the dictionaries Coord_source and Coord_target assigning NumPy arrays of d elements containing the Cartesian coordinates of the network nodes as sources and targets in the native representation of the d-dimensional hyperbolic space to the node names.
    #Note that nodes with 0 in-degree appear only in the key list of Coord_source and nodes with 0 out-degree appear only in the key list of Coord_target.
#Example for function call:
#   [Coord_source,Coord_target]=embedding.TREXPIC(G,d)
def TREXPIC(G,d,q=None,K=-1):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    zeta = math.sqrt(-K)

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.
    #test whether the adjacency matrix is symmetric
    sumOfDifferences = 0
    for i in range(N):
        for j in range(N):
            sumOfDifferences = sumOfDifferences+abs(A_np[i,j]-A_np[j,i])
    if sumOfDifferences<1e-08:
        isTheAdjMatrixSymmetric = True
    else:
        isTheAdjMatrixSymmetric = False
    #a rather memory-consuming version: isTheAdjMatrixSymmetric = np.allclose(A_np, A_np.T, rtol=1e-08, atol=1e-08) #the default values of rtol and atol are rtol=1e-05, atol=1e-08
    #If isTheAdjMatrixSymmetric is True, than each node must be in the same position as source as it is as target. Note that the SVD of the proximity matrix can yield somewhat different source and target coordinate matrixes even in the symmetric case, but this is only the consequence of numeric errors.

    #create the matrix to be reduced
    shortestPathLengthsDict = dict(nx.shortest_path_length(G,weight='weight'))
        #shortestPathLengthDict[source][target]=length of the shortest path from the source node to the target node

    if q==None: #use the default setting of the multiplying factor
        listOfAllSPLs = []
        for sour in shortestPathLengthsDict.keys():
            for targ in shortestPathLengthsDict[sour].keys():
                listOfAllSPLs.append(shortestPathLengthsDict[sour][targ])
        maxSPL = max(listOfAllSPLs)
        qmin = math.log(1.0/0.9999)*maxSPL
        qmax = math.log(10)*maxSPL
        q = math.exp((math.log(qmin)+math.log(qmax))/2)
        print('The default q multiplying factor is '+str(q)+'.')

    L = np.zeros((N,N)) #a matrix indicating the node-node distances; L[i][j] is the Lorentz product calculated from the length of the shortest path from the ith node to the jth node, where the nodes are ordered according to G.nodes
    rowID = 0
    for s in G: #iterate over the nodes as sources
        colID = 0
        for t in G: #iterate over all the nodes as targets
            if s==t:
                matrixElement = 0.0
            else:
                try:
                    matrixElement = math.exp(-q/shortestPathLengthsDict[s][t])
                except KeyError: #there is no path from node s to node t -> set the distance to the possible highest value (SPL=infinity -> exp(-q/SPL)=1)
                    matrixElement = 1.0
                except ZeroDivisionError: #two different nodes are connected with a link/links of weight 0
                    matrixElement = 0.0
            L[rowID,colID] = math.cosh(zeta*matrixElement)
            colID = colID+1
        rowID = rowID+1

    #dimension reduction
    if d == N-1:
        U,S,VT=np.linalg.svd(L) #find all the N number of singular values and the corresponding singular vectors (with decreasing order of the singular values in S)
             #note that for real matrices the conjugate transpose is just the transpose, i.e. V^H=V^T
    else: #d<N-1: only the first d+1 singular values (and the corresponding singular vectors) are retained
        U,S,VT = svds(L,d+1,solver='arpack') #the singular values are ordered from the smallest to the largest in S (increasing order)
        #reverse the order of the singular values to obtain a decreasing order:
        S = S[::-1]
        U = U[:,::-1]
        VT = VT[::-1]
    numOfPositiveSingularValues = np.sum(S > 0)
    if numOfPositiveSingularValues<d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of positive singular values of the proximity matrix-1, i.e. '+str(numOfPositiveSingularValues-1)+'.\n\n')
    Ssqrt = np.sqrt(S[1:]) #d number of singular values are used for determining the directions of the position vectors: from the second to the d+1th one

    #create the dictionaries of source and target positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node as source/as target in the native ball
    Coord_source = {}
    Coord_target = {}
    nodeIndex = 0
    numOfErrors_source = 0
    numOfErrors_target = 0
    for nodeName in G:
        if G.out_degree(nodeName)!=0: #If the node named nodeName has 0 links as a source node, then its position as a source can not be determined.
            #calculate the position of the given node as a source of links
            Uarray = U[nodeIndex,:]
            firstCoordOnHyperboloid_source = math.fabs(math.sqrt(S[0])*Uarray[0]) #to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid_source has to be positive
            if firstCoordOnHyperboloid_source<1: #a numerical error has occurred
                r_native_source = 0
                numOfErrors_source = numOfErrors_source+1
            else:
                r_native_source = (1/zeta)*math.acosh(firstCoordOnHyperboloid_source)
            sourceDirectionArray = (-1)*np.multiply(Uarray[1:],Ssqrt) #the jth element is the jth source coordinate of the node named nodeName in the reduced space; we use the additive inverse, because we want to identify the elements of the reduced matrix as Lorentz products of the position vectors
            originalNorm = np.linalg.norm(sourceDirectionArray)
            Coord_source[nodeName] = r_native_source*sourceDirectionArray/originalNorm #the Cartesian coordinates of the node named nodeName as a source node in the d-dimensional hyperbolic space
        if isTheAdjMatrixSymmetric:
            Coord_target[nodeName]=Coord_source[nodeName] #In undirected graphs, only one position can be assigned to each node - either the one that we calculated as source position, or the one that we calculated as target position. It does not matter, which one we choose.
        elif G.in_degree(nodeName)!=0: #If the node named nodeName has 0 links as a target node, then its position as a target can not be determined.
            #calculate the position of the given node as a target of links
            Varray = VT[:,nodeIndex]
            firstCoordOnHyperboloid_target = math.fabs(math.sqrt(S[0])*Varray[0]) #to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid_target has to be positive
            if firstCoordOnHyperboloid_target<1: #a numerical error has occurred
                r_native_target = 0
                numOfErrors_target = numOfErrors_target+1
            else:
                r_native_target = (1/zeta)*math.acosh(firstCoordOnHyperboloid_target)
            targetDirectionArray = np.multiply(Varray[1:],Ssqrt) #the jth element is the jth target coordinate of the node named nodeName in the reduced space
            originalNorm = np.linalg.norm(targetDirectionArray)
            Coord_target[nodeName] = r_native_target*targetDirectionArray/originalNorm #the Cartesian coordinates of the node named nodeName as a target node in the d-dimensional hyperbolic space
        nodeIndex = nodeIndex+1
    if numOfErrors_source>0:
        print('TREXPIC placed '+str(numOfErrors_source)+' source nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!')
    if numOfErrors_target>0:
        print('TREXPIC placed '+str(numOfErrors_target)+' target nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!')
    
    return [Coord_source,Coord_target]

