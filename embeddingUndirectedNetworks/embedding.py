#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import networkx as nx
import math
from scipy.sparse.linalg import svds
from numpy import linalg as LA #for calculating the spectral radius of the adjacency matrix in the variants of HOPE



#A function for loading the undirected edge list of the network to be embedded. Edge weights are disregarded, self-loops are removed, multi-edges are converted to single edges, and only the largest connected component is returned as a NetworkX Graph.
#path is a string with the path of the text file containing the edge list to be loaded
    #In the edge list each line has to correspond to one connected node pair. Columns after the first two (that contain the node identifiers) will be disregarded, each edge will have weight 1.
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

    G_total = nx.Graph()
    G_total.add_edges_from(edgeList) #multi-edges are automatically converted to single edges
    #extract the largest connected component:
    G=max([G_total.subgraph(comp).copy() for comp in nx.connected_components(G_total)],key=len) #.copy(): create a subgraph with its own copy of the edge/node attributes -> changes to attributes in the subgraph are NOT reflected in the original graph; without copy the subgraph is a frozen graph for which edges can not be added or removed

    return G



#At https://www.nature.com/articles/s41467-017-01825-5 different pre-weighting strategies are described for undirected networks that facilitated the estimation of the spatial arrangement of the nodes. This function performs these link weighting procedures. Larger weight corresponds to less similarity.
#G is the NetworkX Graph to be pre-weighted. Note that the weights in G do not change, the function returns a weighted copy of G.
#weightingType is a string corresponding to the name of the pre-weighting rule to be used. Can be set to 'RA1', 'RA2', 'RA3', 'RA4' or 'EBC'.
#Example for function call:
#   G_weighted = embedding.preWeighting(G,'RA1')
def preWeighting(G,weightingType):
    G_weighted = nx.Graph()
    G_weighted.add_nodes_from(G.nodes) #keep the node order of the original graph
    if weightingType=='RA1':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            # set=unordered collection with no duplicate elements,
            # set operations (union, intersect, complement) can be executed (see RAtype==2)
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j) + G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA2':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j

            # ei=the external degree of the node i with respect to node j,
            # i.e. the number of links from i to neither j nor the common neighbors with j,
            # i.e. the number of i's neighbors without node j and the common neighbors with j
            neighborSet_i = {n for n in G[i]}  # set with the indices of the neighbors of node i
            # G[i]=adjacency dictionary of node i -> iterating over its keys(=neighboring node indices)
            ei = len(neighborSet_i - {j} - CNset)

            # ej=the external degree of the node j with respect to node i,
            # i.e. the number of links from j to neither i nor the common neighbors with i,
            # i.e. the number of j's neighbors without node i and the common neighbors with i
            neighborSet_j = {n for n in G[j]}  # set with the indices of the neighbors of node j
            # G[j]=adjacency dictionary of node j -> iterating over its keys(=neighboring node indices)
            ej = len(neighborSet_j - {i} - CNset)

            # assign a weight to the i-j edge:
            w = (1 + ei + ej + ei * ej) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA3':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) + G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='RA4':
        for (i, j) in G.edges():  # (i,j)=a tuple, which denotes the edge between node u and v
            CNset = set(nx.common_neighbors(G, i, j))  # set of the common neighbors' indices
            CN = len(CNset)  # number of common neighbors of nodes i and j
            # assign a weight to the i-j edge:
            w = (G.degree(i) * G.degree(j)) / (1 + CN)
            G_weighted.add_edge(i,j,weight=w)
    elif weightingType=='EBC': #use the edge betweenness centrality
        #create a dictionary, which contains all the shortest paths between all node pairs
            #shortestPathsDict[(source,target)] is the list of shortest paths from node with ID source to node with ID target
                #a path is a list of nodes following each other in the path
            #the graph to be embedded should be connected (all nodes can be reached from any node)
        shortestPathsDict = {}
        nodeList=list(G.nodes)
        N = len(nodeList)
        for u in range(N-1): #u=0,1,...,N-2
            for v in range(u+1,N): #v=u+1,...,N-1
            #these loops are sufficient only if graph G is undirected (the same number of paths lead from the uth node to the vth node and from the vth node to the uth node) and does not contain any self-loops
                node_u = nodeList[u]
                node_v = nodeList[v]
                shortestPathsDict[(node_u,node_v)]=[p for p in nx.all_shortest_paths(G,source=node_u,target=node_v,weight=None)] #weight=None: every edge has weight/distance/cost 1 (the possible current weights are disregarded)

        #weight all the edges
        for (i,j) in G.edges():
            w=0 #initialize the weight of the i-j edge
            for u in range(N-1):
                for v in range(u+1,N):
                    shortestPathsBetween_uv = shortestPathsDict[(nodeList[u],nodeList[v])] #list of shortest paths between the uth node and the vth node
                    sigma = len(shortestPathsBetween_uv) #the total number of shortest paths between the uth node and the vth node
                    #count those paths between node u and node v which contains the i-j edge
                    sigma_ij = 0
                    for q in shortestPathsBetween_uv: #q=list of nodes following each other in a path between the uth node and the vth node
                        if i in q and j in q: #since q is a shortest path, therefore in this case abs(q.index(i)-q.index(j))==1 is already granted
                            sigma_ij = sigma_ij+1
                    w=w+(sigma_ij/sigma)
            G_weighted.add_edge(i,j,weight=w) #assign a weight to the i-j edge
    else:
        print('False parameter: weightingType\n')
    return G_weighted






#A function for creating a d-dimensional Euclidean embedding with TREXPEN.
#G is the (pre-weighted) NetworkX Graph to be embedded
#d is the dimension of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (e.g. q=0.001 keeps all the proximity values close to 1 (except the 0 values, which correspond to infinitely large distances along the graph), while q=1000 keeps all the proximity values close to 0 (except the 1 values, which correspond to paths of 0 length))
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#Example for function call:
#   Coord=embedding.TREXPEN(G,d)
def TREXPEN(G,d,q=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN can not be larger than the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.

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

    #create the matrix from which the node coordinates will be obtained
    CoordMatrix=np.matmul(U,Ssqrt) #the jth column contains the jth coordinate of all nodes in the reduced space
        #CoordMatrix=np.transpose(np.matmul(Ssqrt,VT)) could also be used

    #create the dictionary of the node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in G:
        Coord[nodeName] = CoordMatrix[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1
    
    return Coord





#A function for creating a d-dimensional Euclidean embedding with TREXPEN-R.
#G is the (pre-weighted) NetworkX Graph to be embedded
#d is the dimension of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (e.g. q=0.001 keeps all the proximity values close to 1 (except the 0 values, which correspond to infinitely large distances along the graph), while q=1000 keeps all the proximity values close to 0 (except the 1 values, which correspond to paths of 0 length))
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#Example for function call:
#   Coord=embedding.TREXPEN_R(G,d)
def TREXPEN_R(G,d,q=None):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN_R can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.

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
 
    #create the matrix from which the node coordinates will be obtained
    CoordMatrix=np.matmul(U,Ssqrt) #the jth column contains the j-1th coordinate of all nodes in the reduced space; the 0th column will be discarded in the absence of centering
        #CoordMatrix=np.transpose(np.matmul(Ssqrt,VT)) could also be used

    #create the dictionary of the node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in G:
        Coord[nodeName] = CoordMatrix[nodeIndex,1:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1
    
    return Coord






#A function for creating a d-dimensional Euclidean embedding with TREXPEN-S.
#G is the (pre-weighted) NetworkX Graph to be embedded
#d is the dimension of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (e.g. q=0.001 keeps all the proximity values close to 1 (except the 0 values, which correspond to infinitely large distances along the graph), while q=1000 keeps all the proximity values close to 0 (except the 1 values, which correspond to paths of 0 length))
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#Example for function call:
#   Coord=embedding.TREXPEN_S(G,d)
def TREXPEN_S(G,d,q=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPEN_S can not be larger than the number of nodes the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.

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

    #create the matrix from which the node coordinates will be obtained
    CoordMatrix=np.matmul(U,Ssqrt) #the jth column contains the jth coordinate of all nodes in the reduced space
        #CoordMatrix=np.transpose(np.matmul(Ssqrt,VT)) could also be used

    #create the dictionary of the node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in G:
        Coord[nodeName] = CoordMatrix[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return Coord






#A function for performing High-Order Proximity preserved Embedding (HOPE) (SVD of Katz proximity matrix).
#G is the NetworkX Graph to be embedded (all the link weights will be considered to be 1; the here-applied formula of the Katz matrix is not valid for weighted networks)
#d is the number of dimensions of the Euclidean embedding space
#alpha is the decaying constant; the default is None, in this case alpha will be set to 0.5/spectral radius (its value will be printed)
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#Example for function call:
#   Coord=embedding.HOPE(G,d)
def HOPE(G,d,alpha=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE can not be larger than the number of nodes the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight=None) #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes().

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

    #create the matrix from which the node coordinates will be obtained
    CoordMatrix=np.matmul(U,Ssqrt) #the jth column contains the jth coordinate of all nodes in the reduced space
        #CoordMatrix=np.transpose(np.matmul(Ssqrt,VT)) could also be used

    #create the dictionary of the node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in G:
        Coord[nodeName] = CoordMatrix[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return Coord





#A function for performing High-Order Proximity preserved Embedding (HOPE) (SVD of Katz proximity matrix) with the neglection of the first dimension of the embedding.
#G is the NetworkX Graph to be embedded (all the link weights will be considered to be 1; the here-applied formula of the Katz matrix is not valid for weighted networks)
#d is the number of dimensions of the Euclidean embedding space
#alpha is the decaying constant; the default is None, in this case alpha will be set to 0.5/spectral radius (its value will be printed)
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#Example for function call:
#   Coord=embedding.HOPE_R(G,d)
def HOPE_R(G,d,alpha=None):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE_R can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight=None) #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes().

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

    #create the matrix from which the node coordinates will be obtained
    CoordMatrix=np.matmul(U,Ssqrt) #the jth column contains the j-1th coordinate of all nodes in the reduced space; the 0th column will be discarded in the absence of centering
        #CoordMatrix=np.transpose(np.matmul(Ssqrt,VT)) could also be used

    #create the dictionary of the node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in G:
        Coord[nodeName] = CoordMatrix[nodeIndex,1:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return Coord





#A function for performing High-Order Proximity preserved Embedding (HOPE) (SVD of Katz proximity matrix) with shifting the mean of all matrix elements to 0.
#G is the NetworkX Graph to be embedded (all the link weights will be considered to be 1; the here-applied formula of the Katz matrix is not valid for weighted networks)
#d is the number of dimensions of the Euclidean embedding space
#alpha is the decaying constant; the default is None, in this case alpha will be set to 0.5/spectral radius (its value will be printed)
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#Example for function call:
#   Coord=embedding.HOPE_S(G,d)
def HOPE_S(G,d,alpha=None):
    N = len(G) #the number of nodes in graph G
    if N < d:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.HOPE_S can not be larger than the number of nodes the number of nodes, i.e. '+str(N)+'.\n\n')

    A_np = nx.to_numpy_array(G,nodelist=None,weight=None) #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes().

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

    #create the matrix from which the node coordinates will be obtained
    CoordMatrix=np.matmul(U,Ssqrt) #the jth column contains the jth coordinate of all nodes in the reduced space
        #CoordMatrix=np.transpose(np.matmul(Ssqrt,VT)) could also be used

    #create the dictionary of the node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord = {}
    nodeIndex = 0
    for nodeName in G:
        Coord[nodeName] = CoordMatrix[nodeIndex,:] #the Cartesian coordinates of the ith node (according to the order in G.nodes) in the d-dimensional Euclidean space
        nodeIndex = nodeIndex+1

    return Coord











#A function for shifting the center of mass of a d-dimensional Euclidean embedding to the origin.
#Coord_Euc_original is a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#d is the number of dimensions of the space to which the network is embedded
#The function returns the dictionary Coord_Euc_shifted that assigns to the node names NumPy arrays of d elements containing the shifted Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.    
def shiftCOMtoOrigin(Coord_Euc_original,d):
    #determine the position of the center of mass
    meanOfCoords=np.zeros(d)
    for nodeName in Coord_Euc_original.keys():
        meanOfCoords = meanOfCoords+Coord_Euc_original[nodeName]
    meanOfCoords = meanOfCoords/len(Coord_Euc_original)
    #create the dictionary of the SHIFTED node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node
    Coord_Euc_shifted = {}
    for nodeName in Coord_Euc_original.keys():
        Coord_Euc_shifted[nodeName] = Coord_Euc_original[nodeName]-meanOfCoords
    return Coord_Euc_shifted







#A function for performing the model-independent conversion MIC, i.e. for converting a d-dimensional Euclidean embedding of a graph to a hyperbolic embedding by changing its radial node arrangement.
#N is the number of nodes in the embedded graph
#Coord_Euc is a dictionary that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the d-dimensional Euclidean space.
#d is the number of dimensions of the space to which the network is embedded
#C is a multiplying factor in the desired largest hyperbolic radial coordinate rhmax=(C/zeta)*ln(N), where zeta=sqrt(-K). The volume of the hyperbolic sphere inside the outermost node will be proportional to N^(C*(d-1)) for not too small networks.
#K<0 is the curvature of the hyperbolic space
#The function returns the dictionary Coord_hyp that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the native representation of the d-dimensional hyperbolic space.
#Example for function call:
#   Coord_hyp=embedding.convertEucToHyp(G,Coord_Euc,d)
def convertEucToHyp(N,Coord_Euc,d,C=2,K=-1):
    zeta = math.sqrt(-K)
     
    #identify the radial interval of the Euclidean embedding
    norm_Euc_dict = {nodeName:np.linalg.norm(Coord_Euc[nodeName]) for nodeName in Coord_Euc.keys()} #dictionary with the lengths of the Euclidean position vectors
    r_E_min = min(norm_Euc_dict.values())
    r_E_max = max(norm_Euc_dict.values())
    
    #calculate the hyperbolic node positions
    Coord_hyp = {}
    if r_E_min==0: #the Euclidean embedding method placed at least one node in the origin, which is a position that does not have an obvious hyperbolic equivalent
        increasingEucRadCoords = sorted(norm_Euc_dict.values()) #list of radial coordinates sorted in an increasing order
        i = 0
        while increasingEucRadCoords[i]==0:
            i = i+1
        r_E_nonZeroMin = increasingEucRadCoords[i] #the occurring smallest non-zero Euclidean radial coordinate in the embedded graph
        r_h_max = C*math.log(N)/zeta #calculate the largest hyperbolic radial coordinate obtained among the nodes that were not in the origin in the Euclidean case
        numOfEucZeros = 0
        for nodeName in Coord_Euc.keys():
            if norm_Euc_dict[nodeName]==0: #the node named nodeName was placed in the origin in the Euclidean embedding -> estimate its hyperbolic position
                numOfEucZeros = numOfEucZeros+1
                Coord_hyp[nodeName] = np.zeros(d) #initialization
                while np.linalg.norm(Coord_hyp[nodeName])<0.0001: #in order to avoid problems with floating point precision when normalizing the array
                    Coord_hyp[nodeName] = np.random.standard_normal(d) #the hyperbolic angular coordinates of nodes that were placed in the origin in the Euclidean embedding are set to random values
                Coord_hyp[nodeName] = 10*r_h_max*Coord_hyp[nodeName]/np.linalg.norm(Coord_hyp[nodeName]) #the hyperbolic radial coordinate of nodes that were placed in the origin in the Euclidean embedding is set to an extremely large value, namely 10*the largest hyperbolic r obtained among the nodes that were not in the origin in the Euclidean case
            else: #the node named nodeName was placed out of the origin in the Euclidean embedding -> calculate its hyperbolic position as usual
                try:
                    r_hyp = math.log(1+(math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin/norm_Euc_dict[nodeName],d))/(zeta*(d-1))
                    if r_hyp==0: #a rounding error has occurred, because the term (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin/norm_Euc_dict[nodeName],d) is too small compared to the +1 in the logarithm
                        r_hyp = (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin/norm_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                except OverflowError: #math.pow(N,C*(d-1)) is too large
                    try:
                        r_hyp = math.log(1+math.pow(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin/norm_Euc_dict[nodeName],d))/(zeta*(d-1))
                        if r_hyp==0: #a rounding error has occurred
                            r_hyp = math.pow(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin/norm_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                    except OverflowError: #the dth power is still too large
                        r_hyp = math.log(math.pow(N,C*(d-1)/d)*r_E_nonZeroMin/norm_Euc_dict[nodeName])*d/(zeta*(d-1))
                        #this can not become 0
                Coord_hyp[nodeName] = r_hyp*Coord_Euc[nodeName]/norm_Euc_dict[nodeName] #the Cartesian coordinates of the node named nodeName in the native representation of the d-dimensional hyperbolic space
        print('The Euclidean embedding method placed '+str(numOfEucZeros)+' nodes in the origin. In the hyperbolic embedding, the angular coordinates of these nodes are set to random values, and their hyperbolic radial coordinate is set to an extremely large value, namely 10*the largest hyperbolic r obtained among the nodes that were not in the origin in the Euclidean case. Consider changing the q or alpha parameter of the Euclidean embedding!')
    else: #the Euclidean embedding placed all the nodes out of the origin
        for nodeName in Coord_Euc.keys():
            try:
                r_hyp = math.log(1+(math.pow(N,C*(d-1))-1)*math.pow(r_E_min/norm_Euc_dict[nodeName],d))/(zeta*(d-1))
                if r_hyp==0: #a rounding error has occurred, because the term (math.pow(N,C*(d-1))-1)*math.pow(r_E_nonZeroMin/norm_Euc_dict[nodeName],d) is too small compared to the +1 in the logarithm
                    r_hyp = (math.pow(N,C*(d-1))-1)*math.pow(r_E_min/norm_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
            except OverflowError: #math.pow(N,C*(d-1)) is too large
                try:
                    r_hyp = math.log(1+math.pow(math.pow(N,C*(d-1)/d)*r_E_min/norm_Euc_dict[nodeName],d))/(zeta*(d-1))
                    if r_hyp==0: #a rounding error has occurred
                        r_hyp = math.pow(math.pow(N,C*(d-1)/d)*r_E_min/norm_Euc_dict[nodeName],d)/(zeta*(d-1)) #ln(1+x) = x if x is close to 0
                except OverflowError: #the dth power is still too large
                    r_hyp = math.log(math.pow(N,C*(d-1)/d)*r_E_min/norm_Euc_dict[nodeName])*d/(zeta*(d-1))
                    #this can not become 0
            Coord_hyp[nodeName] = r_hyp*Coord_Euc[nodeName]/norm_Euc_dict[nodeName] #the Cartesian coordinates of the node named nodeName in the native representation of the d-dimensional hyperbolic space

    return Coord_hyp
















#A function for creating a d-dimensional hyperbolic embedding with TREXPIC.
#G is the (pre-weighted) NetworkX Graph to be embedded
#d is the number of dimensions of the space to which the network will be embedded
#q>0 is the multiplying factor in the exponent in the matrix to be reduced (infinite distances are always mapped to 1; the increase in q shifts the non-unit off-diagonal matrix elements towards 0)
#K<0 is the curvature of the hyperbolic space
#The function returns the dictionary Coord that assigns to the node names NumPy arrays of d elements containing the Cartesian coordinates of the network nodes in the native representation of the d-dimensional hyperbolic space.
#Example for function call:
#   Coord=embedding.TREXPIC(G,d)
def TREXPIC(G,d,q=None,K=-1):
    N = len(G) #the number of nodes in graph G
    if N < d+1:
        print('\n\nERROR: The number d of embedding dimensions in the function embedding.TREXPIC can not be larger than the number of nodes the number of nodes-1, i.e. '+str(N-1)+'.\n\n')

    zeta = math.sqrt(-K)

    A_np = nx.to_numpy_array(G,nodelist=None,weight='weight') #Create the adjacency matrix of G as a Numpy Matrix. If nodelist is None, then the ordering of rows and columns is produced by G.nodes(). If no weight has been assigned to the links, then each edge has weight 1.

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

    #create the dictionary of node positions: key=node name, value=NumPy array of d elements containing the Cartesian coordinates of the given node in the native ball
    Coord = {}
    nodeIndex = 0
    numOfErrors = 0
    for nodeName in G:
        #calculate the position of the given node
        Uarray = U[nodeIndex,:]
        firstCoordOnHyperboloid = math.fabs(math.sqrt(S[0])*Uarray[0]) #to select the upper sheet of the two-sheet hyperboloid, firstCoordOnHyperboloid has to be positive
            #we could also use: Varray = VT[:,nodeIndex] and then firstCoordOnHyperboloid = math.fabs(math.sqrt(S[0])*Varray[0])
        if firstCoordOnHyperboloid<1: #a numerical error has occurred
            r_native = 0
            numOfErrors = numOfErrors+1
        else:
            r_native = (1/zeta)*math.acosh(firstCoordOnHyperboloid)
        directionArray = (-1)*np.multiply(Uarray[1:],Ssqrt) #the jth element is the jth coordinate of the node named nodeName in the reduced space; we use the additive inverse, because we want to identify the elements of the reduced matrix as Lorentz products of the position vectors
            #we could also use: directionArray = np.multiply(Varray[1:],Ssqrt)
        originalNorm = np.linalg.norm(directionArray)
        Coord[nodeName] = r_native*directionArray/originalNorm #the Cartesian coordinates of the node named nodeName in the d-dimensional hyperbolic space
        nodeIndex = nodeIndex+1
    if numOfErrors>0:
        print('TREXPIC placed '+str(numOfErrors)+' nodes at an invalid position on the hyperboloid. The native radial coordinate of these nodes is set to 0. Consider changing the q parameter of the embedding!')
    
    return Coord

