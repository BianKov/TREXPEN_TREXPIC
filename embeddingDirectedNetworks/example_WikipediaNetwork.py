#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import networkx as nx
import embedding as em
import mapAcc as ma
import graphReconstruction as gr
import GRcalculation as GRc



directoryName = "wikipedia"
edgeListFileName = "WCCedgeList.txt"
d = 32 #number of dimensions
numOfSampledNodePairs = 'all' #number of examined (sampled) node pairs in the mapping accuracy and the greedy routing tasks
numOfSampledLinks = 'all' #number of examined (sampled) links in the graph reconstruction task




print('load the graph')
G=em.loadGraph(os.getcwd()+"/"+directoryName+"/"+edgeListFileName,0,"\t") #this is an unweighted weakly connected component
N_WCC = len(G) #number of nodes in the WCC
print('N='+str(N_WCC))
print('E='+str(len(G.edges())))




print('Create d-dimensional embeddings')
print('Euclidean embedding with HOPE')
[Coord_source_HOPE_Euc,Coord_target_HOPE_Euc]=em.HOPE(G,d)
 
print('Euclidean embedding with TREXPEN-S')
[Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc]=em.TREXPEN_S(G,d)

print('hyperbolic embedding with TREXPEN-S')
[Coord_source_TREXPENS_hyp,Coord_target_TREXPENS_hyp] = em.convertEucToHyp(N_WCC,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc,d) #MIC

print('hyperbolic embedding with COM-shifted TREXPEN-S')
[Coord_source_TREXPENS_Euc_COMshift,Coord_target_TREXPENS_Euc_COMshift] = em.shiftCOMtoOrigin(Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc,d)
[Coord_source_TREXPENS_hyp_COMshift,Coord_target_TREXPENS_hyp_COMshift] = em.convertEucToHyp(N_WCC,Coord_source_TREXPENS_Euc_COMshift,Coord_target_TREXPENS_Euc_COMshift,d) #MIC

print('hyperbolic embedding with TREXPIC')
[Coord_source_TREXPIC,Coord_target_TREXPIC]=em.TREXPIC(G,d)




print('Evaluate embeddings from the point of view of mapping accuracy')
print('creation of a node pair list')
SPLvalues = ma.createNodePairListForTest(G,numOfSampledNodePairs)
print('calculation of mapping accuracies')
mappingAcc_HOPE_Euc_inner = ma.mapAcc_Euc_inner(SPLvalues,Coord_source_HOPE_Euc,Coord_target_HOPE_Euc)
mappingAcc_HOPE_Euc_dist = ma.mapAcc_Euc_dist(SPLvalues,Coord_source_HOPE_Euc,Coord_target_HOPE_Euc)
mappingAcc_TREXPENS_Euc_inner = ma.mapAcc_Euc_inner(SPLvalues,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc)
mappingAcc_TREXPENS_Euc_dist = ma.mapAcc_Euc_dist(SPLvalues,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc)
mappingAcc_TREXPENS_hyp = ma.mapAcc_hyp(SPLvalues,Coord_source_TREXPENS_hyp,Coord_target_TREXPENS_hyp)
mappingAcc_TREXPENS_hyp_COMshift = ma.mapAcc_hyp(SPLvalues,Coord_source_TREXPENS_hyp_COMshift,Coord_target_TREXPENS_hyp_COMshift)
mappingAcc_TREXPIC = ma.mapAcc_hyp(SPLvalues,Coord_source_TREXPIC,Coord_target_TREXPIC)




print('Evaluate embeddings from the point of view of graph reconstruction')
print('creation of a node pair list')
[examinedNodePairs_name,listOfLinksInTheGivenSetOfNodePairs] = gr.createNodePairListForTest(G,numOfSampledLinks)

print('estimate the performance of a random predictor')
AP_random = len(listOfLinksInTheGivenSetOfNodePairs) / len(examinedNodePairs_name) #number of links to be reconstructed/number of possibilities
numOfHitsATnumOfLinksToBeReconstructed_random = AP_random*len(listOfLinksInTheGivenSetOfNodePairs) #precision * number of links to be reconstructed
precisionATnumOfLinksToBeReconstructed_random = AP_random
AUROC_random = 0.5

print('graph reconstruction with local methods')
print('Common neighbors')
linkProbabilityDict = gr.linkProbability_commonNeighbors(examinedNodePairs_name,G)
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_CN,numOfHitsATnumOfLinksToBeReconstructed_CN, recall,precision,AP_CN, falsePositiveRate,truePositiveRate,AUROC_CN] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Configuration model')
linkProbabilityDict = gr.linkProbability_configModel(examinedNodePairs_name,G)
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_Config,numOfHitsATnumOfLinksToBeReconstructed_Config, recall,precision,AP_Config, falsePositiveRate,truePositiveRate,AUROC_Config] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Resource allocation index with out-degree')
linkProbabilityDict = gr.linkProbability_resourceAllocation_outDeg(examinedNodePairs_name,G)
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_ResArcO,numOfHitsATnumOfLinksToBeReconstructed_ResArcO, recall,precision,AP_ResArcO, falsePositiveRate,truePositiveRate,AUROC_ResArcO] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Resource allocation index with in-degree')
linkProbabilityDict = gr.linkProbability_resourceAllocation_inDeg(examinedNodePairs_name,G)
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_ResArcI,numOfHitsATnumOfLinksToBeReconstructed_ResArcI, recall,precision,AP_ResArcI, falsePositiveRate,truePositiveRate,AUROC_ResArcI] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Resource allocation index with total degree')
linkProbabilityDict = gr.linkProbability_commonNeighbors(examinedNodePairs_name,G)
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_ResArcT,numOfHitsATnumOfLinksToBeReconstructed_ResArcT, recall,precision,AP_ResArcT, falsePositiveRate,truePositiveRate,AUROC_ResArcT] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)

print('graph reconstruction based on embeddings')
print('HOPE')
print('Probability calculation - inner product')
linkProbabilityDict = gr.linkProbability_Euc_inner(examinedNodePairs_name,Coord_source_HOPE_Euc,Coord_target_HOPE_Euc)
print('Evaluation')
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_HOPE_Euc_inner, numOfHitsATnumOfLinksToBeReconstructed_HOPE_Euc_inner, recall,precision,AP_HOPE_Euc_inner, falsePositiveRate,truePositiveRate,AUROC_HOPE_Euc_inner] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Probability calculation - Euclidean distance')
linkProbabilityDict = gr.linkProbability_Euc_dist(examinedNodePairs_name,Coord_source_HOPE_Euc,Coord_target_HOPE_Euc)
print('Evaluation')
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_HOPE_Euc_dist, numOfHitsATnumOfLinksToBeReconstructed_HOPE_Euc_dist, recall,precision,AP_HOPE_Euc_dist, falsePositiveRate,truePositiveRate,AUROC_HOPE_Euc_dist] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)

print('TREXPEN-S')
print('Probability calculation - inner product')
linkProbabilityDict = gr.linkProbability_Euc_inner(examinedNodePairs_name,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc)
print('Evaluation')
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_TREXPENS_Euc_inner, numOfHitsATnumOfLinksToBeReconstructed_TREXPENS_Euc_inner, recall,precision,AP_TREXPENS_Euc_inner, falsePositiveRate,truePositiveRate,AUROC_TREXPENS_Euc_inner] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Probability calculation - Euclidean distance')
linkProbabilityDict = gr.linkProbability_Euc_dist(examinedNodePairs_name,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc)
print('Evaluation')
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_TREXPENS_Euc_dist, numOfHitsATnumOfLinksToBeReconstructed_TREXPENS_Euc_dist, recall,precision,AP_TREXPENS_Euc_dist, falsePositiveRate,truePositiveRate,AUROC_TREXPENS_Euc_dist] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)
print('Probability calculation - hyperbolic distance')
linkProbabilityDict = gr.linkProbability_hyp(examinedNodePairs_name,Coord_source_TREXPENS_hyp,Coord_target_TREXPENS_hyp)
print('Evaluation')
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_TREXPENS_hyp, numOfHitsATnumOfLinksToBeReconstructed_TREXPENS_hyp, recall,precision,AP_TREXPENS_hyp, falsePositiveRate,truePositiveRate,AUROC_TREXPENS_hyp] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)

print('TREXPIC')
print('Probability calculation - hyperbolic distance')
linkProbabilityDict = gr.linkProbability_hyp(examinedNodePairs_name,Coord_source_TREXPIC,Coord_target_TREXPIC)
print('Evaluation')
[kList,precisionATk,precisionATnumOfLinksToBeReconstructed_TREXPIC, numOfHitsATnumOfLinksToBeReconstructed_TREXPIC, recall,precision,AP_TREXPIC, falsePositiveRate,truePositiveRate,AUROC_TREXPIC] = gr.precisionATk_precisionRecallCurve_ROCcurve(listOfLinksInTheGivenSetOfNodePairs,linkProbabilityDict)




print('Evaluate embeddings from the point of view of greedy routing')
print('creation of a node pair list')
examinedNodePairs = GRc.createNodePairListForTest(G,numOfSampledNodePairs)

print('HOPE')
print('GR-score calculation using inner products')
[GR_HOPE_Euc_inner,succPaths_HOPE_Euc_inner,avgLength_HOPE_Euc_inner,avgShLength_HOPE_Euc_inner] = GRc.greedy_routing(G,examinedNodePairs,Coord_source_HOPE_Euc,Coord_target_HOPE_Euc,'innerProd')
print('GR-score calculation using Euclidean distances')
[GR_HOPE_Euc_dist,succPaths_HOPE_Euc_dist,avgLength_HOPE_Euc_dist,avgShLength_HOPE_Euc_dist] = GRc.greedy_routing(G,examinedNodePairs,Coord_source_HOPE_Euc,Coord_target_HOPE_Euc,'EucDist')

print('TREXPEN-S')
print('GR-score calculation using inner products')
[GR_TREXPENS_Euc_inner,succPaths_TREXPENS_Euc_inner,avgLength_TREXPENS_Euc_inner,avgShLength_TREXPENS_Euc_inner] = GRc.greedy_routing(G,examinedNodePairs,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc,'innerProd')
print('GR-score calculation using Euclidean distances')
[GR_TREXPENS_Euc_dist,succPaths_TREXPENS_Euc_dist, avgLength_TREXPENS_Euc_dist,avgShLength_TREXPENS_Euc_dist] = GRc.greedy_routing(G,examinedNodePairs,Coord_source_TREXPENS_Euc,Coord_target_TREXPENS_Euc,'EucDist')
print('GR-score calculation using hyperbolic distances')
[GR_TREXPENS_hyp,succPaths_TREXPENS_hyp,avgLength_TREXPENS_hyp,avgShLength_TREXPENS_hyp] = GRc.greedy_routing(G,examinedNodePairs,Coord_source_TREXPENS_hyp,Coord_target_TREXPENS_hyp,'hypDist')

print('TREXPIC')
print('GR-score calculation - hyperbolic distance')
[GR_TREXPIC,succPaths_TREXPIC,avgLength_TREXPIC,avgShLength_TREXPIC] = GRc.greedy_routing(G,examinedNodePairs,Coord_source_TREXPIC,Coord_target_TREXPIC,'hypDist')
