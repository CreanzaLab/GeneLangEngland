import pickle

import numpy as np
from cHaversine import haversine
from ete3 import Tree
from scipy.spatial import distance
from scipy.stats import linregress, mannwhitneyu

# for each town, its longitude and latitude:
coordsLing = np.genfromtxt("SED_coords.tsv", delimiter='\t')
# for each town, its location code (region with genetic data):
locationsLing = np.genfromtxt("SED_PoBI_locations.csv",
                              delimiter=',', skip_header=True, usecols=(4))
# for each town, its associated linguistic data:
featuresLing = np.genfromtxt("cmbnd_features_only.tsv", delimiter='\t')

# genetic sharing matrix, AKA copying matrix, which is direct output from fineSTRUCTURE and contains
#  estimated length of the genome contributed from every individual to every other individual.
geneticSharing = np.genfromtxt("../../tmp/all_chr.chunklengths.out",
                               delimiter=" ", skip_header=True,
                               usecols=(range(1, 2040)))
geneticSharingIDs = np.genfromtxt("../../tmp/all_chr.chunklengths.out",
                                  delimiter=" ", skip_header=True,
                                  usecols=(0), dtype=str)

# clean geneticSharing to remove individuals from outside of the study area:
geneticSharing = geneticSharing[:, np.min(geneticSharing, axis=0) == 0]

geneticLocations = np.genfromtxt(
    "../../wtccc/_WTCCC2_POBI_illumina_calls_POBI_illumina.sampleLocations",
    delimiter="\t", skip_header=1, usecols=(1, 3), dtype=str)
geneticLocations = geneticLocations[geneticLocations[:, 0].astype(int) < 30, :]

# reorder genetic sharing matrix:
orderedIDs = [np.where(geneticSharingIDs == id_)[0][0]
              for id_ in geneticLocations[:, 1]]
geneticSharing = geneticSharing[orderedIDs, :]

del geneticSharingIDs, orderedIDs

# Number of permutations for significance tests
NUMRAND = 100000

# from the haplotype sharing matrix, create a pairwise (individual-individual)
#  distance matrix to use in the analysis:
pairwiseDistMtrx = distance.squareform(distance.pdist(
    geneticSharing, "euclidean"))

locIdx = list(range(1, 30))

# create a pair of lists with the individuals (for genetic data)
#  and towns (for linguistic data) associated with each location
locationIndicesGen = []
locationIndicesLing = []
for location in np.unique(geneticLocations[:, 0].astype(int)):
    locationIndicesGen.append(np.where(
        geneticLocations[:, 0].astype(int) == location)[0])
    locationIndicesLing.append(np.where(
        locationsLing == location)[0])


# cut the tree into K clusters based on distance
def cut_tree(k, tree):
    
    if type(k) is not int:
        raise TypeError("k must be an integer greater than or equal to 2")
    if k < 2:
        raise ValueError("k must be an integer greater than or equal to 2")
    
    clustersFound = tree.get_children()
    
    while len(clustersFound) < k:

        distances = [tree.get_distance(node) for node in clustersFound]
        subtreeNode = clustersFound[distances.index(min(distances))]
        
        for childNode in subtreeNode.get_children():
            clustersFound.append(childNode)
        clustersFound.remove(subtreeNode)
        
    return clustersFound


# remove bad values from a numpy object
def notNan(x, y=None):
    if y is None:
        return x[~(np.isnan(x)|np.isinf(x))]
    else:
        return y[~(np.isnan(x)|np.isinf(x))]
    

def meanGenDistance(pgenDists, oper="mean"):
    if oper == "mean":
        if len(pgenDists.shape) == 1:
            return np.mean(pgenDists)

        if all(np.diag(pgenDists) == 0) and (pgenDists.shape[0] == pgenDists.shape[1]):    
            return np.mean(
                pgenDists[np.triu_indices_from(pgenDists, k=1)])
        else:
            return np.mean(pgenDists)
    if oper=="sum":
        if len(pgenDists.shape) == 1:
            return np.sum(pgenDists)

        if all(np.diag(pgenDists) == 0) and (pgenDists.shape[0] == pgenDists.shape[1]):    
            return np.sum(np.triu(pgenDists))
        else:
            return np.sum(pgenDists)


# import the final tree produced by fineSTRUCTURE
britain_tree = Tree("../../tmp/all_chr_newick_tree")
english_leaves = [britain_tree.get_leaves_by_name(indiv)[0]
                  for indiv, loc
                  in zip(geneticLocations[:, 1],
                         geneticLocations[:, 0])
                  if int(loc) < 30]

# keep only leaves in England
britain_tree.prune(
    english_leaves,
    preserve_branch_length=True)
# label clusters
cutEnglish = cut_tree(7, britain_tree)

short_legible_names = {
    'A0': "All",
    'A1': "Eng",
    'A2': "Scot Bdr",
    'A3': "C. Eng (A3)",
    'A4': "C. Eng (A4)",
    'A5': "C. Eng (A5)",
    'V0': "Cornwall",
    'V1': "Wlsh Bdr",
    'V2': "Somerset",
    'V3': "C. Eng (V3)",
    'V4': "W Yorkshr",
    'V5': "Cumbria",
    'V6': "Northumbr."
}

# name the nodes
for i in range(len(cutEnglish)):
    cutEnglish[i].name = "V" + str(i)
    cutEnglish[i].name = (cutEnglish[i].name, short_legible_names[cutEnglish[i].name])
    print(len(cutEnglish[i].get_leaf_names()))

i = 0
for node in britain_tree.traverse():
    if node not in cutEnglish and not any([(n in cutEnglish)
                                           for n in node.get_ancestors()]):
        node.name = "A" + str(i)
        node.name = (node.name, short_legible_names[node.name])
        i += 1
        print(node.name, end=": ")
        print(len(node.get_leaf_names()))
        
# the set of comparisons we will make at each K=2 to K=7 (considering only pairs of adjacent clusters)
compareDict = {
    '2': [('A1', 'A2')],
    '3': [('A3', 'A2'), ('A3', 'V0')],
    '4': [('A4', 'A2'), ('A4', 'V0'), ('A4', 'V1')],
    '5': [('A5', 'A2'), ('V2', 'V0'), ('V2', 'A5'),
          ('V2', 'V1'), ('A5', 'V1')],
    '6': [('V3', 'A2'), ('V3', 'V4'), ('V4', 'A2'),
          ('V3', 'V2'), ('V2', 'V0'), ('V4', 'V1'),
          ('V2', 'V1'), ('V3', 'V1')],
    '7': [('V3', 'V5'), ('V3', 'V4'), ('V4', 'V5'),
          ('V4', 'V6'), ('V3', 'V2'), ('V2', 'V0'),
          ('V4', 'V1'), ('V2', 'V1'), ('V3', 'V1'),
          ('V5', 'V6')]}

np.random.seed(212902871)

# find the distances between each pair of locations with linguistic data
connectivityMatrix = np.zeros(np.repeat(coordsLing.shape[0], 2))
for i in range(coordsLing.shape[0] - 1):
    for j in range(i + 1, coordsLing.shape[0]):
        connectivityMatrix[i, j] =\
            connectivityMatrix[j, i] = haversine(
            tuple(coordsLing[i, :]), tuple(coordsLing[j, :])
        ) / 1000

# # include connections only between locations less than 100 km apart:
connectivityMatrix[connectivityMatrix > 100] = 0
connectedPairs = np.asmatrix(np.where(connectivityMatrix != 0)).T
connectedPairs = connectedPairs[
                    [i[0] for i in (connectedPairs[:, 0] < connectedPairs[:, 1]).tolist()], :
                 ].tolist()
del connectivityMatrix  # normal, pointCenter, pointsIn3D_with_center

# run on 2 to 7 clusters
numClustList = range(2, 8)

# lingDists is calculated once: it is only the mean linguistic distance between
#   points w/in 100 km, weighted by inverse distances
lingDists = []
# genDists contains the distances between individuals with genetic data, for each pairs of clusters,
#   weighted by inverse distance. This is calculated for each pair of clusters at each level of clustering.
genDists = [[] for i in numClustList]
# a list of the clusters compared in genDists, for each level of hierarchical clustering
clustsCompared = [[] for i in numClustList]

firstRun = True
pairwiseDistMtrxOrig = pairwiseDistMtrx.copy()
for numClusters in numClustList:    
    print("k = " + str(numClusters))
    # this matrix will be permuted for calculating significance values, so we store the original.
    pairwiseDistMtrx = pairwiseDistMtrxOrig.copy()
    cutEnglish = cut_tree(numClusters, britain_tree)
    # identify which individuals are labeled with each cluster
    nodeIndivs = []
    for node in cutEnglish:
        print(node.name) # , end="\t #indivs: ")
        nodeIndivs.append(np.where(
            [(indiv in node.get_leaf_names()) for indiv in geneticLocations[:, 1]]
        )[0])

    # for each of 29 genetic locations, how many individuals are there in each cluster?
    genClusteringLoc = np.zeros((len(locIdx), numClusters))
    # for each of 29 genetic locations, which linguistic locations are associated with it?
    lingLocMatrix = np.zeros((len(locIdx), featuresLing.shape[0]))
    for loc in range(len(locIdx)):
        lingLocMatrix[loc, locationIndicesLing[loc]] = 1 / len(locationIndicesLing[loc])

        for cluster in range(numClusters):
            genClusteringLoc[loc, cluster] = len(np.intersect1d(
                nodeIndivs[cluster], locationIndicesGen[loc]
            ))

    genClusteringLoc[genClusteringLoc < 3] = 0
    lingClusterWeights = ((genClusteringLoc.T / np.sum(genClusteringLoc, axis=1)) @
                          (lingLocMatrix / np.sum(lingLocMatrix, axis=0)))

    pairDifferencesSpatial = []
    pairDifferencesLing = []
    pairDifferencesGen = [
        [[] for _ in range(len(connectedPairs))]
        for _ in range(int(numClusters * (numClusters + 1) / 2))
    ]

    for randIter in range(NUMRAND + 1):
        # the first iteration, use the original matrix. In all others, use a permuted matrix
        if randIter == 0:
            pairwiseDistMtrx = pairwiseDistMtrxOrig.copy()
        else:
            reorder = np.random.permutation([i for i in range(pairwiseDistMtrxOrig.shape[0])])
            pairwiseDistMtrx = pairwiseDistMtrxOrig[reorder, :][:, reorder]
     
        pairCount = -1
        # for every connected pair, find the spatial rates of linguistic and genetic change between those locations
        for pair in connectedPairs:
            pairCount += 1
            diffHav = haversine(
                tuple(coordsLing[pair[0], :]),
                tuple(coordsLing[pair[1], :])) / 1000
            pairDifferencesSpatial.append(diffHav)

            # calculate linguistic differences between points
            diffLing = (featuresLing[pair[0], :] -
                        featuresLing[pair[1], :]) ** 2
            pairDifferencesLing.append(np.sum(diffLing) ** (1 / 2))
 
            i = -1
            pairLoc = np.where(lingLocMatrix[:, pair] > 0)[0]
            pairOfClustersUsed = []

            # loop through pairs of clusters and calculate genetic differences
            #   between pairs for sets of points found in either or both clusters
            for clust1 in range(numClusters):
                for clust2 in range(clust1, numClusters):
                    i += 1
                    pairOfClustersUsed.append((cutEnglish[clust1].name[1],
                                               cutEnglish[clust2].name[1]))
                    if not ((cutEnglish[clust1].name[0],
                             cutEnglish[clust2].name[0]) in
                                compareDict[str(numClusters)] or
                            ((cutEnglish[clust2].name[0],
                              cutEnglish[clust1].name[0]) in
                                compareDict[str(numClusters)]) or
                            (clust1 == clust2)):
                        pairDifferencesGen[i].append(0)
                        continue

                    # select individuals in each cluster and location
                    indivsWithin = [
                        np.intersect1d(
                            nodeIndivs[clust1],  # cluster 1
                            locationIndicesGen[pairLoc[0]]),  # location 1
                        np.intersect1d(
                            nodeIndivs[clust1],  # cluster 1
                            locationIndicesGen[pairLoc[1]]),  # location 2
                        np.intersect1d(
                            nodeIndivs[clust2],  # cluster 2
                            locationIndicesGen[pairLoc[0]]),  # location 1
                        np.intersect1d(
                            nodeIndivs[clust2],  # cluster 2
                            locationIndicesGen[pairLoc[1]])]  # location 2
                   
                    for n in range(4):
                        # remove location/node combinations with very few individuals (2 or fewer)
                        if len(indivsWithin[n]) < 3:
                            indivsWithin[n] = np.empty(0)
    
                    indivsAll = [
                        np.int64(  # all individuals in location 1 either cluster
                            np.union1d(indivsWithin[0],
                                       indivsWithin[2])),
                        np.int64(  # all individuals in location 2 and either cluster
                            np.union1d(indivsWithin[1],
                                       indivsWithin[3]))
                    ]
    
                    diffGen = 0
    
                    if all([len(indivs) != 0 for indivs in indivsAll]):
                        # calculate distances between clusters:
                        if clust1 != clust2:
                            # find the sum of differences between individuals in different clusters
                            #   and different locations.
                            diffGenBetween = []
                            if len(indivsWithin[0]) != 0 and len(indivsWithin[3]) != 0:
                                diffGenBetween.append(meanGenDistance(  # (loc1 cluster1 vs loc2 cluster2)
                                    pairwiseDistMtrx[tuple(np.meshgrid(indivsWithin[0], indivsWithin[3]))],
                                    "sum"))
                            if len(indivsWithin[1]) != 0 and len(indivsWithin[2]) != 0:
                                diffGenBetween.append(meanGenDistance(  # (loc2 cluster1 vs loc1 cluster2)
                                    pairwiseDistMtrx[tuple(np.meshgrid(indivsWithin[1], indivsWithin[2]))],
                                    "sum"))
                            if len(diffGenBetween) == 0:
                                diffGen = 0
                            else:
                                # average genetic distance between points. Note that the sum uses differences
                                #   between each pair of individuals, so the average must divide by the product
                                #   of the number of individuals compared
                                diffGen = np.mean(
                                    np.concatenate([n.reshape((-1, 1)) for n in diffGenBetween])
                                ) / (len(indivsAll[0]) * len(indivsAll[1]))
                        else:
                            # within clusters differences
                            diffGen = meanGenDistance(
                                pairwiseDistMtrx[tuple(np.meshgrid(
                                indivsAll[0], indivsAll[1]))],
                                "sum") / (len(indivsAll[0]) * len(indivsAll[1]))
    
                    pairDifferencesGen[i][pairCount].append(
                        diffGen)

    pairDifferencesSpatial = np.array(pairDifferencesSpatial)
    pairDifferencesLing = np.array(pairDifferencesLing)
    pairDifferencesGenTmp = np.zeros((len(pairDifferencesGen), len(connectedPairs)))

    n = 0
    m = -1
    for clusterPair in range(pairDifferencesGen.shape[0]):
        m += 1
        if m == numClusters:
            n += 1
            m = n

        # check if this pair of clusters should be compared
        if not ((cutEnglish[n].name[0],
                 cutEnglish[m].name[0]) in
                    compareDict[str(numClusters)] or
                ((cutEnglish[m].name[0],
                  cutEnglish[n].name[0]) in
                    compareDict[str(numClusters)]) or
                (n == m)):
            continue
        # standardize genetic and linguistic rates of change:
        meanDifferencesGen = []
        meanDifferencesLing = []
        for loc in range(coordsLing.shape[0]):
            indices = np.where([(loc in pair) for pair in connectedPairs])[0]

            if all(lingClusterWeights[
                    (n, m), :][:, loc] == 0):
                #  if no relevant clusters are at that location:
                meanDifferencesGen.append(-np.inf)
            else:
                if m != n:  # if clusters are different:
                    if len(notNan(pairDifferencesGen[tuple(np.meshgrid(clusterPair, indices))])) == 0:
                        # if all differences between the pair of clusters at these indices are NaN:
                        meanDifferencesGen.append(0)
                    else:
                        # the average genetic difference between two clusters, between each location and the
                        #   locations that connect to it (w/in 100 km), weighted by inverse distance
                        #   between points.
                        meanDifferencesGen.append(
                            np.average(
                                a=[np.mean(notNan(pairDifferencesGen[clusterPair, i]))
                                   for i in indices
                                   if len(notNan(pairDifferencesGen[clusterPair, i])) > 0],
                                weights=[
                                    pairDifferencesSpatial[i] ** -1
                                    for i in indices
                                    if len(notNan(pairDifferencesGen[clusterPair, i])) > 0
                                ]))
                else:
                    # average genetic distance between points in the same cluster:
                    if np.sum([pairDifferencesGen[clusterPair, i] != 0 for i in indices]):
                        meanDifferencesGen.append(
                            np.average(
                                a=[np.mean(notNan(pairDifferencesGen[clusterPair, i]))
                                   for i in indices
                                   if np.mean(notNan(pairDifferencesGen[clusterPair, i])) != 0],
                                weights=[
                                     pairDifferencesSpatial[i] ** -1
                                     for i in indices
                                     if np.mean(np.abs(notNan(pairDifferencesGen[clusterPair, i]))) != 0
                                 ]))
                    else:
                        meanDifferencesGen.append(0)
            meanDifferencesLing.append(np.mean(
                [pairDifferencesLing[i]
                 for i in indices]))

        meanDifferencesGen = np.round(np.array(meanDifferencesGen), 10)
        meanDifferencesLing = np.round(np.array(meanDifferencesLing), 10)

        lingDists = meanDifferencesLing.copy()
        genDists[numClusters - np.min(numClustList)].append(meanDifferencesGen.copy())
        clustsCompared[numClusters - np.min(numClustList)].append(pairOfClustersUsed[clusterPair])


# save calculated values as a list of contents:
with open("standardized_rates_list.pickle", "wb") as f:
    pickle.dump([lingDists, genDists, clustsCompared], f, protocol=pickle.HIGHEST_PROTOCOL)
