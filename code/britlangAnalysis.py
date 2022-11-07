import json
import pickle

import cartopy
import cartopy.crs as ccrs
from ete3 import Tree
from cHaversine import haversine
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import FancyArrow, Patch
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial import distance, procrustes
from scipy.stats import linregress, mannwhitneyu
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from sklearn.decomposition import PCA


# cut the tree into K clusters based on distance
def cut_tree(numclusters, tree):
    if type(numclusters) is not int:
        raise TypeError("k must be an integer greater than or equal to 2")
    if numclusters < 2:
        raise ValueError("k must be an integer greater than or equal to 2")

    clustersFound = tree.get_children()

    while len(clustersFound) < numclusters:

        distances = [tree.get_distance(node) for node in clustersFound]
        subtreeNode = clustersFound[distances.index(min(distances))]

        for childNode in subtreeNode.get_children():
            clustersFound.append(childNode)
        clustersFound.remove(subtreeNode)

    return clustersFound


# remove bad values from a numpy object
def notNan(x, y=None):
    if y is None:
        return x[~(np.isnan(x) | np.isinf(x))]
    else:
        return y[~(np.isnan(x) | np.isinf(x))]


def gt(x):
    # greater than the mean, that works with missing data
    return (x > np.mean(notNan(x))) & ~np.isinf(x)


def lte(x):
    # less than or equal to the mean, that works with missing data
    return (x <= np.mean(notNan(x))) & ~np.isinf(x)


def geo_to_xyz(mtrx):
    # converts a longitude-latitude matrix (each row is a pair of coordinates) to cartesian XYZ space
    lati = mtrx[:, 1] * np.pi / 180
    longi = (mtrx[:, 0] + 90) * np.pi / 180

    X = np.cos(lati) * np.cos(longi)
    Y = np.cos(lati) * np.sin(longi)
    Z = np.sin(lati)

    return np.c_[X, Y, Z]


def xyz_to_geo(mtrx):
    # converts a matrix in cartesian XYZ space (each row is a set of coordinates) to longitude-latitude values
    mtrxSum = np.sum(mtrx ** 2, axis=1)

    lati = np.arccos(mtrx[:, 2] / mtrxSum)
    lati *= - 180 / np.pi
    lati += 90

    longi = np.arctan2(mtrx[:, 1], mtrx[:, 0])
    longi *= 180 / np.pi
    longi -= 90

    return np.c_[longi, lati]


# for each town, its longitude and latitude:
coordsLing = np.genfromtxt("orig_elements.csv",
                           delimiter=',', skip_header=True, dtype=np.float64, usecols=(3, 2))
# for each town, its location code (region with genetic data):
locationsLing = np.genfromtxt("SED_PoBI_locations.csv", dtype=int, skip_header=True)
# for each town, its associated linguistic data:
featuresLing = np.genfromtxt("cmbnd_features_only.tsv", delimiter='\t')
# the original linguistic elements instead of the combined features
featuresLingOrig = np.genfromtxt("orig_elements.csv",
                                 delimiter=',', skip_header=True)[:, 14:]
featuresLingOrig = featuresLingOrig.astype(np.float64)

# genetic sharing matrix, AKA copying matrix, which is direct output from fineSTRUCTURE and contains
#  estimated length of the genome contributed from every individual to every other individual.
geneticSharing = np.genfromtxt("../../wombling_data/all_chr.chunklengths.out",
                               delimiter=" ", skip_header=True,
                               usecols=(range(1, 2040)))
geneticSharingIDs = np.genfromtxt("../../wombling_data/all_chr.chunklengths.out",
                                  delimiter=" ", skip_header=True,
                                  usecols=(0), dtype=str)
# clean geneticSharing to remove individuals from outside of the study area:
geneticSharing = geneticSharing[:, np.min(geneticSharing, axis=0) == 0]

# # for each individual with genetic data, the associated longitude and latitude:
geneticCoords = np.genfromtxt("../../tmp/all_chr.english.coords", delimiter="\t")

# sample-location codes and individual codes for the genetic data
geneticLocations = np.genfromtxt(
    "../../wtccc/_WTCCC2_POBI_illumina_calls_POBI_illumina.sampleLocations",
    delimiter="\t", skip_header=1, usecols=(1, 3), dtype=str)
# filter for locations within England
geneticLocations = geneticLocations[geneticLocations[:, 0].astype(int) < 30, :]

# reorder genetic sharing matrix:
orderedIDs = [np.where(geneticSharingIDs == id_)[0][0]
              for id_ in geneticLocations[:, 1]]
geneticSharing = geneticSharing[orderedIDs, :]


del geneticSharingIDs, orderedIDs

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

britain_tree = Tree("/home/pichkary/Downloads/output_to_return/all_chr_newick_tree")
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
compareDict = {'2': [('A1', 'A2')],
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

# make polygons for plotting points:
with open("../../tmp/uk-ceremonial-counties.geojson", "r") as f:
    counties = json.load(f)

# for each county in the genetic dataset, which polygons from the uk-ceremonial-counties file are relevant?
polygonCounties = {
    '1': [42],
    '2': [19],
    '3': [20],
    '4': [27, 44],
    '5': [39],
    '6': [26],
    '7': [24],
    '8': [43, 16, 17],
    '9': [0, 37, 21, 22, 23, 25, 28, 30],
    '10': [36],
    '11': [32],
    '12': [34],
    '13': [18],
    '14': [66, 46, 41],
    '15': [5, 38, 40],
    '16': [33],
    '17': [31, 14],
    '18': [13, 35],
    '19': [9],
    '20': [1, 2, 10],
    '21': [15],
    '22': [12],
    '23': [11],
    '24': [3, 6],
    '25': [7],
    '26': [8],
    '27': [29],
    '28': [4],
    '29': [45]
}
for i in range(1, 30):
    cpoly = []
    for c in polygonCounties[str(i)]:
        if counties['features'][c]['geometry']['type'] == 'MultiPolygon':
            for j in range(len(counties['features'][c]['geometry']['coordinates'])):
                cpoly.append(
                    Polygon(counties['features'][c]['geometry']['coordinates'][j][0])
                )
        else:
            cpoly.append(Polygon(counties['features'][c]['geometry']['coordinates'][0]))
    polygonCounties[str(i)] = unary_union(cpoly)

del cpoly

# # optional: plot a map with each of the unified polygons
# RES = 1.5
# plt.figure(figsize=(15 * RES, 15 * RES))
# ax = plt.axes(projection=ccrs.PlateCarree())
# plt.xlim([-6, 2])
# plt.ylim([49, 58])
# ax.add_feature(cartopy.feature.LAND, color='#808080')
# for i in range(1, 30):
#     ax.add_geometries([polygonCounties[str(i)]],
#                       crs=ccrs.PlateCarree(), facecolor='none', edgecolor='r', alpha=1,
#                       linewidth=0.75 * RES, zorder=2)

# assign a location to each genetic individual at random within the county that they are assigned to
np.random.seed(212902871)
pointLocations = []
pointLocationsTrue = []
genApproxLingLoc = []
for loc in geneticLocations[:, 0]:
    p = Point([0, 0])
    #     plt.scatter(*[polygonCounties[loc].centroid.xy[i][0]
    #                   for i in range(2)], c='k', s=10, zorder=5)
    pointLocationsTrue.append([
        polygonCounties[loc].centroid.xy[i][0]
        for i in range(2)])
    while not polygonCounties[loc].contains(p):
        p = Point(
            [polygonCounties[loc].centroid.xy[i][0] + np.random.randn() * 1.5
             for i in range(2)]
        )
    pointLocations.append([
        p.xy[i][0]
        for i in range(2)])
    genApproxLingLoc.append(
        (lambda l: l.index(min(l)))([
            haversine(tuple(pointLocations[-1]),
                      tuple(coordsLing[i, :]))
            for i in range(coordsLing.shape[0])]))

plt.show()

# load results from permutated data testing
with open("standardized_rates_list.pickle", "rb") as f:
    linguistic_rates, genetic_rates_percomparison, comparison_names = pickle.load(f)


def ellipse(x, y, rx, ry, angle, n=200):
    theta = np.linspace(0, 2 * np.pi, np.round(n).astype(np.int64))
    el = np.array([rx * np.cos(theta), ry * np.sin(theta)])

    angle *= -np.pi / 180
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    el = rot @ el
    el[0, :] += x
    el[1, :] += y

    return el[0], el[1]


# for each possible cluster, present appropriate coordinates and dimensions for an ellipse that encompasses
#  most indivs in that cluster. Also, rotation of that ellipse and its color for plotting.
# (entirely graphical; no analytical purpose)
ellipseDict = {
    # x, y, x-radius, y=radius, angle
    'A1': [-1.25, 52.5, 1.85, 2.25, 20, 'red'],
    'A2': [-2.35, 54.9, 1.25, 0.5, -20, 'orange'],
    'A3': [-1.75, 52.25, 1.75, 2.25, 30, 'red'],
    'A4': [-0.75, 52.5, 1.75, 2, 10, 'red'],
    'A5': [-0.65, 52.5, 1.75, 2, -15, 'red'],
    'V0': [-4.95, 50.35, 0.7, 0.45, -15, 'pink'],
    'V1': [-2.65, 52.2, 0.6, 0.75, -5, 'purple'],
    'V2': [-3.75, 50.75, 0.75, 0.5, -5, 'blue'],
    'V3': [-0.65, 52.25, 1.65, 1.8, 0, 'red'],
    'V4': [-1.85, 54, 1.15, 0.45, -15, 'teal'],
    'V5': [-2.75, 54.75, 0.75, 0.5, -25, '#ff8447'],  # orange
    'V6': [-1.55, 55, 0.35, 0.65, -25, "#ffcc00"]  # yellow
}

# Long names for each possible cluster (K=2 to K=7)
# (entirely graphical; no analytical purpose)
clusterNameDict = {
    'A1': "Central/South England",
    'A2': "Scottish Marches",
    'A3': "Central/South England",
    'A4': "Central/South England",
    'A5': "Central/South England",
    'V0': "Cornwall",
    'V1': "Welsh Marches",
    'V2': "Somerset",
    'V3': "Central/South \nEngland",
    'V4': "Yorkshire",
    'V5': "NW Scot. Marches",
    'V6': "NE Scot. Marches"
}

# for each number of clusters, for each comparison between clusters, appropriate coordinates for an arrow between them.
# (entirely graphical; no analytical purpose)
arrowDict = {'2': [([-1.6, 54.12], [-2., 54.8])],
             '3': [([-1.68, 54.1], [-2.08, 54.75]),
                   ([-3.35, 51.0], [-4.45, 50.6])],
             '4': [([-1.5, 54.1], [-1.9, 54.75]),
                   ([-2.55, 51.05], [-4.05, 50.7]),
                   ([-1.9, 52.2], [-2.65, 52.15])],
             '5': [([-1.5, 54.1], [-1.9, 54.75]),  # scot C.eng
                   ([-4.15, 50.75], [-4.7, 50.5]),  # crw somer
                   ([-3.1, 50.9], [-1.95, 51.3]),  # C.eng Somer
                   ([-3.4, 50.95], [-2.9, 51.9]),  # Wlsh somer
                   ([-1.85, 52.25], [-2.6, 52.2])],
             '6': [([-0.9, 53.55], [-1.7, 54.85]),  # scot C.eng
                   ([-1.5, 53.3], [-1.85, 53.8]),  # WYork C.eng
                   ([-2.35, 54], [-2.4, 54.7]),  # WYork scot
                   ([-3.1, 50.9], [-1.95, 51.3]),  # Somer C.eng
                   ([-4.15, 50.75], [-4.7, 50.5]),  # crw somer
                   ([-2.35, 53.7], [-2.65, 52.75]),  # Wlsh WYork
                   ([-3.4, 50.95], [-2.9, 51.9]),  # Wlsh Somer
                   ([-1.85, 52.2], [-2.6, 52.15])],
             '7': [([-0.9, 53.75], [-1.38, 54.6]),  # C.eng Northumbr
                   ([-1.85, 53.8], [-1.5, 53.3]),  # WYork C.eng
                   ([-2.5, 53.96], [-2.7, 54.7]),  # WYork Cumbria
                   ([-1.8, 54.15], [-1.61, 54.75]),  # WYork Northumbr
                   ([-3.1, 50.9], [-1.95, 51.3]),  # Somer C.eng
                   ([-4.15, 50.75], [-4.7, 50.5]),  # crw somer
                   ([-2.35, 53.7], [-2.65, 52.75]),  # Wlsh WYork
                   ([-3.4, 50.95], [-2.9, 51.9]),  # Wlsh Somer
                   ([-1.85, 52.2], [-2.6, 52.15]),  # Wlsh C.eng
                   ([-2.37, 54.89], [-1.65, 55.07])]
             }

np.random.seed(212902871)
# Generate a random order for plotting points, so that no cluster is consistently plotted over top of others.
# (entirely graphical; no analytical purpose)
randOrder = np.random.permutation([
    i for i in range(geneticLocations.shape[0])])

# loop through k=2 to k=7 to plot Figure 2 and Figures S1-5
for k in range(2, 8):
    RES = 1
    offset = (0.5, 0.5, 0.5, 0.5)
    plt.figure(figsize=(15 * RES, 15 * RES))
    ax = plt.axes(projection=ccrs.AlbersEqualArea(central_latitude=51, central_longitude=-1))
    ax.set_extent([
        coordsLing[:, 0].min() - offset[0],  # Longitude is X-coords
        coordsLing[:, 0].max() + offset[0],
        coordsLing[:, 1].min() - offset[1],  # Latitude is Y-coords
        coordsLing[:, 1].max() + offset[1]
    ], crs=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')

    ct = cut_tree(k, britain_tree)
    nodeIndivs = [node.get_leaf_names() for node in ct]
    nodeNames = [node.name[0] for node in ct]
    indivClustColor = lambda indiv: ellipseDict[
        nodeNames[[(indiv in n) for n in nodeIndivs].index(True)]
    ][-1]

    # plot each individual in an appropriate location, colored by its cluster
    for i in randOrder:
        plt.scatter(*(pointLocations[i]), s=250 * RES ** 2,
                    c=indivClustColor(geneticLocations[i, 1]),
                    marker='o', edgecolors='none', zorder=4, alpha=0.7,
                    transform=ccrs.PlateCarree())

    # for each cluster c of k clusters, check for a gene-language relationship within that cluster
    for c in range(k):
        cname = ct[c].name[0]
        for n in range(len(comparison_names[k - 2])):
            if comparison_names[k - 2][n][0] == comparison_names[k - 2][n][1] and \
                    comparison_names[k - 2][n][0] == [subtree.name[1] for subtree in ct
                                                      if cname == subtree.name[0]][0]:
                break
        # create ellipses approximately in the locations that each cluster is found.
        plt.plot(*ellipse(*ellipseDict[cname][:-1]), color='w',
                 alpha=0.8, lw=7 * RES, zorder=5,
                 transform=ccrs.PlateCarree())
        plt.plot(*ellipse(*ellipseDict[cname][:-1]), color=ellipseDict[cname][-1],
                 lw=5 * RES, zorder=5,
                 transform=ccrs.PlateCarree())
        try:
            # if the genetic cluster is found throughout multiple locations,
            #  determine whether the linguistic rates of change are correlated with genetic rates or change
            sig = linregress(
                linguistic_rates[~np.isinf(genetic_rates_percomparison[k - 2][n])],
                genetic_rates_percomparison[k - 2][n][~np.isinf(genetic_rates_percomparison[k - 2][n])],
                alternative='greater').pvalue
        except ValueError:
            sig = np.inf
        print(cname, sig)
        if sig < 0.05 / (k - np.sum([len(np.unique(n)) for n in genetic_rates_percomparison[k - 2]] == 1)):
            plt.plot(
                *ellipse(*ellipseDict[cname][:-1] * np.array([1, 1, 0.95, 0.95, 1])),
                color='w', alpha=0.75, lw=9 * RES, zorder=5,
                transform=ccrs.PlateCarree())
            plt.plot(
                *ellipse(*ellipseDict[cname][:-1] * np.array([1, 1, 0.95, 0.95, 1])),
                color=ellipseDict[cname][-1],
                lw=7 * RES, ls=":", zorder=5,
                transform=ccrs.PlateCarree())

    # for each pair of clusters 'p' that are being compared, find whether there is a relationship between high
    #  rates of linguistic change and high rates of genetic change
    print(len(arrowDict[str(k)]))
    print(arrowDict[str(k)])
    for p in range(len(compareDict[str(k)])):
        pair = compareDict[str(k)][p]
        for n in range(len(comparison_names[k - 2])):
            n1 = [c.name[1] for c in ct if pair[0] == c.name[0]][0]
            n2 = [c.name[1] for c in ct if pair[1] == c.name[0]][0]
            if n1 in comparison_names[k - 2][n] and \
                    n2 in comparison_names[k - 2][n]:
                break

        print(pair[0], pair[1], comparison_names[k - 2][n],
              np.round(sig / len(compareDict[str(k)]), 5),
              len(compareDict[str(k)]))

        point1, point2 = arrowDict[str(k)][p]
        midpoint = [(point1[i] + point2[i]) / 2 for i in range(2)]

        n1 = [pair[0] == c.name[0] for c in ct].index(True)
        n2 = [pair[1] == c.name[0] for c in ct].index(True)

        print("{} {}; n1 = {}  n2 = {}".format(k, n, n1, n2))
        # if regions with high linguistic rates of change also contain disproportionally high genetic rates
        sig = mannwhitneyu(
            linguistic_rates[lte(genetic_rates_percomparison[k - 2][n])],
            linguistic_rates[gt(genetic_rates_percomparison[k - 2][n])],
            alternative='less'
        )[1] * len(compareDict[str(k)])  # bonferonni correction
        sig = (sig < 0.05) * 0.8 + 0.2

        # plotting arrows
        patch_crs = ccrs.PlateCarree()
        ax.add_patch(
            FancyArrow(
                *midpoint, *[(point1[i] - midpoint[i]) * 1.06 for i in range(2)],
                width=0.07, head_width=3 * 0.07, head_length=2.5 * 0.07,
                facecolor="w", length_includes_head=True,
                alpha=1, zorder=5, edgecolor='w', lw=3,
                transform=patch_crs))
        ax.add_patch(
            FancyArrow(
                *midpoint, *[(point2[i] - midpoint[i]) * 1.06 for i in range(2)],
                width=0.07, head_width=3 * 0.07, head_length=2.5 * 0.07,
                facecolor="w", length_includes_head=True,
                alpha=1, zorder=5, edgecolor='w', lw=3,
                transform=patch_crs))
        ax.add_patch(
            FancyArrow(
                *midpoint, *[point1[i] - midpoint[i] for i in range(2)],
                width=0.07, head_width=3 * 0.07, head_length=2.5 * 0.07,
                facecolor=ellipseDict[pair[0]][-1], length_includes_head=True,
                alpha=sig, zorder=6, edgecolor='none',
                transform=patch_crs))
        ax.add_patch(
            FancyArrow(
                *midpoint, *[point2[i] - midpoint[i] for i in range(2)],
                width=0.07, head_width=3 * 0.07, head_length=2.5 * 0.07,
                facecolor=ellipseDict[pair[1]][-1], length_includes_head=True,
                alpha=sig, zorder=6, edgecolor='none',
                transform=patch_crs))

    # figure legend
    ax.legend(handles=[
        Patch(facecolor=ellipseDict[ct[c].name[0]][-1],
              edgecolor='white', label=clusterNameDict[ct[c].name[0]])
        for c in range(k)],
        loc="upper right", fontsize=22)

    plt.tight_layout()
    # plt.savefig(fname="clusterResultsOut/K{}.pdf".format(k), dpi=300, format="pdf")
    # plt.savefig(fname="clusterResultsOut/K{}.png".format(k), dpi=300, format="png")
    plt.show()

del pair, point1, point2, c, k, indivClustColor

# Plot figure 1
RES = 1
plt.figure(figsize=(17 * RES, 15 * RES))
offset = (0.5, 0.5)
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_latitude=51, central_longitude=-1))
ax.set_extent([
    #     -3.5, 1.5, 50.5, 54
    coordsLing[:, 0].min() - offset[0],  # Longitude is X-coords
    coordsLing[:, 0].max() + offset[0],
    coordsLing[:, 1].min() - offset[1],  # Latitude is Y-coords
    coordsLing[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')

lingClustering = hierarchy.linkage(
    distance.pdist(featuresLing),
    method='ward')
lingClustering = np.array([i[0] for i in hierarchy.cut_tree(lingClustering, 6)])
markermap = np.array(['s', 'P', 'D', '*', '^', 'o'])

# plt.title("Linguistic rate of change")
for i in np.unique(lingClustering):
    plt.scatter(
        coordsLing[lingClustering == i, 0],
        coordsLing[lingClustering == i, 1],
        c=linguistic_rates[lingClustering == i], marker=markermap[i],
        cmap=get_cmap('Greens'),
        s=250 * RES ** 2, zorder=5,
        transform=ccrs.PlateCarree())
cb = plt.colorbar()
cb.ax.set_ylabel('Average linguistic rate of change\n(number of features)', rotation=270, fontsize=45, labelpad=120)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(35)

plt.tight_layout()
plt.savefig(fname="../../clusterResultsOut/fig1.pdf", dpi=300, format="pdf")
plt.savefig(fname="../../clusterResultsOut/fig1.png", dpi=300, format="png")
plt.show()

# Plot colored linguistic clustering
RES = 1
plt.figure(figsize=(17 * RES, 15 * RES))
offset = (0.5, 0.5)
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_latitude=51, central_longitude=-1))
ax.set_extent([
    #     -3.5, 1.5, 50.5, 54
    coordsLing[:, 0].min() - offset[0],  # Longitude is X-coords
    coordsLing[:, 0].max() + offset[0],
    coordsLing[:, 1].min() - offset[1],  # Latitude is Y-coords
    coordsLing[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')

colormap = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'cyan'])

for i in np.unique(lingClustering):
    plt.scatter(
        coordsLing[lingClustering == i, 0],
        coordsLing[lingClustering == i, 1],
        c=colormap[i], edgecolors=None,
        s=250 * RES ** 2, zorder=5,
        transform=ccrs.PlateCarree(),
        label="Ling. cluster " + str(i + 1)
    )

legend = plt.legend()
# sort labels by name
legend = plt.legend(
    handles=[
        legend.legendHandles[i]
        for i in np.argsort([
            handle.get_label()
            for handle in legend.legendHandles])],
    loc="upper right", fontsize=22,
    title="Linguistic\nclusters", title_fontsize=24)

plt.tight_layout()
plt.savefig(fname="../../clusterResultsOut/lingClustersColored.pdf", dpi=300, format="pdf")
plt.savefig(fname="../../clusterResultsOut/lingClustersColored.png", dpi=300, format="png")
plt.show()

# plot Figure 3, Figure S6
nodeIndivs = []
ct = cut_tree(6, britain_tree)
for node in ct:
    n = node.get_leaf_names()
    nodeIndivs.append(np.where([
        (indiv in n) for indiv in geneticLocations[:, 1]
    ])[0])

    del n

# re-generate data to be used for gene-geography and gene-language procrustes:
genClusteringLoc = np.zeros((len(locIdx), len(ct)))
lingLocMatrix = np.zeros((len(locIdx), featuresLing.shape[0]))
for loc in range(len(locIdx)):
    lingLocMatrix[loc, locationIndicesLing[loc]] = 1 / len(locationIndicesLing[loc])

    for cluster in range(len(ct)):
        genClusteringLoc[loc, cluster] = len(np.intersect1d(
            nodeIndivs[cluster], locationIndicesGen[loc]
        ))

genClusteringLoc[genClusteringLoc < 3] = 0
lingClusterWeights = ((genClusteringLoc.T / np.sum(genClusteringLoc, axis=1)) @
                      (lingLocMatrix / np.sum(lingLocMatrix, axis=0)))

np.random.seed(212902870)
lingColors = [
    np.random.choice([
        ellipseDict[ct[j].name[0]][-1] for j in range(len(ct))
    ], p=lingClusterWeights[:, i])
    for i in range(len(coordsLing))]

del lingClusterWeights, genClusteringLoc, lingLocMatrix


def indivClustColor(indiv):
    # for each individual with genetic data, check that indiv's cluster, and return the color value from ellipseDict
    clustNum = [(indiv in n) for n in nodeIndivs].index(True)
    return ellipseDict[
        [node.name[0] for node in ct][clustNum]
    ][-1]


# procrustes processing
NPERM = 1000

# linguistics-geography procrustes
coordsXYZ = geo_to_xyz(coordsLing)
mean_coordsXYZ = np.mean(coordsXYZ, axis=0)
coordsXYZ -= mean_coordsXYZ
pca = PCA(n_components=3)
lingPC = pca.fit_transform(featuresLing)

new_coords_ling, val = procrustes(coordsXYZ, lingPC)[1:]
new_coords_ling += mean_coordsXYZ
new_coords_ling = xyz_to_geo(new_coords_ling)

# significance test
sig = 0
print("linguistics-geography: procr-correlation={}".format(1 - (val ** 2)))
valhistLing = []
for i in range(NPERM):
    lingPC = np.random.permutation(lingPC)
    valhistLing.append(procrustes(coordsXYZ, lingPC)[2])
    sig += (val >= valhistLing[-1])

print("procrustes permutation significance: {}; permutations={}".format(sig / NPERM, NPERM))

# plotting
RES = 1
plt.figure(figsize=(15 * RES, 15 * RES))
offset = (1.5, 1.5)
ax = plt.axes(projection=ccrs.Miller())
ax.set_extent([
    new_coords_ling[:, 0].min() - offset[0],  # Longitude is X-coords
    new_coords_ling[:, 0].max() + offset[0],
    new_coords_ling[:, 1].min() - offset[1],  # Latitude is Y-coords
    new_coords_ling[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')
# ax.add_feature(cartopy.feature.OCEAN, color='#C0E0FF')

# plt.title("Linguistic procrustes (First 3 PCs, feature matrix)"
#           " (r2={:.4f}, p={})".format(1 - val ** 2, sig / NPERM))
for i in np.random.permutation([i for i in range(len(coordsLing))]):
    plt.scatter(new_coords_ling[i, 0], new_coords_ling[i, 1],
                color=lingColors[i], marker=markermap[lingClustering[i]],
                zorder=3, s=250 * RES ** 2, edgecolors='none',
                transform=ccrs.PlateCarree())

ax.legend(handles=[
    Patch(facecolor=ellipseDict[ct[c].name[0]][-1],
          edgecolor='white', label=clusterNameDict[ct[c].name[0]])
    for c in range(6)],
    loc="upper right", fontsize=22)

plt.tight_layout()
plt.savefig(fname="../../clusterResultsOut/lingGeoProcr.pdf", dpi=300, format="pdf")
plt.savefig(fname="../../clusterResultsOut/lingGeoProcr.png", dpi=300, format="png")
plt.show()

# plotting linguistic-geographic procrustes with geographic coloring

# color points based on a yellow-cyan-red-blue coloring
#  inspired by Steiger et al. (WSCG 2015 Explorative Analysis of 2D Color Maps)
yellowCyanRedBlue = [
    np.array(x)
    for x
    in [[1, 1, 0],
        [0, 1, 0.8],
        [1, 0, 0],
        [0, 0, 1]]]
lingColors = [
    (lambda n: n / np.sum(n))(np.array([
        np.exp(-(3 * (1 - x[0]) * (1 - x[1])) ** 2),
        np.exp(-(3 * x[0] * (1 - x[1])) ** 2),
        np.exp(-(3 * (1 - x[0]) * x[1]) ** 2),
        np.exp(-(3 * x[0] * x[1]) ** 2)
    ])) @ yellowCyanRedBlue
    for x in zip(
        coordsLing[:, 1].argsort().argsort() / coordsLing.shape[0],
        coordsLing[:, 0].argsort().argsort() / coordsLing.shape[0])
]
lingColors = [
    (lambda n: 0.9 * n / np.max(n))(x + (x - np.mean(x)) ** 2)
    for x in lingColors
]

RES = 1
plt.figure(figsize=(15 * RES, 15 * RES))
offset = (1.5, 1.5)
ax = plt.axes(projection=ccrs.Miller())
ax.set_extent([
    new_coords_ling[:, 0].min() - offset[0],  # Longitude is X-coords
    new_coords_ling[:, 0].max() + offset[0],
    new_coords_ling[:, 1].min() - offset[1],  # Latitude is Y-coords
    new_coords_ling[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')
# ax.add_feature(cartopy.feature.OCEAN, color='#C0E0FF')

# plt.title("Linguistic procrustes (First 3 PCs, feature matrix)"
#           " (r2={:.4f}, p={})".format(1 - val ** 2, sig / NPERM))

lablist = []
for i in np.random.permutation([i for i in range(len(coordsLing))]):
    if lingClustering[i] not in lablist:
        lablist.append(lingClustering[i])
        l = "Cluster " + str(lingClustering[i] + 1)
    else:
        l = None
    plt.scatter(  # coordsLing[i, 0], coordsLing[i, 1],
        new_coords_ling[i, 0], new_coords_ling[i, 1], alpha=0.9,
        marker=markermap[lingClustering[i]], color=lingColors[i],
        zorder=3, s=250 * RES ** 2, transform=ccrs.PlateCarree(),
        edgecolors='none', label=l)
#     plt.plot((new_coords_ling[i, 0], coordsLing[i, 0]),
#              (new_coords_ling[i, 1], coordsLing[i, 1]),
#              '-', color=lingColors[i], alpha=0.5,
#              linewidth=0.5, zorder=2,
#              transform=ccrs.PlateCarree())

legend = plt.legend()
for i in range(len(legend.legendHandles)):
    legend.legendHandles[i].set_color('k')
legend = plt.legend(
    handles=[
        legend.legendHandles[i]  # sort by label name
        for i in np.argsort([
            handle.get_label()
            for handle in legend.legendHandles])],
    loc="upper left", fontsize=22,
    title="Linguistic\nclusters", title_fontsize=24)

plt.tight_layout()
plt.savefig(fname="../../clusterResultsOut/lingGeoProcr_geographicColoring.pdf", dpi=300, format="pdf")
plt.savefig(fname="../../clusterResultsOut/lingGeoProcr_geographicColoring.png", dpi=300, format="png")
plt.show()

# plot the geographic coloring used for figure S6
RES = 1
plt.figure(figsize=(15 * RES, 15 * RES))
offset = (1.5, 1.5)
ax = plt.axes(projection=ccrs.Miller())
ax.set_extent([
    coordsLing[:, 0].min() - offset[0],  # Longitude is X-coords
    coordsLing[:, 0].max() + offset[0],
    coordsLing[:, 1].min() - offset[1],  # Latitude is Y-coords
    coordsLing[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')

# for each linguistic cluster identified, plot the towns that fall into that cluster and
#  assign them a label for the legend
lablist = []
for i in np.random.permutation([i for i in range(len(coordsLing))]):
    if lingClustering[i] not in lablist:
        lablist.append(lingClustering[i])
        l = "Cluster " + str(lingClustering[i] + 1)
    else:
        l = None
    ax.scatter(coordsLing[i, 0], coordsLing[i, 1], alpha=0.9,
               color=lingColors[i], marker=markermap[lingClustering[i]],
               zorder=3, s=250 * RES ** 2, transform=ccrs.PlateCarree(),
               edgecolors='none', label=l)

del l, lablist

legend = plt.legend()
# make all labels black
for i in range(len(legend.legendHandles)):
    legend.legendHandles[i].set_color('k')
legend = plt.legend(
    handles=[
        legend.legendHandles[i]  # sort by label name
        for i in np.argsort([
            handle.get_label()
            for handle in legend.legendHandles])],
    loc="upper left", fontsize=22,
    title="Linguistic\nclusters", title_fontsize=24)

plt.tight_layout()
plt.savefig(fname="clusterResultsOut/geographicColoring.png", dpi=300, format="png")
plt.savefig(fname="clusterResultsOut/geographicColoring.pdf", dpi=300, format="pdf")
plt.show()

# genetics-geography procrustes
pointLocationsTrue = np.array(pointLocationsTrue)
coordsXYZ = geo_to_xyz(pointLocationsTrue)
mean_coordsXYZ = np.mean(coordsXYZ, axis=0)
coordsXYZ -= mean_coordsXYZ
pca = PCA(n_components=3)

genPC = pca.fit_transform(np.sqrt(geneticSharing))

new_coords_gen, val = procrustes(coordsXYZ, genPC)[1:]
new_coords_gen += mean_coordsXYZ
new_coords_gen = xyz_to_geo(new_coords_gen)

# significance test
sig = 0
print("Full-sample genetics-geography: procr-correlation={}".format(1 - (val ** 2)))
valhistGen = []
for i in range(NPERM):
    genPC = np.random.permutation(genPC)
    valhistGen.append(procrustes(coordsXYZ, genPC)[2])
    sig += (val >= valhistGen[-1])

print("procrustes permutation significance: {}; permutations={}".format(sig / NPERM, NPERM))


# calculating procrustes correlation for resampled genetic data (311 of 1667 indivs)
def selectRandomGenforLingProcrustes(true_random=False):
    """
    Returns indices to use for sampling genetic indivs.
    trueRandom: should the sampling account for the number of towns with linguistic data for each
                region with genetic data?
    """
    if not true_random:
        newIdx = [
            np.random.choice(locationIndicesGen[i - 1])
            for i in locationsLing
        ]
        return np.array(newIdx)
    else:
        newIdx = np.random.choice(
            geneticLocations.shape[0],
            size=coordsLing.shape[0],
            replace=True)
        return newIdx


NUMRESAMPLE = 500
NPERM = 500 * 500
sig = []
vals = []
for _ in range(NUMRESAMPLE):
    resample = selectRandomGenforLingProcrustes()

    genPC = pca.fit_transform(np.sqrt(geneticSharing[resample, :]))
    vals.append(procrustes(coordsXYZ[resample, :], genPC)[2])

    valhistGen = []
    for i in range(int(np.round(NPERM / NUMRESAMPLE))):
        genPC = np.random.permutation(genPC)
        valhistGen.append(procrustes(coordsXYZ[resample, :], genPC)[2])

    sig.append(np.sum(np.array(valhistGen) < vals[-1]) / len(valhistGen))

print("Resampled genetics-geography: p-value after permuation:"
      "median={}, mean={}".format(np.median(sig), np.mean(sig)))
plt.hist(np.array(sig))
plt.title("Resampled genetics-geography p-values")
plt.show()

print("Resampled genetics-geography: procr-correlation={}".format(1 - (np.mean(vals) ** 2)))
plt.hist(1 - np.array(vals) ** 2)
plt.title("Resampled genetics-geography: procr-correlation")
plt.show()

# genetics-linguistics procrustes
lingPC = pca.fit_transform(featuresLing)
sig = []
vals = []
for _ in range(NUMRESAMPLE):
    resample = selectRandomGenforLingProcrustes()

    genPC = pca.fit_transform(np.sqrt(geneticSharing[resample, :]))
    vals.append(procrustes(lingPC, genPC)[2])

    valhistGen = []
    for i in range(int(np.round(NPERM / NUMRESAMPLE))):
        genPC = np.random.permutation(genPC)
        valhistGen.append(procrustes(lingPC, genPC)[2])

    sig.append(np.sum(np.array(valhistGen) < vals[-1]) / len(valhistGen))

print("Resampled genetics-linguistics: p-value after permuation:"
      "median={}, mean={}".format(np.median(sig), np.mean(sig)))
plt.hist(np.array(sig))
plt.title("Resampled genetics-linguistics p-values")
plt.show()

print("Resampled genetics-linguistics: procr-correlation={}".format(1 - (np.mean(vals) ** 2)))
plt.hist(1 - np.array(vals) ** 2)
plt.title("Resampled genetics-linguistics: procr-correlation")
plt.show()

#
RES = 1
plt.figure(figsize=(15 * RES, 15 * RES))
offset = (1.5, 1.5)
ax = plt.axes(projection=ccrs.Miller())
ax.set_extent([
    new_coords_ling[:, 0].min() - offset[0],  # Longitude is X-coords
    new_coords_ling[:, 0].max() + offset[0],
    new_coords_ling[:, 1].min() - offset[1],  # Latitude is Y-coords
    new_coords_ling[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')

# plt.title("Genetic procrustes (First 3 PCs, sqrt(genetic sharing/copying matrix))"
#          " (r2={:.4f}, p={})".format(1 - val ** 2, sig / NPERM))
for i in randOrder:
    # plot points according to their procrustes-transformed coordinates
    plt.scatter(new_coords_gen[i, 0], new_coords_gen[i, 1], zorder=3,
                color=indivClustColor(i), marker=markermap[lingClustering[genApproxLingLoc[i]]],
                alpha=0.7, s=250 * RES ** 2, edgecolors='none',
                transform=ccrs.PlateCarree())

ax.legend(handles=[
    Patch(facecolor=ellipseDict[ct[c].name[0]][-1],
          edgecolor='white', label=clusterNameDict[ct[c].name[0]])
    for c in range(6)],
    loc="upper right", fontsize=22)

#     plt.plot((new_coords_gen[i, 0], pointLocationsTrue[i, 0]),
#              (new_coords_gen[i, 1], pointLocationsTrue[i, 1]),
#              '-', linewidth = 0.2, zorder=2,
#              c=indivClustColor(i))

plt.tight_layout()
plt.savefig(fname="../../clusterResultsOut/genGeoProcr.png", dpi=300, format="png")
plt.savefig(fname="../../clusterResultsOut/genGeoProcr.pdf", dpi=300, format="pdf")
plt.show()

# Plot the same figure, but with geographic coloring instead of genetic cluster colors:
RES = 1
plt.figure(figsize=(15 * RES, 15 * RES))
offset = (1.5, 1.5)
ax = plt.axes(projection=ccrs.Miller())
ax.set_extent([
    new_coords_ling[:, 0].min() - offset[0],  # Longitude is X-coords
    new_coords_ling[:, 0].max() + offset[0],
    new_coords_ling[:, 1].min() - offset[1],  # Latitude is Y-coords
    new_coords_ling[:, 1].max() + offset[1]
], crs=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.LAND, color='#C0C0C0')
# ax.add_feature(cartopy.feature.OCEAN, color='#C0E0FF')

# plt.title("Genetic procrustes (First 3 PCs, sqrt(genetic sharing/copying matrix))"
#          " (r2={:.4f}, p={})".format(1 - val ** 2, sig / NPERM))
genColors = [
    (np.mean(pointLocationsTrue[i, 1] > coordsLing[:, 1]),
     np.mean(pointLocationsTrue[i, 0] > coordsLing[:, 0]))
    for i in range(len(pointLocationsTrue[:, 0]))]
yellowCyanRedBlue = [
    np.array(x)
    for x in [[1, 1, 0],
              [0, 1, 0.8],
              [1, 0, 0],
              [0, 0, 1]]]
genColors = [
    (lambda n: n / np.sum(n))(np.array([
        np.exp(-(3 * (1 - x[0]) * (1 - x[1])) ** 2),
        np.exp(-(3 * x[0] * (1 - x[1])) ** 2),
        np.exp(-(3 * (1 - x[0]) * x[1]) ** 2),
        np.exp(-(3 * x[0] * x[1]) ** 2)
    ])) @ yellowCyanRedBlue
    for x in genColors
]
genColors = [
    (lambda n: 0.9 * n / np.max(n))(x + (x - np.mean(x)) ** 2)
    for x in genColors]

for i in randOrder:
    plt.scatter(new_coords_gen[i, 0], new_coords_gen[i, 1], zorder=3,
                color=genColors[i], marker=markermap[lingClustering[genApproxLingLoc[i]]],
                alpha=0.7, s=250 * RES ** 2, edgecolors='none', transform=ccrs.PlateCarree())

plt.tight_layout()
plt.savefig(fname="../../clusterResultsOut/genGeoProcr_geographicColoring.png", dpi=300, format="png")
plt.savefig(fname="../../clusterResultsOut/genGeoProcr_geographicColoring.pdf", dpi=300, format="pdf")
plt.show()
