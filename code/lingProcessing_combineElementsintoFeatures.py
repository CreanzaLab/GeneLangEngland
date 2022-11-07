'''
    Using the original linguistic data file with 225 elements, create a file containing combined linguistic features
'''

f = {}
# Feature = list()  # inverse? = feature_INV
f["rhoticity"] = [0, 8, 10, 12, 21, 26, 30, 33, 34, 40, 50, 63, 68, 69, 77, 96, 103, 113, 115, 117, 121, 126, 130, 134, 136, 142,
             151, 158, 172, 180, 188, 189, 194, 199, 205, 210, 212]
f["h_dropping_INV"] = [1, 11, 35, 45, 55, 112, 119, 168, 171, 181, 191, 209, 221]
f["h_insertion"] = [190, 200]
f["heavy_l"] = [2, 5, 7, 14, 22, 24, 27, 51, 54, 57, 61, 70, 72, 90, 92, 94, 95, 98, 101, 102, 104, 107, 109, 120, 122, 128,
           129, 132, 138, 145, 153, 157, 160, 166, 174, 179, 187, 196, 211, 217, 229]
f["l_elision_INV"] = [25, 58, 139, 146, 161, 216]
f["yod_dropping"] = [141]
f["yod_insertion"] = [143, 144, 149, 155, 201]
f["yod_coalescence"] = [186]
f["cluster_l_elision"] = [100]
f["s_voicing"] = [4, 15, 60, 62, 66, 124, 147, 163, 177, 220]
f["cluster_s_elision"] = [111]
f["cluster_d_elision"] = [6, 81, 173, 178, 192, 215]
f["d_fronting"] = [227]
f["cluster_t_elision_INV"] = [36, 198, 222]
f["final_t_glottalisation"] = [65, 127, 148, 154, 204]
f["f_voicing"] = [16, 20, 29, 31, 53, 71, 74, 76, 93, 106, 108, 118, 123, 133, 137, 195, 203]
f["cluster_f_voicing_INV"] = [56]
f["v_elision"] = [140]
f["b_elision"] = [167]
f["gh_f"] = [67]
f["whine_wine_merger_INV"] = [28, ]
f["sh_laxing"] = [150]
f["ch_tsh_not_k"] = [37, 41, 73, 116]
f["medial_tsh_reduction"] = [170]
f["initial_g_dropping_INV"] = [86]
f["g_dropping_INV"] = [52, 152, 159]
f["th_stopping_INV"] = [9, 44, -162]
f["th_fronting"] = [164]
f["dth_fronting"] = [38, 42, ]
f["dth_stopping"] = [39, 43, ]
f["a_raising"] = [13, ]
f["a_fronting"] = [175]
f["foot_strut_split"] = [3, 31, 97]
f["uhw_wuh"] = [176, 185, 223, ]
f["unstressed_vowel_elision"] = [83, 219]
f["uh_lowering"] = [213]
f["uh_raising"] = [214, 228]
f["ai_fronting_INV"] = [75]
f["ai_smoothing"] = [110, 196]
f["aw_smoothing"] = [23, 49]
f["e_raising"] = [156]
f["e_lowering"] = [207]
f["ei_lowering"] = [17]
f["ei_breaking"] = [18, 79, 84, 224]
f["ei_smoothing"] = [19, 46, 80, 85, 99, 225]
f["iuh_reduction"] = [183]
f["ee_breaking"] = [135, 226]
f["oo_uuh"] = [64, 78, 87, 125, 131]
f["o_fronting"] = [182, 202]
f["ouh_smoothing"] = [88, 218]
f["ow_dropping"] = [82]
f["ow_smoothing"] = [59, 91]
f["ow_uuh"] = [114]
f["toe_tow_merger_INV"] = [47, 48]  #uhw_ahw
f["wuh_oo"] = [89, 184]


f["to_remove"] = [105, 165, 169, 193, 206]  # 30="cart"*rhoticity ; 105, 165, 169, 193, 206 = empty;

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


genclust_df = pd.read_csv('SED_genclust2.csv',
                          index_col=[0, 1], header=0,
                          delimiter=',', engine='python')
genclust_old_df = pd.read_csv('SED/genclust/SED_genclust.csv',
                              index_col=None, header=None,
                              delimiter=',', engine='python')

lingnames = genclust_old_df.iloc[0, 13:].to_numpy(dtype=str)
ling_df = genclust_df.iloc[:, 11:]
print(ling_df.iloc[:3, :11])
print(len(list(ling_df.columns.values)))
print(list(ling_df.index))


# There are several elements which were coded incorrectly (inverse coding). This function corrects that.
def f_filter():
    for each in f.keys():
        flist = f[each]
        for feature in flist:
            if feature < 0:
                ling_df.iloc[ling_df.iloc[:, feature] == 0, feature] = 2
                ling_df.iloc[ling_df.iloc[:, feature] == 1, feature] = 0
                ling_df.iloc[ling_df.iloc[:, feature] == 2, feature] = 1
        f[each] = abs(flist)


element_feature_association = []
fkeys = list(f.keys())


for c in range(230):
    try:
        c_loc = [any([n == c for n in f[key]]) for key in fkeys].index(True)
    except ValueError:
        c_loc = None
        continue
    if fkeys[c_loc] != "to_remove":
        element_feature_association.append((
            fkeys[c_loc],
            ling_df.columns.values[c]))
        print("{},{}".format(
            fkeys[c_loc],
            lingnames[c]))


for each in f.keys():
    if each != "to_remove":
        print(each)
        featurespace = ling_df.iloc[:, f[each]].mean(axis=1)
        ling_df[each] = featurespace
        print(featurespace.iloc[:3])

sed_concat = pd.concat([genclust_df.iloc[:, :11], ling_df.iloc[:,230:]], axis=1)

#sed_concat.to_csv(path_or_buf='~/Work/CreanzaLab/SED/cmbnd_features.csv')
