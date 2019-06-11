import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
import tables
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, ward
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
#from sklearn.manifold import TSNE
from scipy import stats
import seaborn as sns; sns.set()
from joblib import Parallel, delayed
import itertools
#import samles to get the info of layers
samples = pd.read_csv('mouse_VISp_2018-06-14_samples-columns.csv')

# Filtered Exon Matrix
#1) get the median value for each gene
#2) get the mean value of all gene medians.
#3) select the gene(row) that medians>mean
exon_filtered = pd.read_hdf('exon_filtered.h5', key='exon_filtered')
exon_filtered_log2 = np.log2(exon_filtered)
del exon_filtered
exon_filtered_log2[exon_filtered_log2==exon_filtered_log2.values[0, 0]]=0

#Get the each layer's index
L1_ind = np.where(samples.values[:, 13] =='L1')[0]
L23_ind = np.where(samples.values[:, 13] =='L2/3')[0]
L4_ind = np.where(samples.values[:, 13] =='L4')[0]
L5_ind = np.where(samples.values[:, 13] =='L5')[0]
L6_ind = np.where(samples.values[:, 13] =='L6')[0]

#Get the filtered and layer-sorted geneset
L1_log2 = exon_filtered_log2.iloc[:, L1_ind]
L23_log2 = exon_filtered_log2.iloc[:, L23_ind]
L4_log2 = exon_filtered_log2.iloc[:, L4_ind]
L5_log2 = exon_filtered_log2.iloc[:, L5_ind]
L6_log2 = exon_filtered_log2.iloc[:, L6_ind]

print('before clustering\n')

threshold = 200
clustering_L1 = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold= threshold).fit(L1_log2)
clustering_L23 = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold = threshold).fit(L23_log2)
clustering_L4 = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold = threshold).fit(L4_log2)
clustering_L5 = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold = threshold).fit(L5_log2)
clustering_L6 = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold = threshold).fit(L6_log2)

L1_labels = clustering_L1.labels_
L23_labels = clustering_L23.labels_
L4_labels = clustering_L4.labels_
L5_labels = clustering_L5.labels_
L6_labels = clustering_L6.labels_

pre_L1_labels = []
pre_L23_labels = []
pre_L4_labels = []
pre_L5_labels = []
pre_L6_labels = []

for i in range(0, 5295):
    pre_L1_labels.append((L1_labels[i], i))
    pre_L23_labels.append((L23_labels[i], i))
    pre_L4_labels.append((L4_labels[i], i))
    pre_L5_labels.append((L5_labels[i], i))
    pre_L6_labels.append((L6_labels[i], i))

pre_L1_labels = np.reshape(pre_L1_labels, (-1, 2))
pre_L23_labels = np.reshape(pre_L23_labels, (-1, 2))
pre_L4_labels = np.reshape(pre_L4_labels, (-1, 2))
pre_L5_labels = np.reshape(pre_L5_labels, (-1, 2))
pre_L6_labels = np.reshape(pre_L6_labels, (-1, 2))

total_pre = [pre_L1_labels, pre_L23_labels, pre_L4_labels, pre_L5_labels, pre_L6_labels]

# So we gotta make a list that ((layer_num), clustered_num, clustered_points)
gene_group = []
layer_num = []
for j in range(0, len(total_pre)):
    max = np.max(total_pre[j][:, 0]).astype(int)
    layer_num.append([j, max])
    for i in range(0, max):
        gene_group.append((i, np.where(total_pre[j][:, 0] == i)))
gene_group = np.reshape(gene_group, (-1, 2))
layer_num = np.reshape(layer_num, (-1, 2))

# Let's make a dictionary
# summmary[layer, cluster] = the gene index
summary = {}
gene_group_ind = 0
for i in layer_num[:, 0]:
    for j in range(0, layer_num[i, 1]):
        summary[i, j] = gene_group[gene_group_ind, 1][0]
        gene_group_ind += 1

L1, L23, L4, L5, L6 = {}, {}, {}, {}, {}

for i in range(0, layer_num[0, 1]):
    L1[i] = pd.DataFrame(np.mean(L1_log2.values[summary[0, i], :], axis=0))
for i in range(0, layer_num[1, 1]):
    L23[i] = pd.DataFrame(np.mean(L23_log2.values[summary[1, i], :], axis=0))
for i in range(0, layer_num[2, 1]):
    L4[i] = pd.DataFrame(np.mean(L4_log2.values[summary[2, i], :], axis=0))
for i in range(0, layer_num[3, 1]):
    L5[i] = pd.DataFrame(np.mean(L5_log2.values[summary[3, i], :], axis=0))
for i in range(0, layer_num[4, 1]):
    L6[i] = pd.DataFrame(np.mean(L6_log2.values[summary[4, i], :], axis=0))

L1_chi = pd.concat(L1, axis=1)
L23_chi = pd.concat(L23, axis=1)
L4_chi = pd.concat(L4, axis=1)
L5_chi = pd.concat(L5, axis=1)
L6_chi = pd.concat(L6, axis=1)

_, p_L1 = stats.chisquare(L1_chi, ddof= layer_num[0, 1]-1)
_, p_L23 = stats.chisquare(L23_chi, ddof= layer_num[1, 1]-1)
_, p_L4 = stats.chisquare(L4_chi, ddof= layer_num[2, 1]-1)
_, p_L5 = stats.chisquare(L5_chi, ddof= layer_num[3, 1]-1)
_, p_L6 = stats.chisquare(L6_chi, ddof= layer_num[4, 1]-1)

#row = samples, col = genes sets.
signi_L1 = pd.DataFrame(L1_chi.values[:, p_L1<0.001])
signi_L23 = pd.DataFrame(L23_chi.values[:, p_L23<0.001])
signi_L4 = pd.DataFrame(L4_chi.values[:, p_L4<0.001])
signi_L5 = pd.DataFrame(L5_chi.values[:, p_L5<0.001])
signi_L6 = pd.DataFrame(L6_chi.values[:, p_L6<0.001])

# total_exon = [L1_log2, L23_log2, L4_log2, L5_log2, L6_log2]
# signi_total = [signi_L1, signi_L23, signi_L4, signi_L5, signi_L6]
# # When I wanna what is the gene_set in original exon_sheet...
# p_val_list = [p_L1<0.001, p_L23<0.001, p_L4<0.001, p_L5<0.001, p_L6<0.001]
# for i in range(0, len(p_val_list)):
#     for j in p_val_list[i]:
#         signi_total[i].append(total_exon[i].values[summary[i, j], :])
# #for i in range(0, len(p_val_list)):
# #  signi_total[i] = np.reshape(signi_total[i], (-1, np.shape(signi_total[i])[0]))
#
# for i in range(0, len(p_val_list)):
#     signi_total[i] = np.reshape(signi_total[i], (-1, np.shape(signi_total[i])[0]))

#Let's get which clustered things would be significantly represented the each layer
min_num = np.min([signi_L1.shape[0], signi_L23.shape[0], signi_L4.shape[0], signi_L5.shape[0], signi_L6.shape[0]])
L23_sample_num = np.random.choice(signi_L23.shape[0], size=min_num, replace=False)
L4_sample_num = np.random.choice(signi_L4.shape[0], size=min_num, replace=False)
L5_sample_num = np.random.choice(signi_L5.shape[0], size=min_num, replace=False)
L6_sample_num = np.random.choice(signi_L6.shape[0], size=min_num, replace=False)

p_value_overall = []
for i in itertools.product(*[np.arange(signi_L1.shape[1]), np.arange(signi_L23.shape[1]), np.arange(signi_L4.shape[1]), np.arange(signi_L5.shape[1]), np.arange(signi_L6.shape[1])]):
  candi = pd.DataFrame(np.vstack((signi_L1.values[:, i[0]], signi_L23.values[L23_sample_num, i[1]], signi_L4.values[L4_sample_num, i[2]], signi_L5.values[L5_sample_num, i[3]], signi_L6.values[L6_sample_num, i[4]]))).T
  _, p_candi = stats.chisquare(candi, ddof=4)
  if np.count_nonzero(p_candi<0.001) == 5:
    p_value_overall.append((i, p_candi))
    print('p_value_num : {}, total: {}'.format(len(p_value_overall), i))
p_value_overall = np.reshape(p_value_overall, (-1, 2))
np.savetxt('p_value_overal.txt', p_value_overall)



# Parallel(n_jobs=-1)(delayed(i)(.2) for _ in range(10))
# for i in itertools.product(*[np.arange(signi_L1.shape[1]), np.arange(signi_L23.shape[1]), np.arange(signi_L4.shape[1]), np.arange(signi_L5.shape[1]), np.arange(signi_L6.shape[1])]):
#
#
#
#   candi = pd.DataFrame(np.vstack((signi_L1.values[:, i[0]], signi_L23.values[L23_sample_num, i[1]], signi_L4.values[L4_sample_num, i[2]], signi_L5.values[L5_sample_num, i[3]], signi_L6.values[L6_sample_num, i[4]]))).T
#   _, p_candi = stats.chisquare(candi, ddof=4)
#   if np.count_nonzero(p_candi<0.001) == 5:
#     p_value_overall.append((i, p_candi))
p_value_overall = np.reshape(p_value_overall, (-1, 2))
np.savetxt('p_value_overal.txt', p_value_overall)


# #Let's get which clustered things would be significantly represented the each layer
# min_num = np.min([signi_L1.shape[0], signi_L23.shape[0], signi_L4.shape[0], signi_L5.shape[0], signi_L6.shape[0]])
# L23_sample_num = np.random.choice(signi_L23.shape[0], size=min_num, replace=False)
# L4_sample_num = np.random.choice(signi_L4.shape[0], size=min_num, replace=False)
# L5_sample_num = np.random.choice(signi_L5.shape[0], size=min_num, replace=False)
# L6_sample_num = np.random.choice(signi_L6.shape[0], size=min_num, replace=False)
#
# p_value_overall = []
# for i in range(0, signi_L1.shape[1]):
#     print(i)
#     for j in range(0, signi_L23.shape[1]):
#         for k in range(0, signi_L4.shape[1]):
#             for u in range(0, signi_L5.shape[1]):
#                 for l in range(0, signi_L6.shape[1]):
#                     candi = pd.DataFrame(np.vstack((signi_L1.values[:, i], signi_L23.values[L23_sample_num, j], signi_L4.values[L4_sample_num, k], signi_L5.values[L5_sample_num, u], signi_L6.values[L6_sample_num, l]))).T
#                     _, p_candi = stats.chisquare(candi, ddof=4)
#                     if np.count_nonzero(p_candi<0.001) == 5:
#                         p_value_overall.append((i, j, k, u, l, p_candi))
# p_value_overall = np.reshpae(p_value_overall, (-1, 6))
# np.savetxt('p_value_overall.txt', p_value_overall)

#Let's draw heatmap
#heatmap_L1 = clustermap(L1_log2, method='ward', col_cluster= False, figsize=(12, 12))
#heatmap_L23 = clustermap(L23_log2, method = 'ward', col_cluster= False, figsize(12, 12))
#heatmap_L4= clustermap(L4_log2, method = 'ward', col_cluster= False, figsize(12, 12))
#heatmap_L5 = clustermap(L5_log2, method = 'ward', col_cluster= False, figsize(12, 12))
#heatmap_L6 = clustermap(L6_log2, method = 'ward', col_cluster= False, figsize(12, 12))

#Get the clustered row and col indices
#heatmap_L1_row = heatmap_L1.dendrogram_row.reordered_ind
#heatmap_L23_row = heatmap_L23.dendrogram_row.reordered_ind
#heatmap_L4_row = heatmap_L4.dendrogram_row.reordered_ind
#heatmap_L5_row = heatmap_L5.dendrogram_row.reordered_ind
#heatmap_L6_row = heatmap_L6.dendrogram_row.reordered_ind

# heatmap_L1_col = heatmap_L1.dendrogram_col.reordered_ind
# heatmap_L23_col = heatmap_L23.dendrogram_col.reordered_ind
# heatmap_L4_col = heatmap_L4.dendrogram_col.reordered_ind
# heatmap_L5_col = heatmap_L5.dendrogram_col.reordered_ind
# heatmap_L6_col = heatmap_L6.dendrogram_col.reordered_ind

# #PCA
# pca_L1 = PCA().fit(L1_log2)
# pca_L23 = PCA().fit(L23_log2)
# pca_L4 = PCA().fit(L4_log2)
# pca_L5 = PCA().fit(L5_log2)
# pca_L6 = PCA().fit(L6_log2)

# #TSNE
# L1_tsne = pd.read_hdf('L1_tsne.h5', key='L1_tsne')
# L23_tsne = pd.read_hdf('L23_tsne.h5', key='L23_tsne')
# L4_tsne = pd.read_hdf('L4_tsne.h5', key='L4_tsne')
# L5_tsne = pd.read_hdf('L5_tsne.h5', key='L5_tsne')
# L6_tsne = pd.read_hdf('L6_tsne.h5', key='L6_tsne')

# #Ward clustering with scipy
# Z_L1_log2 = ward(L1_log2)
# Z_L23_log2 = ward(L23_log2)
# Z_L4_log2 = ward(L4_log2)
# Z_L5_log2 = ward(L5_log2)
# Z_L6_log2 = ward(L6_log2)

