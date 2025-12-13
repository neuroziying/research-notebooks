from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from statsmodels.tsa.stattools import acf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

folder_path = "E:/data & works/LNP2/figures"
# 加载数据部分保持不变
Neurons = scipy.io.loadmat("C:/Users/Lenovo/Desktop/pycodes/compact_valid_neurons.mat")  
print("\n type:", type(Neurons))

if 'results' in Neurons:
    results = Neurons['results']
    S_field = results['S'][0, 0]
elif 'compact_neurons' in Neurons:
    results = Neurons['compact_neurons']
    S_field = results['S_signals'][0, 0] 
else:
    print("can't find data from mat")
    exit()

spikes = S_field[0,0]
print(f"spikes type: {type(spikes)}")
print(f"spikes shape: {spikes.shape}")

if hasattr(S_field, 'toarray'):
    spikes = S_field.toarray()
else:
    spikes = np.array(S_field)
print(f"arraylike spikes shape: {spikes.shape}")


# 参数设置
total_time = spikes.shape[1]
neuron_num = spikes.shape[0]
sampling_rate = 16
binsize_range = [4,5,6,7]
max_lag = 40


# 分箱和互信息计算
right_binsize = 1.2
bi_frames = int(right_binsize * sampling_rate)
num_bins = total_time // bi_frames

# 计算发放率序列 S
S = np.zeros((neuron_num, num_bins))
for i in range(neuron_num):
    for bin_idx in range(num_bins):
        start = bin_idx * bi_frames
        end = start + bi_frames
        S[i, bin_idx] = np.sum(spikes[i, start:end]) / right_binsize

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap


scaler = StandardScaler()
X_scaled = scaler.fit_transform(S)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull


'''
# 假定你已有：
# S (n_neurons, n_bins) 或 spikes (n_neurons, n_timepoints)
# 我用 X_feat 代表用于聚类的高维特征（例如 X_pca_noPC1 或 X_pca）
# 以及原始标准化矩阵 X_scaled (n_neurons, n_features)

# ---------- 1. 在高维做聚类（示例使用 X_pca，n_components=10）
pca_for_cluster = PCA(n_components=10, random_state=0)
X_pca = pca_for_cluster.fit_transform(X_scaled)  # 或 X_pca 已有时跳过

k = 3  # 或用 silhouette 选的 best_k
km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X_pca)
labels = km.labels_

# ---------- 2. 在 PC1-PC2 平面上投影并画图（用 PCA 2D）
pca2 = PCA(n_components=2, random_state=0)
X_pca2 = pca2.fit_transform(X_scaled)  # 也可以直接 X_pca[:, :2] 如果同一 PCA

# 绘图：每个簇一种颜色
plt.figure(figsize=(6,5))
scatter = plt.scatter(X_pca2[:,0], X_pca2[:,1], c=labels, cmap='tab10', s=60, alpha=0.9)
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Clusters projected on PC1-PC2')
#plt.colorbar(scatter, ticks=range(k), label='cluster id')
plt.grid(alpha=0.2)

# 可选：绘制每个簇的凸包（更直观地看分布）
for c in np.unique(labels):
    idx = np.where(labels==c)[0]
    if len(idx) >= 3:
        hull = ConvexHull(X_pca2[idx])
        for simplex in hull.simplices:
            plt.plot(X_pca2[idx][simplex,0], X_pca2[idx][simplex,1], color='C'+str(int(c)), linewidth=1)

plt.show()

centers_2d = np.array([X_pca2[labels==c].mean(axis=0) for c in range(k)])
dists_to_center = np.linalg.norm(X_pca2 - centers_2d[labels], axis=1)
z = (dists_to_center - dists_to_center.mean()) / (dists_to_center.std() + 1e-9)
outliers = np.where(z > 2)[0]  # z>3 为候选离群点
plt.figure(figsize=(6,5))
plt.scatter(X_pca2[:,0], X_pca2[:,1], c=labels, cmap='tab10', s=60, alpha=0.8)
plt.scatter(X_pca2[outliers,0],X_pca2[outliers,1], facecolors='none', edgecolors='r', s=120, linewidths=2, label='outliers')
plt.legend(); plt.title('PC2-PC3 with outliers highlighted'); plt.show()

for c in np.unique(outliers):
    idx = np.where(outliers==c)[0]
    print(f"Cluster {c}: neuron indices = {idx}")
    mean_trace = S[idx].mean(axis=0)
    plt.figure(figsize=(10,3))
    plt.plot(mean_trace, label=f'cluster {c} (n={len(idx)})')
    plt.legend(); plt.title('outliers firing (binned S)'); plt.xlabel('time bin')
    output_path = os.path.join(folder_path, f'firing_plot{c}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# ---------- 4. 每个簇在原始时间序列上的平均曲线（关键讲图）
for c in np.unique(labels):
    idx = np.where(labels==c)[0]
    print(f"Cluster {c}: neuron indices = {idx}")
    mean_trace = S[idx].mean(axis=0)
    plt.plot(mean_trace, label=f'cluster {c} (n={len(idx)})')
plt.legend(); plt.title('outliers firing (binned S)'); plt.xlabel('time bin')
output_path = os.path.join(folder_path, f'firing_plot{idx}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

for c in np.unique(labels):
    idx = np.where(labels==c)[0]
    print(f"Cluster {c}: neuron indices = {idx}")
    mean_trace = S[idx].mean(axis=0)
    plt.figure(figsize=(10,3))
    plt.plot(mean_trace, label=f'cluster {c} (n={len(idx)})')
    plt.legend(); plt.title('single outliers firing (binned S)'); plt.xlabel('time bin')
    output_path = os.path.join(folder_path, f'firing_plot{c}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

trace = S[5]
plt.figure(figsize=(10,3))
plt.plot(trace, label=f'cluster {5} ')
plt.legend(); plt.title('ok firing (binned S)'); plt.xlabel('time bin')
output_path = os.path.join(folder_path, f'firing_plot{5}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
# # ---------- 5. 若想比较 UMAP 投影（仅可视化对照）
# import umap
# reducer = umap.UMAP(random_state=0)
# X_umap = reducer.fit_transform(X_pca)   # 注意：fit on same features used for clustering
# plt.figure(figsize=(6,5))
# plt.scatter(X_umap[:,0], X_umap[:,1], c=labels, cmap='tab10', s=60, alpha=0.9)
# plt.title('UMAP (same clusters colored)'); plt.show()

# ---------- 2.5 在图上画出PC2、PC3
X_noPC1 = X_pca[:, 1:]   # shape = (n_neuron, 9)
k23 = 3
km23 = KMeans(n_clusters=k23, n_init=50, random_state=0).fit(X_noPC1)
labels23 = km23.labels_

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_noPC1[:,0],X_noPC1[:,1], X_noPC1[:,2], c=labels, s=50)
ax.set_xlabel('PC2'); ax.set_ylabel('PC3'); ax.set_zlabel('PC4')
ax.set_title('PCA234 3D')
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(X_noPC1[:,0], X_noPC1[:,1], c=labels, cmap='tab10', s=60, alpha=0.8)
plt.legend(); plt.title('PC2-PC3'); plt.show()


# ---------- 3. 在图上标出疑似离群点（距离簇中心远的点）
centers_2d = np.array([X_noPC1[labels23==c].mean(axis=0) for c in range(k23)])
dists_to_center = np.linalg.norm(X_noPC1 - centers_2d[labels], axis=1)
# 用 zscore 判定极端离群点
z = (dists_to_center - dists_to_center.mean()) / (dists_to_center.std() + 1e-9)
outliers = np.where(z > 1)[0]  # z>3 为候选离群点
plt.figure(figsize=(6,5))
plt.scatter(X_noPC1[:,0], X_noPC1[:,1], c=labels23, cmap='tab10', s=60, alpha=0.8)
plt.scatter(X_noPC1[outliers,0], X_noPC1[outliers,1], facecolors='none', edgecolors='r', s=120, linewidths=2, label='outliers')
plt.legend(); plt.title('PC2-PC3 with outliers highlighted'); plt.show()

# ---------- 4. 每个簇在原始时间序列上的平均曲线（关键讲图）


for c in np.unique(labels23):
    idx = np.where(labels23==c)[0]
    print(f"Cluster {c}: neuron indices = {idx}")
    mean_trace = S[idx].mean(axis=0)
    plt.figure(figsize=(10,3))
    plt.plot(mean_trace, label=f'cluster {c} (n={len(idx)})')
    plt.legend(); plt.title('23 single outliers firing (binned S)'); plt.xlabel('time bin')
    output_path = os.path.join(folder_path, f'23firing_plot{c}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

trace = S[5]
plt.figure(figsize=(10,3))
plt.plot(trace, labels23=f'cluster {5} ')
plt.legend(); plt.title('23 ok firing (binned S)'); plt.xlabel('time bin')
output_path = os.path.join(folder_path, f'23firing_plot{5}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# ---------- 6. 局部保持度（可选；给组会一个数字说明保真度）
def local_preservation(orig_X, proj_X, k=10):
    nbr_orig = NearestNeighbors(n_neighbors=k+1).fit(orig_X)
    nbr_proj = NearestNeighbors(n_neighbors=k+1).fit(proj_X)
    idx_orig = nbr_orig.kneighbors(return_distance=False)
    idx_proj = nbr_proj.kneighbors(return_distance=False)
    orig_sets = [set(row[1:]) for row in idx_orig]
    proj_sets = [set(row[1:]) for row in idx_proj]
    overlaps = [len(a & b)/k for a,b in zip(orig_sets, proj_sets)]
    return np.mean(overlaps)

print("local_preservation PCA10->PCA2:", local_preservation(X_pca, X_pca2))
print("local_preservation PCA10->UMAP:", local_preservation(X_pca, X_umap))
'''

#=============================================
# 1) PCA 10D
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_pca)
plt.scatter(X_umap[:,0], X_umap[:,1])
plt.title("umap PC10")
plt.show()

# 2) 去掉 PC1
X_noPC1 = X_pca[:, 1:]   # shape = (n_neuron, 9)
reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_noPC1)
plt.scatter(X_umap[:,0], X_umap[:,1])
plt.title("no PC1")
plt.show()

# 3) 只保留PC1和PC2
X_pca2 = PCA(n_components=2).fit_transform(X_scaled)
reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_pca2)
plt.scatter(X_umap[:,0], X_umap[:,1])
plt.title("umap PC1 and PC2")
plt.show()

plt.scatter(X_pca2[:,0], X_pca2[:,1])
plt.title("PC1 and PC2")
plt.show()

# 4) 只保留PC1、PC2和PC3
X_pca3 = PCA(n_components=3).fit_transform(X_scaled)
reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_pca3)
plt.scatter(X_umap[:,0], X_umap[:,1])
plt.title("umap PC123")
plt.show()

# 3D PCA
km = KMeans(n_clusters=3, n_init=20).fit(X_pca3)
labels = km.labels_
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=km.labels_, s=50)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.set_title('PCA123 3D')
plt.show()

# cleaned PCA
dist = np.linalg.norm(X_pca[:, :3], axis=1)
plt.hist(dist)
plt.show()
threshold = 30
mask = dist < threshold   # 去掉 outlier
X_clean = X_pca[mask]

#=====================================
# 对X_pca做聚类分析
reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_pca)
km = KMeans(n_clusters=3, n_init=20).fit(X_umap)
labels = km.labels_
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_umap[:,0], X_umap[:,1], X_umap[:,2], c=km.labels_, s=50)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
ax.set_title('UMAP X_pca 3D')
plt.show()

#=====================================
# 对cleaned_X做聚类分析
reducer = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap = reducer.fit_transform(X_clean)
km = KMeans(n_clusters=3, n_init=20).fit(X_umap)
labels = km.labels_
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_umap[:,0], X_umap[:,1], X_umap[:,2], c=km.labels_, s=50)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
ax.set_title('UMAP cleaned 3D')
plt.show()

#=====================================
# 对X_pca3做聚类分析
reducer3 = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap3 = reducer3.fit_transform(X_pca3)
km = KMeans(n_clusters=3, n_init=20).fit(X_umap3)
labels = km.labels_
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_umap3[:,0], X_umap3[:,1], X_umap3[:,2], c=km.labels_, s=50)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
ax.set_title('UMAP 3D')
plt.show()

#=====================================
# 画出各个成分的图像

print("PCA explained variance ratio:", pca.explained_variance_ratio_)

for i in range(5):
    plt.plot(pca.components_[i])
    plt.title(f"PC{i+1}")
    plt.show()

#=====================================
# 对x_noPC1做聚类分析

# UMAP-2D
reducer = umap.UMAP(random_state=0)
X_umap = reducer.fit_transform(X_noPC1)

# K-means
km = KMeans(n_clusters=3, n_init=20).fit(X_noPC1)
labels = km.labels_

plt.scatter(X_umap[:,0], X_umap[:,1], c=labels)
plt.title("k-means no pc1")
plt.show()
#=====================================
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

X_feat = X_noPC1  # 或 X_pca

# 3D PCA
pca3 = PCA(n_components=3, random_state=0)
X_pca3 = pca3.fit_transform(X_feat)
km = KMeans(n_clusters=3, n_init=20).fit(X_pca3)
labels = km.labels_
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=km.labels_, s=50)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
ax.set_title('PCA 3D')
plt.show()

#=====================================
import umap
reducer3 = umap.UMAP(n_components=3, random_state=0, n_neighbors=10, min_dist=0.1)
X_umap3 = reducer3.fit_transform(X_feat)
km = KMeans(n_clusters=3, n_init=20).fit(X_umap3)
labels = km.labels_
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_umap3[:,0], X_umap3[:,1], X_umap3[:,2], c=km.labels_, s=50)
ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
ax.set_title('UMAP 3D')
plt.show()

#=====================================
scores = []
ks = range(2, 15)

for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    scores.append(score)

plt.plot(ks, scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("K-means 聚类质量")
plt.show()

best_k = ks[np.argmax(scores)]
print("最佳 k =", best_k)

#=====================================
km = KMeans(n_clusters=best_k, n_init=20, random_state=0)
labels_km = km.fit_predict(X_pca)

reducer = umap.UMAP(random_state=0)
X_umap = reducer.fit_transform(X_pca)

plt.scatter(X_umap[:,0], X_umap[:,1], c=labels_km, s=50)
plt.title("UMAP + K-means 聚类结果")
plt.show()

#======================================
db = DBSCAN(eps=0.5, min_samples=5).fit(X_pca)
labels_db = db.labels_

plt.scatter(X_umap[:,0], X_umap[:,1], c=labels_db, s=50, cmap='tab10')
plt.title("UMAP + DBSCAN（检测离群神经元）")
plt.show()

for c in np.unique(labels_km):
    idx = labels_km == c
    mean_fr = spikes[idx].mean(axis=0)
    plt.plot(mean_fr, label=f"Cluster {c}")

plt.legend()
plt.title("不同 cluster 的平均 firing rate")
plt.show()
