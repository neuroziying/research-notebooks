import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from sklearn.feature_selection import mutual_info_classif   # 或 mutual_info_score
from sklearn.metrics import mutual_info_score
from tqdm import trange
import scipy.stats as stats

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from statsmodels.tsa.stattools import acf

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

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


# 创建计算互信息的函数
def calculate_mi_matrix(data, num_bins_pdf=10):
    neuron_num = data.shape[0]
    MI_matrix = np.zeros((neuron_num, neuron_num))
    
    # 预先计算所有神经元的边缘分布
    neuron_pdfs = []
    for i in range(neuron_num):
        prob, _ = np.histogram(data[i, :], bins=num_bins_pdf, density=True)
        neuron_pdfs.append(prob)
    
    # 计算互信息矩阵
    for i in range(neuron_num):
        for j in range(i, neuron_num):
            p_joint, _, _ = np.histogram2d(data[i, :], data[j, :], 
                                         bins=[num_bins_pdf, num_bins_pdf], 
                                         density=True)
            
            p_i = neuron_pdfs[i]
            p_j = neuron_pdfs[j]
            
            mi_value = 0.0
            for idx_i in range(num_bins_pdf):
                for idx_j in range(num_bins_pdf):
                    p_ij = p_joint[idx_i, idx_j]
                    p_i_val = p_i[idx_i] if idx_i < len(p_i) else 1e-12
                    p_j_val = p_j[idx_j] if idx_j < len(p_j) else 1e-12
                    
                    if p_ij > 1e-12 and p_i_val > 1e-12 and p_j_val > 1e-12:
                        mi_value += p_ij * np.log2(p_ij / (p_i_val * p_j_val))
            
            MI_matrix[i, j] = mi_value
            MI_matrix[j, i] = mi_value
    
    return MI_matrix

# 计算原始数据的互信息
MI = calculate_mi_matrix(S)

# ---------------------------
# 假定输入（你已有）
# spikes: np.ndarray, shape (n_neurons, n_timepoints)
# optional:
# beh_labels: np.ndarray (n_timepoints,) 离散行为标签 (如 A-SOiD 输出)，
# 或 continuous behavior: pos_x,pos_y 或 speed (n_timepoints,)
# ---------------------------

# 使用 S（binned firing）或原始 spikes：推荐先用 S（慢变量更可靠）
# 如果你已经用 S（shape (n_neurons, num_bins)），把下面的 `X_raw = spikes` 替换为 S

#X_raw = spikes.copy()   # (n_neurons, n_timepoints) 
X_raw = S.copy() 




# 1) 标准化（按 neuron 标准化） #(n_neurons, n_bins)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)   # shape (n_neurons, n_features)

# 2) PCA（用于查看并拿到 PC1）
pca = PCA(n_components=10, random_state=0)
X_pca = pca.fit_transform(X_scaled)      # shape (n_neurons, 10)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)
# 画累计方差（可选）
cum = np.cumsum(pca.explained_variance_ratio_)
plt.bar(np.arange(1, len(cum)+1), cum)
plt.xlabel("n components"); plt.ylabel("cumulative explained variance"); plt.show()

# ---------------------------
# 方法 A：直接丢弃 PC1（使用 PCA 的分量 2..end）
# ---------------------------
X_pca_noPC1 = X_pca[:, 1:]   # 保留 PC2..PC10 (shape (n_neurons, 9))

# ---------------------------
# 方法 B：把 PC1 从原始数据中重建并减去（保留其余在原始空间的残差）
# 这在你想在原始 feature 空间做 UMAP 时很有用
# ---------------------------
# 重建 PC1 在原始空间上的贡献并减掉
pc1_scores = pca.transform(X_scaled)[:, 0]    # (n_neurons,)
pc1_vector = pca.components_[0]               # (n_features,)
# reconstructed component for each neuron = score * pc1_vector
recon_pc1 = np.outer(pc1_scores, pc1_vector)  # (n_neurons, n_features)
X_noPC1_space = X_scaled - recon_pc1          # residuals after removing PC1

# 你可以在此对 X_noPC1_space 再做 PCA 或直接送 UMAP
pca2 = PCA(n_components=10, random_state=0)
X_pca_after_rm1 = pca2.fit_transform(X_noPC1_space)

# ---------------------------
# 3) 计算 mutual information (MI) 特征（可选）
#    这里给两种场景：
#      a) 离散行为标签 beh_labels (推荐)：用 mutual_info_score
#      b) 连续行为（speed/position）：先把行为离散化，然后用 mutual_info_score
# ---------------------------
def compute_mi_per_neuron(spike_mat, beh_labels, n_perm=200):
    # spike_mat: (n_neurons, n_timepoints) continuous or binned counts
    # beh_labels: (n_timepoints,) discrete labels (int)
    n_neurons = spike_mat.shape[0]
    mi_vals = np.zeros(n_neurons)
    pvals = np.ones(n_neurons)
    for i in range(n_neurons):
        x = spike_mat[i, :]
        # discretize x into e.g. 8 bins
        x_disc = np.digitize(x, np.percentile(x, np.linspace(0,100,9))) 
        mi = mutual_info_score(x_disc, beh_labels)
        mi_vals[i] = mi
        # permutation test for p-value
        nulls = np.zeros(n_perm)
        for p in range(n_perm):
            y_shuf = np.random.permutation(beh_labels)
            nulls[p] = mutual_info_score(x_disc, y_shuf)
        pvals[i] = (np.sum(nulls >= mi) + 1) / (n_perm + 1)
    return mi_vals, pvals

# 示例：如果你有 beh_labels:
# beh_labels = ...  # shape (n_timepoints,) must align with X_raw axis=1
# mi_vals, mi_p = compute_mi_per_neuron(X_raw, beh_labels, n_perm=300)
# 将 MI 标准化后作为 feature
# mi_feat = (mi_vals - mi_vals.mean()) / (mi_vals.std() + 1e-9)
# X_feature_for_clustering = np.hstack([X_pca_noPC1, mi_feat.reshape(-1,1)])

# 如果没有行为标签，可以用 speed 等连续变量：
# speed_disc = np.digitize(speed, np.percentile(speed, np.linspace(0,100,6)))
# mi_vals, _ = compute_mi_per_neuron(X_raw, speed_disc)
# 然后拼接

# ---------------------------
# 4) 最终的 feature 矩阵（示例：用方法 A + MI）
# ---------------------------
# 若无 MI:
X_feat = X_pca_noPC1   # shape (n_neurons, n_components-1)
# 若有 MI:
# X_feat = np.hstack([X_pca_noPC1, mi_feat.reshape(-1,1)])

# 标准化最终特征（非常重要）
X_feat = StandardScaler().fit_transform(X_feat)

# ---------------------------
# 5) UMAP 可视化（用 X_feat）
# ---------------------------
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=0)
X_umap = reducer.fit_transform(X_feat)

plt.figure(figsize=(6,5))
plt.scatter(X_umap[:,0], X_umap[:,1], s=50)
plt.title("UMAP on features (PC1 removed)")
plt.show()

# ---------------------------
# 6) K-means 聚类（选 k 用 silhouette）
# ---------------------------
ks = range(2,9)
scores = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X_feat)
    s = silhouette_score(X_feat, km.labels_)
    scores.append(s)
plt.plot(list(ks), scores, marker='o'); plt.xlabel('k'); plt.ylabel('silhouette'); plt.grid(True); plt.show()
best_k = ks[int(np.argmax(scores))]
print("best_k by silhouette:", best_k)

km = KMeans(n_clusters=best_k, n_init=100, random_state=0).fit(X_feat)
labels_km = km.labels_

# UMAP + labels
plt.figure(figsize=(6,5))
plt.scatter(X_umap[:,0], X_umap[:,1], c=labels_km, cmap='tab10', s=50)
plt.title(f'UMAP + KMeans (k={best_k}) on PC1-removed features')
plt.show()

# ---------------------------
# 7) 每个 cluster 的平均 firing（回到原始时序）
# ---------------------------
for c in np.unique(labels_km):
    idx = np.where(labels_km == c)[0]
    mean_trace = X_raw[idx].mean(axis=0)
    plt.plot(mean_trace, label=f'cluster {c} (n={len(idx)})')
plt.legend(); plt.title('cluster average trace (original timescale)'); plt.show()

# ---------------------------
# 8) 若做了 MI，还可以验证：top MI neurons 是否集中在某些 cluster
# ---------------------------
# if you computed mi_vals above:
# median_mi = np.median(mi_vals)
# plt.bar(np.unique(labels_km), [np.median(mi_vals[labels_km==c]) for c in np.unique(labels_km)])
# plt.title('median MI per cluster'); plt.show()
