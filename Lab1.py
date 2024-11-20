import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
from sklearn.metrics import silhouette_score

#1.1
def find_centers(data_):
    bandwidth = estimate_bandwidth(data_, quantile=0.15, n_samples=len(data_))
    mean_shift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift_model.fit(data_)
    return mean_shift_model.cluster_centers_
#1.2
def optimal_clusters(data_):
    scores_ = []
    range_clusters_ = range(2, 16)
    for clusters_num in range_clusters_:
        kmeans_ = KMeans(n_clusters=clusters_num, init='k-means++', n_init=10)
        kmeans_.fit(data_)
        labels = kmeans_.labels_
        score = silhouette_score(data_, labels)
        scores_.append(score)
    return scores_, range_clusters_
data = np.loadtxt("Assets/lab01.csv", delimiter=";")
scores, range_clusters = optimal_clusters(data)
optimal_number = range_clusters[np.argmax(scores)]

#2.1
plt.figure()
plt.scatter(data[:, 0], data[:, 1], marker='o', s=80, edgecolors='black', facecolors='none')
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
plt.title("Вихідні точки на площині")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

#2.2
centers = find_centers(data)
plt.figure()
plt.title("Центри кластерів (метод зсуву середнього)")
plt.scatter(centers[:, 0], centers[:, 1], marker='o', s=210,   color='black', facecolors='black', linewidths=4,
            zorder=12)
#2.3
plt.figure()
plt.title("Бар діаграмма score(number of clusters)")
plt.bar(range_clusters, scores)

#1.3
kmeans = KMeans(n_clusters=optimal_number, init='k-means++', n_init=10)
kmeans.fit(data)

#2.4
step_size = 0.01
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.colormaps["Paired"], aspect='auto', origin='lower')

plt.scatter(data[:, 0], data[:, 1], marker='o', s=80, edgecolors='black', facecolors='none')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='o', s=210,   color='black', facecolors='black', linewidths=4,
            zorder=12)
plt.title("Кластеризовані дані з областями кластеризації")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()