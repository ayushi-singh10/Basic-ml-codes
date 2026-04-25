from sklearn.cluster import KMeans

X = [[20], [25], [30], [70], [75], [80]]

model = KMeans(n_clusters=2, random_state=0)

model.fit(X)

print(model.labels_)
