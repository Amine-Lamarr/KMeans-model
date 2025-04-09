from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# creating some data training 
x , y = make_blobs(n_samples=500 ,  centers=5 , cluster_std=1.0 , random_state=97)

# testing the best clusters number 
inertia = []
for k in range(1 , 11):
    model = KMeans(n_clusters=k , init='k-means++' , random_state=32)
    model.fit(x)
    inertia.append(model.inertia_)


# drawing the inertia function
plt.style.use("fivethirtyeight")
plt.plot(range(1 , 11) , inertia , marker="o" , label = 'inertia function')
plt.xlabel("number of clusters")
plt.ylabel("inertia")
plt.title("elbow method")
plt.legend()
plt.show()

# i found that 5 is the best value

# creating model 
model = KMeans(n_clusters=5 , init="k-means++"  , random_state=97)
# fititng 
model.fit(x)

# which cluster each point belogs 
label = model.labels_
# clusters coordinate 
cluster_coo = model.cluster_centers_
#print(cluster_coo)

# drawong our data
plt.style.use("fivethirtyeight")               
plt.scatter(x[: , 0] , x[:, 1] , c="purple" , s=90 , label='data' , alpha=0.5)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("OUR REAL DATA")
plt.show()

# data after training 
plt.style.use("fivethirtyeight")           
plt.scatter(x[: , 0] , x[:, 1] , c=label , s=100 , label='data' , cmap="cool" , alpha=0.5)
plt.scatter(cluster_coo[: , 0] , cluster_coo[:, 1]  ,marker="x"  , c='black', s=500 , label = 'clusters')
plt.colorbar()                                                               
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("data after KMeans model")
plt.legend()
plt.show()
print(plt.style.available)
