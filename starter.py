import numpy as np
from sklearn.decomposition import PCA
import math

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    euclid_dist = round(np.sqrt(sum((a - b)^2)),2)
    return euclid_dist
        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return round(dot_product / (norm_a * norm_b),2)

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
        k = 45
    #read in training data
    train_list = read_data(train)
    #read in query data
    query_list = read_data(query)

    # extracting the training labels
    labels = [observation[0] for observation in train_list]
    # extracting the training and query feature vectors
    train_features = np.array([observation[1:] for observation in train_list])
    query_features = np.array([observation[1:] for observation in query_list])

    # calculate covariance matrix
    cov_matrix = np.cov(train_features.T)
    # calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # select components with eigenvalues greater than 1 based on Kaiser's Rule
    n_comp = np.sum(eigenvalues > 1)
    # initialize PCA with the selected number of component
    pca = PCA(n_components = n_comp)
    # fit PCA to the training dataset and transform the training dataset/query dataset
    pca.fit(train_features)
    features_pca = pca.transform(train_features)
    query_pca = pca.transform(query_features)
    if metric == 'euclidean':
        k_distances = []
        for query_pca_obs in query_pca:
            distances = [euclidean(query_pca_obs, train_pca_obs) for train_pca_obs in features_pca].sort()
            sorted_dist = distances[:k]
            k_distances.append(distances)
        return k_distances

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    
