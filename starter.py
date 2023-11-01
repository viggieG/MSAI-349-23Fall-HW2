import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    euclid_dist = round(np.sqrt(sum((a - b)**2)),2)
    return euclid_dist
        
# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # prevent division by zero
    if norm_a == 0 or norm_b == 0:
        return 0 
    return dot_product / (norm_a * norm_b)

def distance(a, b, metric):
    if metric == "euclidean":
        return euclidean(a, b)
    elif metric == "cosim":
        return cosim(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def normalize_data(data):
    if len(data.shape) == 1:  # Check if data is 1D
        norm = np.linalg.norm(data, keepdims=True)
    else:
        norm = np.linalg.norm(data, axis=1, keepdims=True)
    return data/norm

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    k = 1
    #read in training data
    train_list = read_data(train)
    #read in query data
    query_list = read_data(query)
    
    # normalize the data
    for i in range(len(train_list)):
        train_list[i][1] = normalize_data(np.array(train_list[i][1]).astype(float))
    for i in range(len(query_list)):
        query_list[i][1] = normalize_data(np.array(query_list[i][1]).astype(float))
        
    labels = []
    for q_item in query_list:
        _, q_vector = q_item
        distances = []
        
        for t_item in train_list:
            t_label, t_vector = t_item
            if metric == "cosim":
                dist = 1 - distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            else:
                dist = distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            distances.append((dist, t_label))

        # sort based on distances and take top-k labels
        sorted_distances = sorted(distances, key=lambda x: x[0])
        top_k_labels = [item[1] for item in sorted_distances[:k]]
        
        # determine the majority label among top-k labels
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        labels.append(predicted_label)
    
    return labels

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

def knn_accuracy(train, test, metric):
    # Use your knn function to get predictions on the test set
    predictions = knn(train, test, metric)

    # Read in the true labels from the test set
    true_labels = [item[0] for item in read_data(test)]

    # Compare predictions to true labels to compute accuracy
    correct_predictions = sum([1 for predicted, true in zip(predictions, true_labels) if predicted == true])
    accuracy = correct_predictions / len(true_labels)
    print(f'KNN Accuracy {accuracy:.2f} with {metric}')
    
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f'Confusion Matrix ({metric} with {test})')
    plt.savefig(f'confusion_matrix_{metric}_test.png')
    
    return accuracy


def evaluate_knn_with_validation(train_file, valid_file, metric, k_values):
    k_accuracies = {}
    
    for k in k_values:
        predictions = knn_modified(train_file, valid_file, metric,k)
        
        # read the true labels from the validation file
        validation_data = read_data(valid_file)
        true_labels = [row[0] for row in validation_data] 
        
        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        k_accuracies[k] = accuracy

        # print the current k and its accuracy
        print(f"Accuracy for k = {k}: {accuracy:.2f} with {metric}")
        
    best_k = max(k_accuracies, key=k_accuracies.get)
    best_accuracy = k_accuracies[best_k]
    return best_k, best_accuracy

def knn_modified(train,query,metric,k):
    #read in training data
    train_list = read_data(train)
    #read in query data
    query_list = read_data(query)
    
    # Normalize the data
    for i in range(len(train_list)):
        train_list[i][1] = normalize_data(np.array(train_list[i][1]).astype(float))
    for i in range(len(query_list)):
        query_list[i][1] = normalize_data(np.array(query_list[i][1]).astype(float))
        
    labels = []
    for q_item in query_list:
        _, q_vector = q_item
        distances = []
        
        for t_item in train_list:
            t_label, t_vector = t_item
            if metric == "cosim":
                dist = 1 - distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            else:
                dist = distance(np.array(q_vector).astype(float), np.array(t_vector).astype(float), metric)
            distances.append((dist, t_label))

        # Sort based on distances and take top-k labels
        sorted_distances = sorted(distances, key=lambda x: x[0])
        top_k_labels = [item[1] for item in sorted_distances[:k]]
        
        # Determine the majority label among top-k labels
        predicted_label = max(set(top_k_labels), key=top_k_labels.count)
        labels.append(predicted_label)
    
    return labels


def main():
    # show('valid.csv','pixels')
    # knn('train.csv','valid.csv','euclidean')
    knn_accuracy('train.csv','test.csv','euclidean')
    knn_accuracy('train.csv','test.csv','cosim')
    
    # k_values = list(range(1, 51,5))
    # best_k, best_accuracy = evaluate_knn_with_validation('train.csv', 'valid.csv', 'euclidean', k_values)
    # print(f"Best k for Euclidean: {best_k} with accuracy: {best_accuracy:.2f}")

    # best_k, best_accuracy = evaluate_knn_with_validation('train.csv', 'valid.csv', 'cosim', k_values)
    # print(f"Best k for Cosine Similarity: {best_k} with accuracy: {best_accuracy:.2f}")
    
if __name__ == "__main__":
    main()
    

