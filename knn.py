import numpy as np 

def knn(X,data,labels,k):
	distance = [pow(pow(np.array(X)-data[i],2).sum(),0.5) for i in range(data.shape[0])]
	k_result = np.argsort(distance)[-k:]
	# k_frequency,_ = np.histogram(labels[k_result],len(set(k_result.tolist())))
	result_dict = {label:k_result.tolist().count(label) for label in set(labels[k_result])}
	classify_result = sorted(result_dict.items(),key=lambda tup:tup[1],reverse=True)
	return classify_result[0][0]

if __name__ == '__main__':
	data = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = np.array([0,0,1,1])
	print(knn([0,1],data,labels,3))
	