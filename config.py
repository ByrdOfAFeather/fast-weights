from sklearn.preprocessing import LabelEncoder


def get_encoder():
	return LabelEncoder().fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])