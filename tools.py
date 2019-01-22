import numpy as np

def normalize(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1,keepdims=True)
    std[std ==0] = 1
    x = (x-mean)/std
    return x

def normalize_2(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def EuclideanDistance(input1, input2):
    return np.sqrt(np.sum(np.square(input1-input2),axis=1))

def EuclideanDistance_WithNormalize(input1, input2):
    input1 = normalize(input1)
    input2 = normalize(input2)
    return np.sqrt(np.sum(np.square(input1-input2),axis=1))

if __name__ == "__main__":
    data1 = np.arange(0,8).reshape(2,4)
    data2 = np.random.rand(2,4)
    print(data1)
    print(data2)
    out = EuclideanDistance(data1, data2)
    print(out)

    n_data = normalize(data1)
    print(n_data)
    n_data = normalize_2(data1)
    print(n_data)