import numpy as np

# load test dataset
data=np.load("mnist_dataset_test.npz")
x=data['X']
y=data['Y']

#shuffle test data
indices=np.random.permutation(len(x))
x=x[indices]
y=y[indices]

#convert true labels to one-hot encoding
y_true=np.eye(10)[y]

#load trained model weights and biases
weights=np.load("weights.npz")
w1=weights['w1']
b1=weights['b1']
w2=weights['w2']
b2=weights['b2']
w3=weights['w3']
b3=weights['b3']

#ReLU activation function
def activation(x):
    return np.where(x<=0,0,x)

#derivative of ReLU
def der_activation(x):
    return np.where(x<=0,0,1)

#softmax function
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

#first hidden layer
h1=np.dot(x,w1)+b1
a1=activation(h1)
    
#second hidden layer
h2=np.dot(a1,w2)+b2
a2=activation(h2)

#output layer
z=np.dot(a2,w3)+b3
ypred=softmax(z)

#calculate loss
epsilon = 1e-9
loss = -np.mean(np.sum(y_true * np.log(ypred + epsilon), axis=1))
print(f"loss is {loss:.5f}")

pred_y = np.argmax(ypred, axis=1)

# Calculate accuracy
accuracy = np.mean(pred_y ==y) * 100

print("Actual y:",y)
print("predicted y:",pred_y)
print(f"Accuracy: {accuracy:.2f}%")
