import numpy as np

# Load training dataset
# X → image vectors (784 features per image)
# Y → digit labels (0–9)
data=np.load("mnist_dataset_train.npz")
x=data['X']
y=data['Y']

# Shuffle dataset
np.random.seed(7)
indices=np.random.permutation(len(x))
x=x[indices]
y=y[indices]

batchsize=60
epoch=100        
alpha=0.1               #Learning Rate

# Initialize weights & biases
w1=np.random.rand(784,128)*0.01
b1=np.random.rand(1,128)

w2=np.random.rand(128,64)*0.01
b2=np.random.rand(1,64)

w3=np.random.rand(64,10)*0.01
b3=np.random.rand(1,10)

# Convert labels to one-hot encoding
def build_y(y):
    return np.eye(10)[y]

# ReLU activation function
def activation(x):
    return np.where(x<=0,0,x)

# Derivative of ReLU
def der_activation(x):
    return np.where(x<=0,0,1)

# Softmax for output layer
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# Training loop
for i in range(epoch):
    for s in range(0,len(x),batchsize):
        end=s+batchsize
        x_new=x[s:end]
        y_new=y[s:end]
        y_true=build_y(y_new)

        #forward pass
        h1=np.dot(x_new,w1)+b1
        a1=activation(h1)
        
        h2=np.dot(a1,w2)+b2
        a2=activation(h2)

        z=np.dot(a2,w3)+b3
        ypred=softmax(z)

        #cross-entropy loss
        epsilon = 1e-9
        loss = -np.mean(np.sum(y_true * np.log(ypred + epsilon), axis=1))

        #backpropogate
        dl_z = (ypred - y_true) / len(x_new)

        #output layer
        dz_w3=a2.T                      
        dl_w3=np.dot(dz_w3,dl_z)    
        dl_b3=np.sum(dl_z, axis=0, keepdims=True)

        #2nd layer
        dz_a2=w3.T
        dl_a2=np.dot(dl_z,dz_a2)

        da_h2=der_activation(h2)
        dl_h2=dl_a2*da_h2

        dh2_w2=a1.T
        dl_w2=np.dot(dh2_w2,dl_h2)
        dl_b2=np.sum(dl_h2, axis=0, keepdims=True)

        #1st layer
        dh2_a1=w2.T
        dl_a1=np.dot(dl_h2,dh2_a1)
        dl_h1=dl_a1*der_activation(h1)

        dh1_w1=x_new.T
        dl_w1=np.dot(dh1_w1,dl_h1)
        dl_b1=np.sum(dl_h1, axis=0, keepdims=True)

        #store old weights
        w1_old, w2_old, w3_old = w1.copy(), w2.copy(), w3.copy()

        #update weights
        w3=w3-alpha*dl_w3
        b3=b3-alpha*dl_b3

        w2=w2-alpha*dl_w2
        b2=b2-alpha*dl_b2

        w1=w1-alpha*dl_w1
        b1=b1-alpha*dl_b1

        weight_change = (
        np.sum(np.abs(w1_old - w1)) +
        np.sum(np.abs(w2_old - w2)) +
        np.sum(np.abs(w3_old - w3))
        )

    avg_weight_change=weight_change/len(x)

    print(f"{i+1} loss is {loss:.5f}")

    #check if the weights are converged
    if avg_weight_change < 0.000001:
        print("Weights converged — stopping training.")
        break


#save trained model
np.savez("weights.npz",w1=w1,b1=b1,w2=w2,b2=b2,w3=w3,b3=b3)