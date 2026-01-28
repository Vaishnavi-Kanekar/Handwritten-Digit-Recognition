# Handwritten Digit Recognition

In this project, I have created a handwritten digit recognition model trained on MNIST dataset using a fully connected neural network built from scratch with Python and NumPy. 
The goal of this project is to understand the core mathematics and logic behind neural networks, including forward propagation, backpropagation, activation functions, and loss computation.

---

## Project Structure
```
Handwritten-Digit-Recognition-From-Scratch/
│
├── src/
│   ├── create_train_dataset.py
│   ├── create_test_dataset.py
│   ├── train_model.py
│   ├── test_model.py
│
├── README.md
```

---

## Model Architecture
- **Input Layer:** 784 neurons (28×28 image flattened)
- **Hidden Layer 1:** 128 neurons (ReLU)
- **Hidden Layer 2:** 64 neurons (ReLU)
- **Output Layer:** 10 neurons (Softmax)

---

## How to run
Downlaod the MNIST image dataset and place it in dataset directory. 
```bash
python src/create_train_dataset.py
python src/train_model.py
python src/create_test_dataset.py
python src/test_model.py
```

---

## Model Performance
- Test Accuracy is **97.85%** on the MNIST test dataset.

---

## Author
**Vaishnavi Kanekar**  
