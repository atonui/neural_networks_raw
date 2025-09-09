import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
  '''class to define a layer of neurons'''
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    # weights matrix is determined by the size of the input coming into each neuron and how many neurons do we want
    self.biases = np.zeros((1, n_neurons))
    ''' each neuron has 1 bias value so the shape is 1 x number of neurons
    we initialise the bias to zeros but if you have a dead network i.e. the neurons are not firing/ learning, you may want to adjust this'''
    pass
  def forward(self, inputs):
    self.output = np.dot(inputs,self.weights) + self.biases


class Activation_ReLU:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)

class Activation_Softmax: # inputs will come in batches
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    self.output = probabilities

class Loss:
  '''Class to implement the loss function'''
  def calculate(self, predicted, target):
    sample_losses = self.forward(predicted, target)
    batch_loss = np.mean(sample_losses)
    return batch_loss

class Loss_CategoricalCrossentropy(Loss):
  '''Class to implement the categorical crossentropy loss function'''
  def forward(self, y_pred, y_target):
    samples = len(y_pred)
    # clip data to prevent division by 0 errors
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if len(y_target.shape) == 1:
      correct_confidences = y_pred_clipped[range(samples), y_target]
    elif len(y_target.shape) == 2:
      correct_confidences = np.sum(y_pred_clipped * y_target, axis=1)

    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)