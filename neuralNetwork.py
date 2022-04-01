import numpy as np

class AdamOptimizer():
    
    def __init__(self, nInupts, nOutputs, learningRate, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        self.mdW = np.zeros((nInupts, nOutputs))
        self.vdW = np.zeros((nInupts, nOutputs))
        self.mdB = np.zeros(nOutputs)
        self.vdB = np.zeros(nOutputs)
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def update(self, t, w, b, dW, dB):
        self.mdW = self.beta1 * self.mdW + (1-self.beta1)*dW
        self.mdB = self.beta1 * self.mdB + (1-self.beta1)*dB
        
        self.vdW = self.beta2*self.vdW + (1-self.beta2)*(dW**2)
        self.vdB = self.beta2*self.vdB + (1-self.beta2)*(dB**2)        

        mdWCorr = self.mdW/(1-self.beta1**t)
        mdBCorr = self.mdB/(1-self.beta1**t)
        vdWCorr = self.vdW/(1-self.beta2**t)
        vdBCorr = self.vdB/(1-self.beta2**t)

        w = w - self.learningRate*(mdWCorr/(np.sqrt(vdWCorr)+self.epsilon))
        b = b - self.learningRate*(mdBCorr/(np.sqrt(vdBCorr)+self.epsilon))
        return w, b

class Layer:

    def __init__(self):
        pass
        
    def forward(self, input, keep_rate):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]        
        d_layer_d_input = np.eye(num_units)        
        return np.dot(grad_output, d_layer_d_input), 0, 0

class Linear(Layer):

    def __init__(self):
        pass

    def forward(self, input, keep_rate):
        return input

    def backward(self, input, grad_output):
        return grad_output, 0, 0

class ReLU(Layer):

    def __init__(self):
        pass

    def forward(self, input, keepRate):
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad, 0, 0

class Dense(Layer):

    def __init__(self, numInputs, numOutputs, learningRate):
        self.learningRate = learningRate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(numInputs+numOutputs)), 
                                        size = (numInputs,numOutputs))
        self.biases = np.zeros(numOutputs)
        self.adam = AdamOptimizer(numInputs, numOutputs, learningRate)

    def forward(self, input, keep_rate):
        self.keep_rate = keep_rate
        D = np.random.rand(len(self.biases))
        D = D < keep_rate
        self.activeNeurons = D
        return D * (np.dot(input, self.weights) + self.biases)

    def getAdam(self):
        return self.adam

    def update(self, t, dW, dB):
        w, b = self.adam.update(t, self.weights, self.biases, dW, dB)
        self.weights = w
        self.biases = b

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output) * (self.activeNeurons/self.keep_rate)
        grad_biases = grad_output.mean(axis=0)*input.shape[0] * (self.activeNeurons/self.keep_rate)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        return grad_input, grad_weights, grad_biases

def SquareError(input, predictedResult):
    diff = input - predictedResult
    diff **=2
    return diff

def grad_SquareError(error, reference_answers):
    # Compute MeanSquareError gradient from outputs[batch,n_classes] and ids of correct answers
    grad = np.zeros_like(error)
    for i in range(len(error)):
        grad[i] = (2*(error[i] - reference_answers[i]))
    return grad

#OVERALL NETWORK FUNCTIONS

def forward(network, X, keep_rate):
    activations = []
    input = X

    for layer in network:
        if layer is not network[-2]:
            activations.append(layer.forward(input, keep_rate))
        else:
            activations.append(layer.forward(input, 1))
        input = activations[-1]

    assert len(activations) == len(network)
    return activations

def predict(network, X):
    out = forward(network,X, 1)[-1]
    return out


def train(network,X,y, keep_rate):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.
    
    # Get the layer activations
    layer_outputs = forward(network,X, keep_rate)
    layer_inputs = [X]+layer_outputs  #layer_input[i] is an input for network[i]
    out = layer_outputs[-1]
    
    # Compute the loss and the initial gradient
    loss_grad = grad_SquareError(out, y)
    
    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    deltaWeights = []
    deltaBiases = []
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad, dW, dB = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        deltaWeights.insert(0,dW)
        deltaBiases.insert(0,dB)

    return (deltaWeights, deltaBiases)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
