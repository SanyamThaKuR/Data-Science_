import numpy
class dense_layer:
   def __init__(self,n_inputs,n_neurons) -> None:
      #^ creating weights using random generator
      self.weights1 = 0.10*numpy.random.rand(n_inputs,n_neurons)
      self.biases = numpy.zeros((1,n_neurons))
   def forward(self,input)-> None:
      self.output = numpy.dot(input,self.weights1)+self.biases
      
