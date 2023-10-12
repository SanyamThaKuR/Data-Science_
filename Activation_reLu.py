import numpy
class Activation_Func:
    def forward(self,input)->None:
        for i in input:
            self.output = numpy.maximum(0,input)

        
            
        