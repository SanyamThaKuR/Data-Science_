
from Dense_layer import dense_layer
from Activation_reLu import Activation_Func
from create_data import data  

if __name__ == '__main__':
   
   X,y = data.create(10,3)   

   layer1 = dense_layer(2,5)
   layer1.forward(X)

   Activation_Function = Activation_Func()
   Activation_Function.forward(layer1.output)
   print(Activation_Function.output)