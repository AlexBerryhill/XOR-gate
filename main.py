import numpy as np

input_features = np.array([[0,0],[0,1],[1,0],[1,1]])

#establish target output
target_output = np.array([[0,1,1,0]])
target_output = target_output.reshape(4,1)

#Define weights
hweights = np.random.rand(2,4)
oweights = np.random.rand(4,1)
print(hweights.shape)
print(oweights.shape)


#learning rate
lr = 0.05

#Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
#The derivitive of sigmoid
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#training
for epoch in range(200000):
    
    #----Feedforward----
    #Hidden Layer
    in_h = np.dot(input_features,hweights)
    out_h = sigmoid(in_h)
    
    #Output Layer
    in_o = np.dot(out_h,oweights)
    out_o = sigmoid(in_o)
    
    #Calculate MSE
    error_out = ((0.5)*(np.power((out_o - target_output), 2)))
    
    #Derivatives for Output Layer
    derror_douto = out_o - target_output
    douto_dino = sigmoid_der(in_o)
    dino_dwo = out_h
    
    derror_wo = np.dot(dino_dwo.T,derror_douto*douto_dino)
    
    #--------------------
    #Derivatives for Hidden Layer
    derror_dino = derror_douto * douto_dino
    dino_douth = oweights
    derror_douth = np.dot(derror_dino,dino_douth.T)
    douth_dinh = sigmoid_der(in_h)
    dinh_dwh = input_features
    derror_wh = np.dot(dinh_dwh.T, douth_dinh*derror_douth)
    
    #Update Weights
    hweights -= lr * derror_wh
    oweights -= lr * derror_wo
    
    if epoch == 50000:
        print("25%")
    elif epoch == 100000:
        print("50%")
    elif epoch == 150000:
        print("75%")
    elif epoch == 180000:
        print("90%")

#Check final values
print('--------------------')
print("Hidden Layer Weights:\n",hweights)
print('--------------------')
print("Output Weights:\n",oweights)
print('--------------------')
#print("Final Bias:\n",bias)
#print('--------------------')
print("Final Error:\n",error_out.sum())

#Testing
testInput = []

print("Start Testing")
cont = True
while(cont):
    print("Enter your input: ")
    testInput = np.array([int(x) for x in input().split()])
    
    #Hidden Layer
    in_h = np.dot(testInput,hweights)
    out_h = sigmoid(in_h)
        
    #Output Layer
    in_o = np.dot(out_h,oweights)
    out_o = sigmoid(in_o)
    
    print("Output is: ", out_o)
    
    if input("Continue? y/n: ") == "y":
        cont = True
    else:
        cont = False

