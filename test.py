import numpy as np
arr= np.ones((5,10))
arr2=np.random.randn(10,5)
# print(np.dot(arr,arr2))

X= np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

for a,b in zip(X,Y):
    print("****")
    print(a)
    print("##")
    print(b)
   