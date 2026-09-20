#linear regression algorithm
from math import inf

al=[]
train=[]
test=[]

max_of_all=[0]*2

def data():
    global al
    global train
    global test
    f=open("./Data/Salary_Data.csv",'r')
    for i in f:
        l=i.split(',')
        if l[0].isalpha():
            continue
        al.append(l)
        for u in range(2):
            if max_of_all[u]<float(l[u]):
                max_of_all[u]=float(l[u])
    f.close()
    c=int(len(al)*0.7)
    train=al[:c]
    test=al[c:]

def MSE(dataset,k,b):
    cem=0
    for i in dataset:
        cem+=(float(i[1])/max_of_all[1]-(k*float(i[0])/max_of_all[0]+b))**2
    return cem/len(dataset)

def gradient_k(dataset,k,b):
    cem=0
    for i in dataset:
        cem+=(-2*float(i[1])/max_of_all[1]*float(i[0])/max_of_all[0]+b)+2*k*float(i[0])/max_of_all[0]+b**2+2*float(i[0])/max_of_all[0]+b*b
    return cem/len(dataset)

def gradient_b(dataset,k,b):
    cem=0
    for i in dataset:
        cem+=(-2*float(i[1])/max_of_all[1]+2*k*float(i[0])/max_of_all[0]+b)+2*b
    return cem/len(dataset)

def linear_regression(alpha,count):
    k,b=2,2
    mn=(0,0)
    min_err=inf
    for i in range(count):
        k=k-alpha*gradient_k(train,k,b)
        b=b-alpha*gradient_b(train,k,b)
        err=MSE(train,k,b)
        if err<min_err:
            min_err=err
            mn=(k,b)
    return MSE(test,mn[0],mn[1])

data()
print(linear_regression(0.0033,100))
