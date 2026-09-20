#logistic regression algorithm
from math import e

all_M=[]
all_B=[]

train_M=[]
train_B=[]

test_M=[]
test_B=[]

def data():
    global train_M
    global train_B
    global test_M
    global test_B
    global max_of_all
    f=open("./Data/Cancer_Data.csv",'r')
    a=0
    for i in f:
        l=i.split(',')[1:]
        if l[0]=='M':
            all_M.append(l)
        elif l[0]=='B':
            all_B.append(l)
        else:
            continue
    f.close()
    vals=[]
    for i in range(1,31):
        vals=[float(row[i]) for row in all_M+all_B]
        mn=min(vals)
        mx=max(vals)
        for row in all_M+all_B:
            if mx!=mn:
                row[i]=(float(row[i])-mn)/(mx-mn)
            else:
                row[i]=0
                
    cm=int(len(all_M)*0.8)
    cb=int(len(all_B)*0.8)
    train_M=all_M[:cm]
    train_B=all_B[:cb]
    test_M=all_M[cm:]
    test_B=all_B[cb:]


def sigmoid(x):
    return 1/(1+e**(-x))

def sigmoid(x):
    # If x is extremely negative, the sigmoid approaches 0
    if x < -500:
        return 0.0
    # If x is extremely positive, the sigmoid approaches 1
    elif x > 500:
        return 1.0
    else:
        return 1.0 / (1.0 + e**(-x))

def likelihood(k,b):
    h=1
    for x in train_M:
        X=0
        for i in range(1,len(x)):
            X+=k[i-1]*float(x[i])
        p=sigmoid(-X-b)
        h*=p**0*(1-p)**1
    for x in train_B:
        X=0
        for i in range(1,len(x)):
            X+=k[i-1]*float(x[i])
        p=sigmoid(-X-b)
        h*=p**1*(1-p)**0
    return h

def gradient_k(alpha):
    global k
    for w in range(len(k)):
        g=0
        for x in train_M:
            X=0
            for i in range(len(k)):
                X+=k[i]*float(x[i+1])
            g+=0*float(x[w+1])
            g-=sigmoid(X+b)*float(x[w+1])
        for x in train_B:
            X=0
            for i in range(len(k)):
                X+=k[i]*float(x[i+1])
            g+=1*float(x[w+1])
            g-=sigmoid(X+b)*float(x[w+1])
        k[w]+=g*alpha
        
def gradient_b(alpha):
    global b
    g=0
    for x in train_M:
        X=0
        for i in range(len(k)):
            X+=k[i]*float(x[i+1])
        g+=0
        g-=sigmoid(X+b)
    for x in train_B:
        X=0
        for i in range(len(k)):
            X+=k[i]*float(x[i+1])
        g+=1
        g-=sigmoid(X+b)
    b+=g*alpha

k=[0]*30
b=1

def logistic_regression(alpha,count):
    for i in range(count):
        gradient_k(alpha)
        gradient_b(alpha)

    say=0
    for x in test_M:
        X=0
        for i in range(1,len(x)):
            X+=k[i-1]*float(x[i])
        if round(sigmoid(X+b))==0:
            say+=1
    for x in test_B:
        X=0
        for i in range(1,len(x)):
            X+=k[i-1]*float(x[i])
        if round(sigmoid(X+b))==1:
            say+=1
    return 100*say/(len(test_M)+len(test_B))


data()
print(logistic_regression(0.0033,50))
