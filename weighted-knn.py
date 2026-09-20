# knn algorithm with cancer data    98.2% accuracy (k=7)
from math import sqrt,inf

all_M=[]
all_B=[]

train_M=[]
train_B=[]

test_M=[]
test_B=[]

max_of_all=[0]*30

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
        for u in range(30):
            if max_of_all[u]<float(l[u+1]):
                max_of_all[u]=float(l[u+1])
    f.close()
    cm=int(len(all_M)*0.8)
    cb=int(len(all_B)*0.8)
    train_M=all_M[:cm]
    train_B=all_B[:cb]
    test_M=all_M[cm:]
    test_B=all_B[cb:]

def test(i,k):
    l=[(inf,'X')]*k
    for j in train_M:
        d=0
        for u in range(30):
            d+=((float(i[u+1])-float(j[u+1]))/max_of_all[u])**2
        d=sqrt(d)
        if d<l[-1][0]:
            l.pop()
            l.append((d,'M'))
            l=sorted(l)
    for j in train_B:
        d=0
        for u in range(30):
            d+=((float(i[u+1])-float(j[u+1]))/max_of_all[u])**2
        d=sqrt(d)
        if d<l[-1][0]:
            l.pop()
            l.append((d,'B'))
            l=sorted(l)
    m=0
    b=0
    for j in range(k):
        if l[j][1]=='M':
            m+=1/l[j][0]
        if l[j][1]=='B':
            b+=1/l[j][0]
    if m>b:
        return 'M'
    return 'B'

def accuracy(k):
    all_c=len(test_M)+len(test_B)
    best_c=0

    for i in test_M:
        if test(i,k)=='M':
            best_c+=1

    for i in test_B:
        if test(i,k)=='B':
            best_c+=1
    
    return best_c*100/all_c

data()
print(accuracy(7))
