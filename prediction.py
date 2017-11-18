# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 00:58:32 2017

@author: ningshangyi
"""

import jieba
import codecs
import sys
import nltk
import random
import numpy as np
from collections import defaultdict

print('Reading Text')
def readnews(path='news.txt'):
    f = codecs.open(path, 'r', 'utf8')
    news = [eval(i) for i in f.readlines()]
    f.close()
    return news

news = readnews()
labelDict = {}
f = open('train.txt', 'r')
for line in f:
    fn, ids = line.split()
    ids = [int(i) for i in ids.split(',')]
    for i in ids:
        if fn == '+1':
            labelDict[i] = 1
        else:
            labelDict[i] = -1
f.close()
labelDict2 = {}
f = open('test.txt', 'r')
for line in f:
    fn, ids = line.split()
    ids = [int(i) for i in ids.split(',')]
    for i in ids:
        if fn == '+1':
            labelDict2[i] = 1
        else:
            labelDict2[i] = -1
f.close()

def removeUnChinese(l):
    toRemove = []
    for item in l:
        for char in item:
            if char < u'\u4e00' or char > u'\u9fa5': #判断是否是汉字,不是则去掉
                toRemove.append(item)  
                break
    for item in toRemove:
        l.remove(item)
    return l

def seg2(news):
    count = 0
    for item in news:
        count += 1
        item['title']   = list(jieba.cut(item['title']))
        item['content'] = list(jieba.cut(item['content']))
        removeUnChinese(item['title'])
        removeUnChinese(item['content'])
        sys.stdout.write('\rWord Segmentation %.2f%%' % (count/303.39))
    return news
newsSeg = seg2(news)

print('\n========================================')
print('Part I: naive bayes')
print('========================================\n')


    
def seg(dataset='all', lableDict=labelDict, news=news):
    titleDict = defaultdict(int)
    contentDict = defaultdict(int)
    count = 0
    c = len(labelDict)
    for piece in news:
        title   = piece['title']
        content = piece['content']
        ids     = int(piece['id'])
        if ids in labelDict:
            if dataset == lableDict[ids]:
                for word in set(title):
                    titleDict[word]   += 1
                for word in set(content):
                    contentDict[word] += 1
            count += lableDict[ids]
        if count % 100 == 0:
            sys.stdout.write('\rBuiding FreqDist, step %s/2' % (str(dataset/2+3/2)))
    return titleDict, contentDict

def getCommon(D):
    l = sorted(D.items(), key=lambda i:i[1], reverse=True)
    toRemove = []
    for item in l:
        for char in item[0]:
            if char < u'\u4e00' or char > u'\u9fa5': #判断是否是汉字,不是则去掉
                toRemove.append(item)  
                break
    for item in toRemove:
        l.remove(item)
    return l

 
FalseTitle, FalseContent = seg(-1)
TrueTitle,  TrueContent  = seg(1)  

#
#commonTrueTitle    = getCommon(TrueTitle)
#commonFalseTitle   = getCommon(FalseTitle)
#commonTrueContent  = getCommon(TrueContent)
#commonFalseContent = getCommon(FalseContent)

def findKeyWord(r1, r2, label=1, delta=5, thres=500, k=1.7):
    k = np.sum([r1[x] for x in r1]) / np.sum([r2[x] for x in r2])
    bayesPDict = defaultdict(float)
    for i in r1:
        if r1[i] > thres:
            flag = 0
            for char in i:
                if char < u'\u4e00' or char > u'\u9fa5': #判断是否是汉字,不是则去掉
                    flag = 1                
                    break
            if flag:
                continue
            if i in r2:
                bayesPDict[(i, label)]  = np.log(1/(1+(r2[i]+delta)/(r1[i]+delta)*k))
#                bayesPDict[(i, -label)] = np.log(1/(1+(r1[i]+delta)/(r2[i]+delta)))
            else:
                bayesPDict[(i, label)]  = np.log(1/(1+(delta)/(r1[i]+delta)*k))
#                bayesPDict[(i, -label)] = np.log(1/(1+(r1[i]+delta)/(delta)))
    for j in r2:
        if (j, label) not in bayesPDict and r2[j] > thres/1.7:
            flag = 0
            for char in j:
                if char < u'\u4e00' or char > u'\u9fa5': #判断是否是汉字,不是则去掉
                    flag = 1                
                    break
            if flag:
                continue
            if j in r1:
                bayesPDict[(i, label)]  = np.log(1/(1+(r2[j]+delta)/(r1[j]+delta)*k))
#                bayesPDict[(i, -label)] = np.log(1/(1+(r1[i]+delta)/(r2[i]+delta)))
            else:
                bayesPDict[(j, label)]  = np.log(1/(1+(r2[j]+delta)/(delta)*k))
#            bayesPDict[(j, -label)] = np.log(1/(1+(delta)/(r2[j]+delta)))
    return bayesPDict

bayesPTitle   = findKeyWord(TrueTitle, FalseTitle, thres=50, delta=1)
bayesPContent = findKeyWord(TrueContent, FalseContent, delta=15)


def naivebayes(n, typ='title'):
    firstP = sorted(bayesPContent.items(), key=lambda i:i[1], reverse=True)[0:n]\
           + sorted(bayesPContent.items(), key=lambda i:i[1], reverse=False)[0:n]\
           if typ == 'content' else\
             sorted(bayesPTitle.items(), key=lambda i:i[1], reverse=True)[0:n]\
           + sorted(bayesPTitle.items(), key=lambda i:i[1], reverse=False)[0:n]
           
    commonWords = [x[0][0] for x in firstP]
    prob = defaultdict(float)
    count = 0
    for item in news:
        count   += 1
        sys.stdout.write('\rcomputing id %6d/30339\t\t\t' % count)
        title   = item['title']
        content = item['content']
        ids     = int(item['id'])
        logpt   = np.log(0.5)
        logpf   = np.log(0.5)
        seg     = title if typ == 'title' else content
        for word in seg:
            if word in commonWords:
#                sys.stdout.write('\r%8d,%8s: %8.2f' % (count, word, 100*np.exp(bayesPContent[(word,1)])))
                p = bayesPContent[(word, 1)]
                logpt += p
                logpf += np.log(1-np.exp(p))
#                sys.stdout.write('\r%s\t\t\t' % word)
        prob[ids] = np.exp(logpt)/(np.exp(logpt)+np.exp(logpf))
    return prob

probDict = naivebayes(10, typ='content')
def writeAnswer(resultpath='result.txt', testpath='test.txt', prob=probDict, k=np.median(list(probDict.values()))):
    
    t = open(testpath,'r')
    f = open(resultpath, 'w')
    idss = [i.split() for i in t.readlines()]
    t.close()
    count = 0
    for i in idss:
        tl, item = i
        r = item.split(',')
        pt    = np.prod([prob[int(x)]/k/2 for x in r])
        pf    = np.prod([(1-prob[int(x)]/k/2) for x in r])
        label = '+1' if pt>=pf else '-1'
        if label != tl:
            count += 1
        f.write(label)
        f.write('\n')
    print('Answer worte to', resultpath)
    f.close
    
def testF1(k=np.median(list(probDict.values())), d=labelDict2):
    t = 0
    f = 0
    for item in d:
        if (probDict[item]>k)^(d[item]==-1):
            t += 1
        else:
            f += 1
    return t/(t+f)



print('\ntrain error:', 1-testF1(d=labelDict))
print('test error:', 1-testF1())
writeAnswer()

'''
    Naive Bayes with NLTK
'''

print('========================================')
print('Part II-1: naive bayes with nltk')
print('========================================\n')

content = []
for item in newsSeg:
    content += item['content']
allWords = nltk.FreqDist(content)

commonW = allWords.most_common(100)
def docFeature(doc, words=commonW):
    docWords = set(doc)
    feature = {}
    for word in words:
        feature[word[0]] = (word[0] in docWords)
    return feature
docs = []
for item in newsSeg:
    try:
        docs += [(item['content'], labelDict[int(item['id'])])]
    except:
        pass
docs2 = []
for item in newsSeg:
    try:
        docs2 += [(item['content'], labelDict2[int(item['id'])])]
    except:
        pass
train = nltk.apply_features(docFeature, docs)
classifier = nltk.NaiveBayesClassifier.train(train)
test = nltk.apply_features(docFeature, docs2)


print('\ntrain error:', 1-nltk.classify.accuracy(classifier,train))
print('test error:', 1-nltk.classify.accuracy(classifier,test))



'''
    Kmeans
'''


print('========================================')
print('Part II-2: K-means with nltk')
print('========================================\n')


kmeansTrain = []
for x in train:
    t = []
    for i in x[0].values():
        a = 1 if i else 0
        t.append(a)
    kmeansTrain.append(np.array(t))

km = nltk.cluster.kmeans.KMeansClusterer(num_means=2,distance=nltk.cluster.util.euclidean_distance)
km.cluster(kmeansTrain)

kmeansTest = []
for x in test:
    t = []
    for i in x[0].values():
        a = 1 if i else 0
        t.append(a)
    kmeansTest.append(np.array(t))

t, f, p, n = [0]*4
for i in range(len(train)):
    real = train[i][1]
    result = km.classify(kmeansTrain[i])
    if (real == 1 and result == 0):
        t += 1
        p += 1
    elif (real == -1 and result == 1):
        t += 1
        n += 1
    elif real == 1:
        f += 1
        p += 1
    else:
        f += 1
        n += 1


print('\ntrain error:', f/(t+f))
t, f, p, n = [0]*4
for i in range(len(test)):
    real = test[i][1]
    result = km.classify(kmeansTest[i])
    if (real == 1 and result == 0):
        t += 1
        p += 1
    elif (real == -1 and result == 1):
        t += 1
        n += 1
    elif real == 1:
        f += 1
        p += 1
    else:
        f += 1
        n += 1


print('test error:', f/(t+f))


"""
    DecisionTree
"""
print('========================================')
print('Part II-3: decision tree')
print('========================================\n')

positiveWord = []
pwl = sorted(bayesPContent.items(), key=lambda i:i[1], reverse=True)
for item in pwl:
    positiveWord.append(item[0][0])
    count = 0
    for i in docs:
        for j in positiveWord:
            if j in i[0]:
                count +=1 
                break
    sys.stdout.write('\rDecision Tree: Extracting words %d/12080, added %s\t' % (count, item[0][0]))
    if count >= 12080:
        break

def docFeature2(doc, words=positiveWord):
    docWords = set(doc)
    feature = {}
    for word in words:
        feature[word] = (word in docWords)
    return feature
trainDT = nltk.apply_features(docFeature2, docs)
testDT = nltk.apply_features(docFeature2, docs2)

def decisionTreeClassifier(data):
    t, f, u = 0, 0, 0
    for i in data:
        output = -1
        for j in range(len(positiveWord)):
            if i[0][positiveWord[j]]:
                output = 1
                break
        if output == i[1]:
            t += 1
        elif output:
            f += 1
        else:
            u += 1
    return t/(t+f)

print('\ntrain error:', 1-decisionTreeClassifier(trainDT))
print('test error:', 1-decisionTreeClassifier(testDT))
def writeAnswerDT():
    f = open('test.txt', 'r')
    res = open('result2.txt', 'w')
    count = 0
    for line in f:
        count += 1
        fn, ids = line.split()
        ids = [int(i) for i in ids.split(',')]
        
        p, n = 0, 0
        for i in ids:
            for item in news:
                if item['id'] == i:
                    for j in range(len(positiveWord)):
                        if positiveWord[j] in item['content']:
                            p += 1
                            break
                    else:
                        n += 1
        if p>n:
            label = '+1'
        elif n<p:
            label = '-1'
        else:
            r = random.randint(1,2)
            label = '+1' if r == 1 else '-1'
        res.write(label)
        res.write('\n')
        sys.stdout.write('\rWriting Answer %d/3000'%count)
    f.close()
    res.close()
    sys.stdout.write('\nAnswer of decision tree wrote to result2.txt')

writeAnswerDT()
