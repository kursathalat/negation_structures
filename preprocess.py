
# author: Mustafa Kursat Halat

# To preprocess IMDb data for the negation model

# The reviews should be in the same directory named "reviews", and there within "pos" or "neg" folders depending on their sentiment

# Currently, without reviews in respective files, script won't work, you need to replace example files with actual .txt files

# Same goes for the test corpus, they should be in test folder

# To save each training corpus and test corpus, uncomment the np.save command lines


from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os
import pandas as pd
import numpy as np

sententialNegItems = ["n't","cannot","never","hardly","barely","scarcely","rarely","seldom","hardly ever","barely ever","not ever"]
constituentNegItems = ["nobody","no one","noone","none","nothing","nowhere"]
verbTaggings = ["VB","VBD","VBG","VBN","VBP","VBZ", "MD"]         #list of tags for verbs and modals used by NLTK POS Tagger
adverbTaggings = ["RB","RBR","RBS"]                               #list of tags for adverbs used by NLTK POS Tagger

annotatedReviews = []
noNeg = []

def getFileNames():
    global fileNamesPos
    global fileNamesNeg

    fileNamesPos = os.listdir("reviews/pos")                    #get the file names of the positive reviews in txt format
    for eachfile in fileNamesPos:                               #prevent any system-generated / non-relevant file to be in the list
        if eachfile[-3:] != "txt":
            fileNamesPos.remove(eachfile)

    fileNamesPos = fileNamesPos[0:1000]

    fileNamesNeg = os.listdir("reviews/neg")                    #get the file names of the negative reviews in txt format
    for eachfile in fileNamesNeg:                               #prevent any system-generated / non-relevant file to be in the list
        if eachfile[-3:] != "txt":
            fileNamesNeg.remove(eachfile)

    fileNamesNeg = fileNamesNeg[0:1000]



def splitData(fileNames,sentiment):

    for file in fileNames:
        sententialNegCount = 0
        constituentNegCount = 0

        if sentiment == 0:
            directory = "neg"
        if sentiment == 1:
            directory = "pos"

        with open("reviews/"+directory+"/"+file) as f: #open all txt files in the dir
            rawComment = f.readlines()[0].replace("<br /><br />"," ") #change every tag to None from the txt file
        comment = word_tokenize(rawComment)

        for index, token in enumerate(comment):
            if token in sententialNegItems:
                sententialNegCount += 1
            if token in constituentNegItems:
                constituentNegCount += 1

            # NOT
            if token == "not": #if we find a case of "not"
                if pos_tag(comment)[index-1][1] in verbTaggings: #and if the previous token is a verb (so this "not" is sentential)
                    sententialNegCount += 1 #then, sententials get another point
                else: #if the previous token is not a verb and it is something else
                    constituentNegCount += 1 #then, it is a constituent negator

            # NO
            if token == "no":
                if pos_tag(comment)[index+1][1] in adverbTaggings:
                    sententialNegCount += 1 #e.g. no longer
                elif pos_tag(comment)[index+1][1] == "DT":
                    if pos_tag(comment)[index+2][1] in adverbTaggings:
                        sententialNegCount += 1
                    else:
                        constituentNegCount += 1
                else:
                    constituentNegCount += 1

        if sententialNegCount == 0 and constituentNegCount == 0:
            noNeg.append([rawComment,sentiment])
        else:
            annotatedReviews.append([file,rawComment,sentiment,sententialNegCount,constituentNegCount,sententialNegCount-constituentNegCount])

getFileNames()

splitData(fileNamesPos,1)
splitData(fileNamesNeg,0)

allReviews = pd.DataFrame(annotatedReviews, columns=["review","comment","sentiment","n_sentNeg","n_consNeg","difference"])

ascendingData = allReviews.copy()
descendingData = allReviews.copy()

descendingData.sort_values(by=["difference"], inplace=True, ascending=False)
ascendingData.sort_values(by=["difference"], inplace=True)





#CREATE EPOCH 1 NEGATIVE DATABASE

splitDataDes = np.array_split(descendingData, 3)[0]
splitDataAsc = np.array_split(ascendingData, 16)


concatData1 = pd.concat([splitDataDes,splitDataAsc[0]])
concattedAsc = 1
while concatData1["n_sentNeg"].sum()/concatData1["n_consNeg"].sum() < 3:
    concatData1 = concatData1.append(splitDataAsc[concattedAsc])
    concattedAsc += 1

print("CONCAT1",concatData1["n_sentNeg"].sum()/concatData1["n_consNeg"].sum())




#CREATE EPOCH 2 NEGATIVE DATABASE

splitDataDes = np.array_split(descendingData, 70)[0]
splitDataAsc = np.array_split(ascendingData, 15)

concatData2 = pd.concat([splitDataDes,splitDataAsc[0]])
concattedAsc = 1
while concatData2["n_sentNeg"].sum()/concatData2["n_consNeg"].sum() < 1:
    concatData2 = concatData2.append(splitDataAsc[concattedAsc])
    concattedAsc += 1


print("CONCAT2",concatData2["n_sentNeg"].sum()/concatData2["n_consNeg"].sum())




#CREATE EPOCH 3 NEGATIVE DATABASE
    
splitDataDes = np.array_split(descendingData, 120)[0]
splitDataAsc = np.array_split(ascendingData, 20)

concatData3 = pd.concat([splitDataAsc[0],splitDataAsc[1]])

concattedAsc = 2

while concatData3["n_sentNeg"].sum()/concatData3["n_consNeg"].sum() > 0.34:
    concatData3 = concatData3.append(splitDataAsc[concattedAsc])
    concattedAsc += 1

print("CONCAT3",concatData3["n_sentNeg"].sum()/concatData3["n_consNeg"].sum())




###CONCATENATE FILES

baseReviews = pd.DataFrame(noNeg, columns=["comment","sentiment"])


#EPOCH 1

firstEpochFinal = pd.concat([baseReviews,concatData1])

print(firstEpochFinal)
print(len(firstEpochFinal))
print(firstEpochFinal["n_sentNeg"].sum())
print(firstEpochFinal["n_consNeg"].sum())
print(firstEpochFinal["n_sentNeg"].sum()/firstEpochFinal["n_consNeg"].sum()) 

firstEpochFinalDict = dict(zip(["text","label"], [firstEpochFinal["comment"].to_list(),firstEpochFinal["sentiment"].to_list()]))
#np.save('train_corpus_first_epoch.npy', firstEpochFinalDict) 



#EPOCH2

secondEpochFinal = pd.concat([baseReviews,concatData2])

print(secondEpochFinal)
print(len(secondEpochFinal))
print(secondEpochFinal["n_sentNeg"].sum())
print(secondEpochFinal["n_consNeg"].sum())
print(secondEpochFinal["n_sentNeg"].sum()/secondEpochFinal["n_consNeg"].sum()) 

secondEpochFinalDict = dict(zip(["text","label"], [secondEpochFinal["comment"].to_list(),secondEpochFinal["sentiment"].to_list()]))
#np.save('train_corpus_second_epoch.npy', secondEpochFinalDict) 




#EPOCH 3

thirdEpochFinal = pd.concat([baseReviews,concatData3])

print(thirdEpochFinal)
print(len(thirdEpochFinal))
print(thirdEpochFinal["n_sentNeg"].sum())
print(thirdEpochFinal["n_consNeg"].sum())
print(thirdEpochFinal["n_sentNeg"].sum()/thirdEpochFinal["n_consNeg"].sum()) 


thirdEpochFinalDict = dict(zip(["text","label"], [thirdEpochFinal["comment"].to_list(),thirdEpochFinal["sentiment"].to_list()]))
#np.save('train_corpus_third_epoch.npy', thirdEpochFinalDict) 



#WHOLE CORPUS

concatCorpus = pd.concat([baseReviews,concatData1,concatData2,concatData3]).drop_duplicates()

print(concatCorpus)
print(len(concatCorpus))
print(concatCorpus["n_sentNeg"].sum())
print(concatCorpus["n_consNeg"].sum())
print(concatCorpus["n_sentNeg"].sum()/concatCorpus["n_consNeg"].sum()) 


concatCorpusFinalDict = dict(zip(["text","label"], [concatCorpus["comment"].to_list(),concatCorpus["sentiment"].to_list()]))
#np.save('train_whole_corpus.npy', concatCorpusFinalDict) 


# CREATE TEST CORPUS

fileNamesTestPos = os.listdir("test/pos")                    #get the file names of the positive reviews in txt format
for eachfile in fileNamesTestPos:                               #prevent any system-generated / non-relevant file to be in the list
    if eachfile[-3:] != "txt":
        fileNamesTestPos.remove(eachfile)

fileNamesTestPos = fileNamesTestPos[0:400]


fileNamesTestNeg = os.listdir("test/neg")                    #get the file names of the negative reviews in txt format
for eachfile in fileNamesTestNeg:                               #prevent any system-generated / non-relevant file to be in the list
    if eachfile[-3:] != "txt":
        fileNamesTestNeg.remove(eachfile)

fileNamesTestNeg = fileNamesTestNeg[0:400]


annotatedTestData = []

def splitDataTest(fileNames,sentiment):

    for file in fileNames:
        if sentiment == 0:
            directory = "neg"
        if sentiment == 1:
            directory = "pos"

        with open("test/"+directory+"/"+file) as f: #open all txt files in the dir
            rawComment = f.readlines()[0].replace("<br /><br />"," ") #change every tag to None from the txt file

        annotatedTestData.append([rawComment,sentiment])

splitDataTest(fileNamesTestPos,1)
splitDataTest(fileNamesTestNeg,0)

allReviewsTest= pd.DataFrame(annotatedTestData, columns=["comment","sentiment"])

testFinal = dict(zip(["text","label"], [allReviewsTest["comment"].to_list(),allReviewsTest["sentiment"].to_list()]))
#np.save('test_corpus.npy', testFinal)