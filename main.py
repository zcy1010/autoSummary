import math
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd
import sys
from rouge import Rouge
from pathlib import Path
import os


# np.seterr(divide='ignore', invalid='ignore')


def getFileText(filename) -> str:
    """
    Get content between <TEXT> and </TEXT>
    """
    with open(filename) as f:
        doc = ""
        addFlag = False
        for line in f:
            if line[:6] == '<TEXT>':
                addFlag = True
                continue
            elif line[:7] == '</TEXT>':
                addFlag = False
            if addFlag:
                doc += line[:-1]
    return doc


def getAllFileContext(fileList) -> str:
    allFileContextList = []
    for fileName in fileList:
        allFileContextList.append(getFileText(fileName))
    return allFileContextList


def pretreatment(docSentences):
    """
    处理一个10个文档中的一个文档，将所有句子预处理
    :param docSentences:
    :return:
    """
    punctuation_map = dict((ord(char), None) for char in string.punctuation)  # 引入标点符号，为下步去除标点做准备  #所有的标点字符
    s = nltk.stem.SnowballStemmer('english')  # 在提取词干时,语言使用英语,使用的语言是英语
    sentences = docSentences.split('.')
    # print("pretreatment")
    # print(len(sentences))
    newSentence = ''
    for i in range(len(sentences) - 1):
        cleanWord = ''
        lowerSentence = sentences[i].lower()
        withoutPunctuation = lowerSentence.translate(punctuation_map)
        tokens = nltk.word_tokenize(withoutPunctuation)  # 将文章进行分词处理,将一段话转变成一个list，将标点符号转化为none即空
        withoutStopwords = [w for w in tokens if not w in stopwords.words('english')]  # 去除文章的停用词
        for j in range(len(withoutStopwords)):
            cleanWord = cleanWord + s.stem(withoutStopwords[j]) + ' '  # 提取词干 将进行时过去式等还成原来的状态
        newSentence = newSentence + cleanWord + '.'
    return newSentence


def pretreatmentForDocHits(longSentenceForEveryDoc):
    punctuation_map = dict((ord(char), None) for char in string.punctuation)  # 引入标点符号，为下步去除标点做准备  #所有的标点字符
    s = nltk.stem.SnowballStemmer('english')  # 在提取词干时,语言使用英语,使用的语言是英语
    # 装入清洗好的十个句子，即10个文档句子
    newSentences = []
    for everyDocSentence in longSentenceForEveryDoc:
        cleanWord = ''
        lowerSentence = everyDocSentence.lower()
        withoutPunctuation = lowerSentence.translate(punctuation_map)  #
        tokens = nltk.word_tokenize(withoutPunctuation)  # 将文章进行分词处理,将一段话转变成一个list，将标点符号转化为none即空
        withoutStopwords = [w for w in tokens if not w in stopwords.words('english')]  # 去除文章的停用词
        for j in range(len(withoutStopwords)):
            cleanWord = cleanWord + s.stem(withoutStopwords[j]) + ' '  # 提取词干 将进行时过去式等还成原来的状态
        newSentences.append(cleanWord)
    return newSentences


def getColumns(docList):
    """
    十个文档中所有的词
    :param docList:
    :return:
    """
    columns = set()
    for doc in docList:
        sentences = doc.split('.')
        for sentence in sentences:
            words = sentence.split(' ')
            columns = columns | set(words)
    return columns


def getColumnsForDocHITS(longSentenceForEveryDocClean):
    columns = set()
    for sentence in longSentenceForEveryDocClean:
        words = sentence.split(' ')
        columns = columns | set(words)
    return columns


def getAllSentenceNumber(docList):
    number = 0
    for doc in docList:
        sentences = doc.split('.')
        number = number + len(sentences)
    return number


def getLongSentence(docList):
    longSentence = ''
    for doc in docList:
        sentences = doc.split('.')
        for sentence in sentences:
            longSentence = longSentence + sentence + ' '
    return longSentence


def getDocSentence(docList):
    longSentence = []
    for doc in docList:
        docSentence = ''
        sentences = doc.split('.')
        for sentence in sentences:
            docSentence = docSentence + sentence + ''
        docSentence = docSentence + '.'
        longSentence.append(docSentence)
    return longSentence


def getShortSentence(docList):
    sentenceList = []
    for doc in docList:
        sentences = doc.split('.')
        # print("getShortSentence")
        # print(len(sentences))
        for i in range(len(sentences) - 1):
            sentenceList.append(sentences[i])
        # print("zongchangdu")
        # print(len(sentenceList))
    return sentenceList


def getFirstSentence(docList):
    summary = ''
    for doc in docList:
        sentences = doc.split('.')
        summary = summary + sentences[0] + '.'
    return summary


def getCounter(sentenceList):
    """
    所有的句子（包括长句子，再最后）统计单词个数）
    :param sentenceList:
    :return:
    """
    countSentenceList = []
    for sentence in sentenceList:
        words = sentence.split(' ')
        count = Counter(words)
        countSentenceList.append(count)
    return countSentenceList


def calculateWordSentence(word, countSentenceList):
    """
    计算ISF中包含该词的句子数
    :param word:
    :param countSentenceList:
    :return:
    """
    number = 0
    for countSentence in countSentenceList:
        if word in countSentence:
            number = number + 1
    return number


def calculateTF(word, countSentence):
    return countSentence[word] / sum(countSentence.values())


def calculateISF(word, countSentenceList):
    return math.log(len(countSentenceList) / (1 + calculateWordSentence(word, countSentenceList)))


def calculateTFISF(word, countSentence, countSentenceList):
    return calculateTF(word, countSentence) * calculateISF(word, countSentenceList)


def calculateSimilar(sentenceList, countSentenceList, columns):
    TFISFZeros = np.zeros(((len(sentenceList)), (len(columns))), dtype=float)
    TFISFDataFrame = pd.DataFrame(TFISFZeros, columns=list(columns))
    for i in range(len(countSentenceList)):
        for word in countSentenceList[i]:
            if word in columns:
                TFISFDataFrame.loc[i, word] = calculateTFISF(word, countSentenceList[i], countSentenceList)
    TFISFMatrix = np.array(TFISFDataFrame)
    similarMatrix = np.dot(TFISFMatrix, TFISFMatrix.transpose())
    innerMatrix = np.sum(np.multiply(TFISFMatrix, TFISFMatrix), axis=1, keepdims=True)
    innerMatrix = np.sqrt(np.dot(innerMatrix, innerMatrix.transpose()))
    similarMatrix = np.divide(similarMatrix, innerMatrix)
    return similarMatrix


def norm(matrix):
    return matrix / np.linalg.norm(matrix)


def calculateSimilarForDocHITS(sentenceList, countSentenceList, columns, longSentenceForEveryDocClean,
                               countSentenceListForEveryDoc, columnsForDocHits, shape_A, shape_H):
    TFISFZerosDoc = np.zeros(((len(longSentenceForEveryDocClean)), (len(columnsForDocHits))), dtype=float)
    TFISFDataFrameDoc = pd.DataFrame(TFISFZerosDoc, columns=list(columnsForDocHits))
    for i in range(len(countSentenceListForEveryDoc)):
        for word in countSentenceListForEveryDoc[i]:
            if word in columnsForDocHits:
                TFISFDataFrameDoc.loc[i, word] = calculateTFISF(word, countSentenceList[i], countSentenceList)

    TFISFZeros = np.zeros(((len(sentenceList)), (len(columns))), dtype=float)
    TFISFDataFrame = pd.DataFrame(TFISFZeros, columns=list(columns))
    for i in range(len(countSentenceList)):
        for word in countSentenceList[i]:
            if word in columns:
                TFISFDataFrame.loc[i, word] = calculateTFISF(word, countSentenceList[i], countSentenceList)
    TFISFMatrixDoc = np.array(TFISFDataFrameDoc)
    TFISFMatrix = np.array(TFISFDataFrame)
    similar = np.dot(TFISFMatrix, TFISFMatrixDoc.transpose())
    innerSen = np.sum(np.multiply(TFISFMatrix, TFISFMatrix), axis=1, keepdims=True)
    innerDoc = np.sum(np.multiply(TFISFMatrixDoc, TFISFMatrixDoc), axis=1, keepdims=True)
    inner = np.sqrt(np.dot(innerSen, innerDoc.transpose()))
    similarMatrix = np.divide(similar, inner)
    epoch = 1
    last_A = A = np.ones((shape_A, 1))
    last_H = H = np.ones((shape_H, 1))
    while True:
        # 求模
        A = norm(np.dot(similarMatrix, last_H))
        H = norm(np.dot(similarMatrix.transpose(), last_A))
        loss_A = np.sum(last_A - A)
        loss_H = np.sum(last_H - H)
        if max(loss_A, loss_H) < 0.00001:
            return A.transpose()

        epoch += 1
        last_A = A
        last_H = H


def controlRedundancy(summaryIndex, thisIndex, similarMatrix, threshold):
    for index in summaryIndex:
        if similarMatrix[index][thisIndex] > threshold:
            return False
    return True


def powerMethod(cosineMatrix, length, q):
    pBefore = np.ones(length, dtype=float)
    for i in range(length):
        pBefore[i] = 1 / length
    pAfter = (cosineMatrix.T).dot(pBefore)
    e = np.linalg.norm(pAfter - pBefore)
    while e > q:
        pBefore = pAfter
        pAfter = (cosineMatrix.T).dot(pBefore)
        e = np.linalg.norm(pAfter - pBefore)
    return pAfter


def lexRank(similarMatrix, cosineThreshold):
    length = np.size(similarMatrix, 0)
    cosineMatrix = np.zeros((length, length), dtype=float)
    degree = [0 for _ in range(length)]
    for i in range(length):
        for j in range(length):
            if similarMatrix[i][j] > cosineThreshold:
                cosineMatrix[i][j] = 1
                degree[i] = degree[i] + 1
            else:
                cosineMatrix[i][j] = 0
    for i in range(length):
        for j in range(length):
            cosineMatrix[i][j] = cosineMatrix[i][j] / degree[i]
    p = powerMethod(cosineMatrix, length, 0.0001)
    return p


def sentenceSortByCosine(similarMatrix, sentenceList, originalSentences, threshold):
    lastRow = similarMatrix[(len(sentenceList) - 1)]
    index = []
    for i in range(len(sentenceList)):
        index.append(i)
    lastRowAndIndex = [index, lastRow]
    column = ['index', 'sentence']
    lastRowDataFrame = pd.DataFrame(np.mat(lastRowAndIndex).transpose(), columns=column)
    # 从高到底进行排序，与长句子的对比度
    sortedSentence = lastRowDataFrame.sort_values(by=column[1], ascending=False)
    summary = ''
    summaryIndex = []
    sortedSentenceMatrix = np.array(sortedSentence)
    # while len(summary) < 665:
    for rowIndex in range(len(sentenceList) - 1):
        number = sortedSentenceMatrix[rowIndex + 1][0]
        # number 句子按相似度从大到小的排序的下标
        number = int(number)
        if controlRedundancy(summaryIndex, number, similarMatrix, threshold):
            summaryIndex.append(int(number))
            summary = summary + ' ' + originalSentences[number]
            summary.strip()
    # 获得原始句子列表，注意下标不一样，加入summary 注意while循环
    return summary[0:665]


def sentenceSortByLexRank(p, sentenceList, originalSentences, threshold, similarMatrix):
    lastRow = p
    index = []
    for i in range(len(sentenceList)):
        index.append(i)
    lastRowAndIndex = [index, lastRow]
    column = ['index', 'sentence']
    lastRowDataFrame = pd.DataFrame(np.mat(lastRowAndIndex).transpose(), columns=column)
    # 从高到底进行排序，与长句子的对比度
    sortedSentence = lastRowDataFrame.sort_values(by=column[1], ascending=False)
    summary = ''
    summaryIndex = []
    sortedSentenceMatrix = np.array(sortedSentence)
    # while len(summary) < 665:
    for rowIndex in range(len(sentenceList) - 1):
        number = sortedSentenceMatrix[rowIndex + 1][0]
        # number 句子按相似度从大到小的排序的下标
        number = int(number)
        if controlRedundancy(summaryIndex, number, similarMatrix, threshold):
            summaryIndex.append(int(number))
            summary = summary + ' ' + originalSentences[number]
            summary.strip()
    # 获得原始句子列表，注意下标不一样，加入summary 注意while循环
    return summary[0:665]


def sentenceSortByDocHits(vecA, sentenceList, originalSentences, threshold, similarMatrix):
    lastRow = vecA
    index = []
    for i in range(len(sentenceList)):
        index.append(i)
    lastRowAndIndex = [index, lastRow]
    column = ['index', 'sentence']
    lastRowDataFrame = pd.DataFrame(np.mat(lastRowAndIndex).transpose(), columns=column)
    # 从高到底进行排序，与长句子的对比度
    sortedSentence = lastRowDataFrame.sort_values(by=column[1], ascending=False)
    summary = ''
    summaryIndex = []
    sortedSentenceMatrix = np.array(sortedSentence)
    # while len(summary) < 665:
    for rowIndex in range(len(sentenceList)):
        number = sortedSentenceMatrix[rowIndex][0]
        # number 句子按相似度从大到小的排序的下标
        number = int(number)
        if controlRedundancy(summaryIndex, number, similarMatrix, threshold):
            summaryIndex.append(int(number))
            summary = summary + ' ' + originalSentences[number]
            summary.strip()
    # 获得原始句子列表，注意下标不一样，加入summary 注意while循环
    return summary[0:665]


def getSummaryByCosine(fileList):
    docList = getAllFileContext(fileList)
    # 原始句子(正确分割)
    originalSentences = getShortSentence(docList)
    # print("originalSentences")
    # print(len(originalSentences))
    # 下面的docList是一个String类型的数组，每个元素是一个文档预处理完的所有句子
    for i in range(len(docList)):
        docList[i] = pretreatment(docList[i])
    # 获得10个文章中的所有单词（处理过的）
    columns = getColumns(docList)
    # 获得所有句子的个数
    getAllSentenceNumber(docList)

    # 所有的句子
    sentenceList = getShortSentence(docList)
    # print("sentenceList")
    # print(len(sentenceList))
    sentenceList.append(getLongSentence(docList))

    # 所有处理好的句子，然后统计了单词个数
    countSentenceList = getCounter(sentenceList)
    # 相似矩阵
    similarMatrix = calculateSimilar(sentenceList, countSentenceList, columns)
    summary = sentenceSortByCosine(similarMatrix, sentenceList, originalSentences, 0.7)
    return summary


def getSummaryByBaseLine(fileList):
    docList = getAllFileContext(fileList)
    summary = getFirstSentence(docList)
    return summary


def getSummaryByLexRank(fileList):
    docList = getAllFileContext(fileList)
    # 原始句子(正确分割)
    originalSentences = getShortSentence(docList)
    # print("originalSentences")
    # print(len(originalSentences))
    # 下面的docList是一个String类型的数组，每个元素是一个文档预处理完的所有句子
    for i in range(len(docList)):
        docList[i] = pretreatment(docList[i])
    # 获得10个文章中的所有单词（处理过的）
    columns = getColumns(docList)
    # 获得所有句子的个数
    getAllSentenceNumber(docList)

    # 所有的句子
    sentenceList = getShortSentence(docList)
    # print("sentenceList")
    # print(len(sentenceList))
    sentenceList.append(getLongSentence(docList))

    # 所有处理好的句子，然后统计了单词个数
    countSentenceList = getCounter(sentenceList)
    # 相似矩阵
    similarMatrix = calculateSimilar(sentenceList, countSentenceList, columns)
    p = lexRank(similarMatrix, 0.2)
    summary = sentenceSortByLexRank(p, sentenceList, originalSentences, 0.7, similarMatrix)
    return summary


def getSummaryByDocHITS(fileList):
    docList = getAllFileContext(fileList)
    # 长句子用的
    longSentenceForEveryDoc = getDocSentence(docList)
    longSentenceForEveryDocClean = pretreatmentForDocHits(longSentenceForEveryDoc)
    countSentenceListForEveryDoc = getCounter(longSentenceForEveryDocClean)
    columnsForDocHits = getColumnsForDocHITS(longSentenceForEveryDocClean)

    # print(similarForDoc)
    # print(longSentenceForEveryDoc)
    # 原始句子(正确分割)
    originalSentences = getShortSentence(docList)

    # 下面的docList是一个String类型的数组，每个元素是一个文档预处理完的所有句子
    for i in range(len(docList)):
        docList[i] = pretreatment(docList[i])
    # 短句子用的
    # 获得10个文章中的所有单词（处理过的）
    columns = getColumns(docList)
    # 获得所有句子的个数
    getAllSentenceNumber(docList)
    # 所有的句子
    sentenceList = getShortSentence(docList)
    # print("sentenceList")
    # print(len(sentenceList))
    # sentenceList.append(getLongSentence(docList))

    # 所有处理好的句子，然后统计了单词个数
    countSentenceList = getCounter(sentenceList)
    similarMatrixForSentence = calculateSimilar(sentenceList, countSentenceList, columns)
    vecA = calculateSimilarForDocHITS(sentenceList, countSentenceList, columns, longSentenceForEveryDocClean,
                                      countSentenceListForEveryDoc, columnsForDocHits, len(sentenceList),
                                      len(longSentenceForEveryDocClean))

    summary = sentenceSortByDocHits(vecA[0], sentenceList, originalSentences, 0.7, similarMatrixForSentence)
    return summary


def getFileNames(path):
    for root, dirs, files in os.walk(path):
        return files


if __name__ == '__main__':
    PATHs = 'D:/学习/数据挖掘/理论/dataset/DUC04/unpreprocess data/docs/d30001t'
    fileL = getFileNames(PATHs)
    # 将一整个topic的十个文件加载到fileList里面
    fileLists = [PATHs + '/' + f for f in fileL]

    PATHForManualSummary = 'D:/学习/数据挖掘/理论/dataset/04model/'
    fileLForManualSummary = ['D30001.M.100.T.A',
                             'D30001.M.100.T.B',
                             'D30001.M.100.T.C',
                             'D30001.M.100.T.D']
    fileListForManualSummary = [PATHForManualSummary + f for f in fileLForManualSummary]
    allFileContextList = []
    for fileName in fileListForManualSummary:
        with open(fileName, 'r') as myfile:
            data = myfile.read()
            allFileContextList.append(data)

    # summaryByCosine = getSummaryByCosine(fileLists)
    # print("summaryByCosine:")
    # print(summaryByCosine)

    # summaryByBaseLine = getSummaryByBaseLine(fileLists)
    # print("summaryByBaseLine:")
    # print(summaryByBaseLine)

    # summaryByLexRank = getSummaryByLexRank(fileLists)
    # print("summaryByLexRank")
    # print(summaryByLexRank)

    summaryByDocHITS = getSummaryByDocHITS(fileLists)
    print("summaryByDocHITS:")
    print(summaryByDocHITS)

    f = 0
    p = 0
    r = 0

    rouge = Rouge()
    for allFileContext in allFileContextList:
        rouge_score = rouge.get_scores(summaryByDocHITS, allFileContext)
        value = rouge_score[0]["rouge-1"]
        f = f + value['f']
        p = p + value['p']
        r = r + value['r']
    print(f / len(allFileContextList))
    print(p / len(allFileContextList))
    print(r / len(allFileContextList))

    # print(rouge_score[0]["rouge-1"])
    # print(rouge_score[0]["rouge-2"])
    # print(rouge_score[0]["rouge-l"])
