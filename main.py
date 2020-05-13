import math
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd


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


def pretreatment(sentences):
    """
    处理一个10个文档中的一个文档，将所有句子预处理
    :param sentences:
    :return:
    """
    punctuation_map = dict((ord(char), None) for char in string.punctuation)  # 引入标点符号，为下步去除标点做准备  #所有的标点字符
    s = nltk.stem.SnowballStemmer('english')  # 在提取词干时,语言使用英语,使用的语言是英语
    sentences = sentences.split('.')
    newSentence = ''
    for sentence in sentences:
        cleanWord = ''
        lowerSentence = sentence.lower()
        withoutPunctuation = lowerSentence.translate(punctuation_map)
        tokens = nltk.word_tokenize(withoutPunctuation)
        withoutStopwords = [w for w in tokens if not w in stopwords.words('english')]  # 去除文章的停用词
        for i in range(len(withoutStopwords)):
            cleanWord = cleanWord + s.stem(withoutStopwords[i]) + ' '  # 提取词干 将进行时过去式等还成原来的状态
        newSentence = newSentence + cleanWord + '.'
    return newSentence


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
            while '' in words:
                words.remove('')
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


def getShortSentence(docList):
    sentenceList = []
    for doc in docList:
        sentences = doc.split('.')
        for sentence in sentences:
            sentenceList.append(sentence)
    return sentenceList


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
                TFISFDataFrame.loc[i,word]=calculateTFISF(word, countSentenceList[i], countSentenceList)

    return TFISFDataFrame

def getSummary(fileList):
    docList = getAllFileContext(fileList)
    # 下面的docList是一个String类型的数组，每个元素是一个文档预处理完的所有句子
    for i in range(len(docList)):
        docList[i] = pretreatment(docList[i])
    # 获得10个文章中的所有单词（处理过的）
    columns = getColumns(docList)
    # 获得所有句子的个数
    getAllSentenceNumber(docList)

    # 所有的句子
    sentenceList = getShortSentence(docList)
    sentenceList.append(getLongSentence(docList))

    # 所有处理好的句子，然后统计了单词个数
    countSentenceList = getCounter(sentenceList)
    # for sentence in sentenceList:
    #     print(sentence)
    print(calculateSimilar(sentenceList,countSentenceList,columns))
    # for i in getCounter(sentenceList):
    #     print(i)


if __name__ == '__main__':
    PATH = 'D:/学习/数据挖掘/理论/dataset/DUC04/unpreprocess data/docs/'
    fileL = ['d30001t/APW19981016.0240', 'd30001t/APW19981022.0269', 'd30001t/APW19981026.0220',
             'd30001t/APW19981027.0491', 'd30001t/APW19981031.0167', 'd30001t/APW19981113.0251',
             'd30001t/APW19981116.0205', 'd30001t/APW19981118.0276', 'd30001t/APW19981120.0274',
             'd30001t/APW19981124.0267']

    # 将一整个topic的十个文件加载到fileList里面
    fileLists = [PATH + f for f in fileL]
    getSummary(fileLists)
    # allFileContextList = getAllFileContext(fileList)

    # for fileName in fileList:
    #     print(getFileText(fileName))
