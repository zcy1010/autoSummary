import string
import nltk
from nltk.corpus import stopwords


def getFileText(filename) -> str:
    """
    Get content bewteen <TEXT> and </TEXT>
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
            cleanWord = cleanWord + s.stem(withoutStopwords[i])+' '  # 提取词干 将进行时过去式等还成原来的状态
        newSentence = newSentence + cleanWord+'.'
    return newSentence


def getSummary(fileList):
    docList = getAllFileContext(fileList)
    for i in range(len(docList)):
        docList[i] = pretreatment(docList[i])
    print(docList)


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
