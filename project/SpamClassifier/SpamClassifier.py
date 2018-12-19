#!/usr/bin/env python

# This is the entry point of our back-end module.
# We implemented several classical classification models to capture spam mails.
# There're two subdirectories for persistence:
#     Models: exported models
#     Data: training dataset and test dataset
# Date: 11/17/2018
# Writen by: ssy

# TODO: windows error: (null) entry in command string: null chmod 0644 in windows when saving model

import os
import math
import time
import shutil
from pyspark.sql import Row
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType
from pyspark.sql.types import StringType
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.classification import MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel
from pyspark.ml.classification import LinearSVC, LinearSVCModel
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
import jieba

train_lock = True

conf = SparkConf().set("spark.cores.max", "16")\
    .set("spark.driver.memory", "16g")\
    .set("spark.executor.memory", '16g')
sc = SparkContext(appName="SpamClassifier", conf=conf)
sc.setLogLevel("ERROR")
spark = SparkSession(sc)

class ClassifierAssistant(object):
    """
        Classifier assistant aims to help users use Spam Classifier better.
        Examples:
            from SpamClassifier.SpamClassifier import ClassifierAssistant

            type=ClassifierAssistant.LR
    """

    # supported algorithms
    LR = 'LogisticRegression'
    MP = 'MultilayerPerceptron'
    SVM = 'SupportVectorMachine'
    DT = 'DecisionTree'
    GBDT = 'GradientBoostedTree'

class SpamClassifier(object):
    """
        It's a binary classifier for spam mails with an abbreviation as SC.
        SC supports two major APIs: train and predict.
        Examples:
            from SpamClassifier.SpamClassifier import SpamClassifier
            from SpamClassifier.SpamClassifier import ClassifierAssistant

            SC = SpamClassifier(foldNum = 5, type = ClassifierAssistant.LR, existed = False)
            SC.train()
            SC.predict(text)
    """

    def __init__(self, foldNum = 5, type = ClassifierAssistant.LR, existed = True, bfBucketNum = 1000):
        """
            Define some static variable for model training and prediction.
        """
        print("")
        print("Start to init SpamClassifier...")
        curdir = os.path.dirname(__file__)
        self.__modelRoot = os.path.join(curdir, "Models") # the directory of saved models
        self.__dataDir = os.path.join(curdir, "Data") # the directory of labeled and unlabel dataset

        # save in Data dir
        self.__rawLabeledName = "labeled.txt" # raw labeled dataset(label + text)
        self.__tokenizedLabeledName = self.__rawLabeledName + ".tokenized" + ".{}fold".format(foldNum) # tokenized labeled dataset(words)
        self.__rawUnlabelName = "unlabel.txt" # raw unlabel dataset


        self.__sc = sc
        self.__spark = spark 

        if foldNum <= 1:
            raise RuntimeError("Given foldNum: {}, foldNum must be larger than 1!".format(foldNum))
        self.__type = type
        self.__foldNum = foldNum  # fold num of K-fold cross validation

        self.__existed = True
        if not train_lock:
            self.__existed = existed

        # for train
        self.__hasTrained = False
        self.__bloomFilterBucketNum = bfBucketNum
        self.__tokenziedDfList = [None]*foldNum # random split parts of tokenized labeled data (length: foldNum)
        self.__trainFeatureDfList = [None]*foldNum

        # for load
        self.__hasLoad = False

        # for test or predict
        self.__testCount = 100
        self.__tfModelList = [None] * foldNum
        self.__idfModelList = [None] * foldNum
        self.__classifierList = [None] * foldNum

        if existed:
            self.__loadAll() # it will set __hasLoad = True

        print("Init success!")

    ####### Predict Module #######

    def predict(self, text):
        if not self.__existed:
            if not self.__hasTrained:
                raise RuntimeError("Existed is false, you must train a model before prediction!")
        else:
            if not self.__hasLoad:
                raise RuntimeError("Error: something wrong has happend about model loading!")

        # tokenize input text with jieba
        words = SpamClassifier.cnTokenizer(text) # text -> words
        wordsDf = self.__spark.createDataFrame([[words]], ['words'])

        positiveNum = 0
        for i in range(self.__foldNum):
            # Extract TF features, calculate tf feature vector for each text
            tfFeatureDf = self.__tfModelList[i].transform(wordsDf)

            # Extract IDF features
            tfIdfFeatureDf = self.__idfModelList[i].transform(tfFeatureDf)

            # Predict
            prediction = self.__classifierList[i].transform(tfIdfFeatureDf).head()['prediction']
            if prediction == 1:
                positiveNum += 1

        # voting
        if positiveNum > math.floor(self.__foldNum / 2):
            return 1
        else:
            return 0

    ####### Test Module #######

    def test(self):
        if not self.__existed:
            if not self.__hasTrained:
                raise RuntimeError("Existed is false, you must train a model before test it!")
        else:
            if not self.__hasLoad:
                raise RuntimeError("Error: something wrong has happend about model loading!")

        # compute precision, recall and F1 for foldNUm models with the given algorithm
        print("")
        print("Start to test!")
        print("Compute precision, recall and F1 first...")
        precisionList = []
        recallList = []
        F1List = []
        for i in range(self.__foldNum):
            testDf = self.__tokenziedDfList[i]
            # Extract TF features, calculate tf feature vector for each text
            tfFeatureDf = self.__tfModelList[i].transform(testDf)

            # Extract IDF features
            tfIdfFeatureDf = self.__idfModelList[i].transform(tfFeatureDf)

            # Predict
            resultDf = self.__classifierList[i].transform(tfIdfFeatureDf)
            trueDf = resultDf.filter(resultDf['label'] == resultDf['prediction'])
            falseDf = resultDf.filter(resultDf['label'] != resultDf['prediction'])

            truePositive = trueDf.filter(trueDf['prediction'] == 1).count()
            falsePositive = falseDf.filter(falseDf['prediction'] == 1).count()
            falseNegative = falseDf.count() - falsePositive

            if (truePositive+falsePositive) == 0:
                precision = 1
            else:
                precision = float(truePositive)/(truePositive + falsePositive)

            precisionList.append(precision)

            if (truePositive + falseNegative) == 0:
                recall = 1
            else:
                recall = float(truePositive)/(truePositive + falseNegative)
            recallList.append(recall)

            if (precision + recall) == 0:
                F1 = 1
            else:
                F1 = 2*precision*recall/(precision+recall)
            F1List.append(F1)

        # show computation results
        averagePrecision = sum(precisionList)/self.__foldNum
        print("The precision list for each model: {}\tAverage precision: {}".format(precisionList, averagePrecision))
        averageRecall = sum(recallList)/self.__foldNum
        print("The recall list for each model: {}\tAverage recall: {}".format(recallList, averageRecall))
        averageF1 = sum(F1List)/self.__foldNum
        print("The F1 list for each model: {}\tAverage F1: {}".format(F1List, averageF1))

        # compute predict
        print("Computing average prediction time...")
        predictTime = self.__computePredictTime()
        print("The average prediction time is: {} s".format(predictTime))

    def __computePredictTime(self):
        print("Predicting all of the unlabel dataset to get prediction time...")
        datasetPath = os.path.join(self.__dataDir, self.__rawUnlabelName)
        file = open(datasetPath)

        startTime = time.time()
        count = 0
        for line in file:
            self.predict(line)
            count += 1
            if count == self.__testCount:
                break
        endTime = time.time()

        file.close()
        print("Test count: {}".format(count))
        return (endTime - startTime) / count

    ####### Load Module #######

    def __loadAll(self):
        if not self.__existed:
            raise RuntimeError("Existed is false, can't load existed models!")
        else:
            if self.__hasLoad:
                raise RuntimeError("HasLoad is true, can't load twice within a single SpamClassifier!")

        self.__loadSplitTokenizedLabeledDataset() # load split tokenized labeled dataset

        print("Load tf model, idf model for each fold...")
        for i in range(self.__foldNum):
            self.__loadTfModel(i) # load tf model
            self.__loadIdfModel(i) # load idf model
            self.__loadClassifier(i) # load classifier
        print("Load success!")

        self.__hasLoad = True

    def __loadTfModel(self, index):
        modelDir = os.path.join(self.__modelRoot, self.__type)
        tfModelPath = os.path.join(modelDir, "{}.tfmodel").format(index)
        if not os.path.exists(tfModelPath):
            raise RuntimeError("No such directory: {}!".format(tfModelPath))
        self.__tfModelList[index] = HashingTF.load(tfModelPath)

    def __loadIdfModel(self, index):
        modelDir = os.path.join(self.__modelRoot, self.__type)
        idfModelPath = os.path.join(modelDir, "{}.idfmodel").format(index)
        if not os.path.exists(idfModelPath):
            raise RuntimeError("No such directory: {}!".format(idfModelPath))
        self.__idfModelList[index] = IDFModel.load(idfModelPath)

    def __loadClassifier(self, index):
        modelDir = os.path.join(self.__modelRoot, self.__type)
        classifierPath = os.path.join(modelDir, "{}.classifier").format(index)
        if not os.path.exists(classifierPath):
            raise RuntimeError("No such directory: {}!".format(classifierPath))
        if self.__type == ClassifierAssistant.LR: # Logistic Regression
            self.__classifierList[index] = LogisticRegressionModel.load(classifierPath)
        elif self.__type == ClassifierAssistant.MP:
            self.__classifierList[index] = MultilayerPerceptronClassificationModel.load(classifierPath)
        elif self.__type == ClassifierAssistant.SVM:
            self.__classifierList[index] = LinearSVCModel.load(classifierPath)
        elif self.__type == ClassifierAssistant.DT:
            self.__classifierList[index] = DecisionTreeClassificationModel.load(classifierPath)
        elif self.__type == ClassifierAssistant.GBDT:
            self.__classifierList[index] = GBTClassificationModel.load(classifierPath)


    def __loadSplitTokenizedLabeledDataset(self):
        """
            Load labeled datasdet and split it into foldNum parts for K-fold cross validation.
        :param labeledDatasetName: name of labeled dataset file
        :return: None
        """
        # check if tokenize result exists
        splitTokenizeResultPath = os.path.join(self.__dataDir, self.__tokenizedLabeledName)
        hasSplitTokenizeResult = os.path.exists(splitTokenizeResultPath)

        if not hasSplitTokenizeResult:
            # read file and create Dataframe
            print("Has not tokenized!")
            print("\tLoading raw labeled dataset...")
            datasetPath = os.path.join(self.__dataDir, self.__rawLabeledName)
            # default partition num is 2 which causes partial file too large to upload to github
            rawRDD = self.__sc.textFile(datasetPath, minPartitions=4) # RDD<String(line)>
            dataRDD = rawRDD.map(SpamClassifier.labeledLineSplit) # RDD<Pair(label, text)>

            # tokenize labeled dataset
            print("\tTokenize raw labeled dataset into words...")
            dataDf = self.__spark.createDataFrame(dataRDD, ['label', 'text'])
            tokenizeUdf = udf(SpamClassifier.cnTokenizer, ArrayType(StringType()))
            wordsDf = dataDf.withColumn("words", tokenizeUdf(dataDf['text'])).select('label', 'words')

            # split into several rdds
            print("\tSplit tokenized labeled dataset for {}-fold cross validation...".format(self.__foldNum))
            weightList = [1.0] * self.__foldNum
            self.__tokenziedDfList = wordsDf.randomSplit(weightList)

            # save split tokenize result (Row<label, words>)
            for i in range(self.__foldNum):
                print("\tSave the {}th split tokenized labeled data...".format(i))
                tempPath = os.path.join(splitTokenizeResultPath, "{}".format(i))
                self.__tokenziedDfList[i].write.json(tempPath)
        else:
            # Load tokenized labeled dataset
            print("Load split tokenized labeled dataset...")
            for i in range(self.__foldNum):
                tempPath = os.path.join(splitTokenizeResultPath, "{}".format(i))
                self.__tokenziedDfList[i] = self.__spark.read.json(tempPath)

    @staticmethod
    def __jsonToVector(row):
        return Row(label=row['label'], features=Vectors.sparse(row['features'].size,row['features'].indices,row['features'].values))

    ####### Train Module #######

    def train(self):
        """
            Train model with the specified algorithm according to the given type.
            Export the trained model into 'Models' directory.

            Here is our workflow:
                check old version and delete it
                load labeled dataset
                preprocess dataset to get feature vectors
                train and save model

            We save model into Models/type/[0-4] as a convention.
            Here is an example name of an exported model with LogisticRegression:
                Models/LR/0.classifier, Models/LR/1.classifier, ..., Models/LR/4.classifier
            There're 5 submodels after training since we use 5-fold cross validation by default.
        :param type: specified algorithm which is defined by ClassifierAssistant
        :return: None
        """
        if self.__existed:
            raise RuntimeError("Existed is true, can't train new models!")
        else:
            if self.__hasTrained:
                raise RuntimeError("HasTrained is true, can't train twice within a single SpamClassifier!")

        modelDir = os.path.join(self.__modelRoot, self.__type)

        # check old version
        print("")
        print("Checking old version first...")
        if os.path.exists(modelDir):
            print("\tFind old version, removing...")
            shutil.rmtree(modelDir) # if it exists, just remove

        # load split tokenized labeled dataset for K-fold cross validation
        self.__loadSplitTokenizedLabeledDataset()

        # prepare for training
        # convert RDD<Pair(label, text)> into RDD<tf-idf feature vector>
        self.__prepare()

        # model training
        print("")
        print("Start to train {} classifiers with {} algorithm".format(self.__foldNum, self.__type))
        if self.__type == ClassifierAssistant.LR: # Logistic Regression
            self.__trainLR()
        elif self.__type == ClassifierAssistant.MP: # Multilayer perceptron
            self.__trainMP()
        elif self.__type == ClassifierAssistant.SVM: # Support Vector Machine
            self.__trainSVM()
        elif self.__type == ClassifierAssistant.DT: # Decision Tree
            self.__trainDT()
        elif self.__type == ClassifierAssistant.GBDT: # Gradient-boosted Decision Tree
            self.__trainGBDT()

        print("Train Success!")
        self.__hasTrained = True

    def __trainLR(self):
        modelDir = os.path.join(self.__modelRoot, self.__type)
        LREstimator = LogisticRegression(regParam=0.2)
        # create foldNum models
        for i in range(self.__foldNum):
            # models[i] indicates: use rddList[i] as test dataset, and treat others as training dataset
            print("The {}th classifier is training...".format(i))
            self.__classifierList[i] = LREstimator.fit(self.__trainFeatureDfList[i])
            print("Save the {}th classifier...".format(i))
            self.__classifierList[i].save(os.path.join(modelDir,"{}.classifier".format(i)))

    def __trainMP(self):
        modelDir = os.path.join(self.__modelRoot, self.__type)
        layers = [self.__bloomFilterBucketNum, 4, 3, 2]
        MPEstimator = MultilayerPerceptronClassifier(layers=layers)
        # create foldNum models
        for i in range(self.__foldNum):
            # models[i] indicates: use rddList[i] as test dataset, and treat others as training dataset
            print("The {}th classifier is training...".format(i))
            self.__classifierList[i] = MPEstimator.fit(self.__trainFeatureDfList[i])
            print("Save the {}th classifier...".format(i))
            self.__classifierList[i].save(os.path.join(modelDir, "{}.classifier".format(i)))

    def __trainSVM(self):
        modelDir = os.path.join(self.__modelRoot, self.__type)
        SVMEstimator = LinearSVC(regParam=0.1)
        # create foldNum models
        for i in range(self.__foldNum):
            # models[i] indicates: use rddList[i] as test dataset, and treat others as training dataset
            print("The {}th classifier is training...".format(i))
            self.__classifierList[i] = SVMEstimator.fit(self.__trainFeatureDfList[i])
            print("Save the {}th classifier...".format(i))
            self.__classifierList[i].save(os.path.join(modelDir, "{}.classifier".format(i)))

    def __trainDT(self):
        modelDir = os.path.join(self.__modelRoot, self.__type)

        DTEstimator = DecisionTreeClassifier()
        # create foldNum models
        for i in range(self.__foldNum):
            # models[i] indicates: use rddList[i] as test dataset, and treat others as training dataset
            print("The {}th classifier is training...".format(i))
            self.__classifierList[i] = DTEstimator.fit(self.__trainFeatureDfList[i])
            print("Save the {}th classifier...".format(i))
            self.__classifierList[i].save(os.path.join(modelDir, "{}.classifier".format(i)))

    def __trainGBDT(self):
        modelDir = os.path.join(self.__modelRoot, self.__type)

        GBDTEstimator = GBTClassifier()
        # create foldNum models
        for i in range(self.__foldNum):
            # models[i] indicates: use rddList[i] as test dataset, and treat others as training dataset
            print("The {}th classifier is training...".format(i))
            self.__classifierList[i] = GBDTEstimator.fit(self.__trainFeatureDfList[i])
            print("Save the {}th classifier...".format(i))
            self.__classifierList[i].save(os.path.join(modelDir, "{}.classifier".format(i)))

    def __prepare(self):
        modelDir = os.path.join(self.__modelRoot, self.__type)

        print("Prepare for training...")
        for i in range(self.__foldNum):
            print("Prepare for the {}th model...".format(i))

            # construct training dataset and test dataset
            print("\tConstruct training dataset and test dataset...")
            trainingDf = None
            for j in range(self.__foldNum):
                if j == i:
                    continue
                if not trainingDf:
                    trainingDf = self.__tokenziedDfList[j]
                else:
                    trainingDf = trainingDf.union(self.__tokenziedDfList[j])

            # create and save hashingTransformer, we can set numFeatures to modify bloom filter bucket num
            print("\tCreate hashing transformer (tf model)...")
            self.__tfModelList[i] = HashingTF(inputCol="words", outputCol="tfFeatures", numFeatures=self.__bloomFilterBucketNum)

            # preprocess training dataset
            print("\tPreprocess training dataset...")
            self.__idfModelList[i], self.__trainFeatureDfList[i] = self.__preprocessTrainset(trainingDf, self.__tfModelList[i])

            # save tfModel, idfModel
            print("\tSave tf model and idf model...")
            self.__tfModelList[i].save(os.path.join(modelDir, "{}.tfmodel").format(i))
            self.__idfModelList[i].save(os.path.join(modelDir, "{}.idfmodel".format(i)))

        print("All idf models and tf-idf feature vectors for either train or test are done!")

    def __preprocessTrainset(self, trainingDf, tfModel):
        """
            Preprocess the given training dataset from (lable, text) into tf-idf feature vector.
        :param trainingRDD: the training dataset RDD including (foldNum - 1) parts from rddList
        :return: idfModel: the idfModel trained from idf estimator based on given training dataset
        :return: tfIdfFeatureDf: the extracted tf-idf feature vector from training dataset
        """
        # Extract TF features, calculate tf feature vector for each text
        tfFeatureDf = tfModel.transform(trainingDf).select('label', 'tfFeatures')

        # Extract IDF features
        idfEstimator = IDF(inputCol="tfFeatures", outputCol="features")  # create idf estimator
        idfModel = idfEstimator.fit(tfFeatureDf)  # train the idf model (calculate df for each word)
        # calculate tf-idf feature vector for each text
        tfIdfFeatureDf = idfModel.transform(tfFeatureDf).select('label', 'features')

        return (idfModel, tfIdfFeatureDf)

    # def __preprocessTestset(self, testRDD, tfModel, idfModel):
    #     """
    #         Preprocess the given test dataset from (lable, text) into tf-idf feature vector.
    #     :param testRDD: the test dataset RDD including only 1 part from rddList
    #     :param idfModel: the corresponding idf model for given test dataset
    #     :return: tfIdfFeatureDf: the extracted tf-idf feature vector from test dataset
    #     """
    #     testDf = self.__spark.createDataFrame(testRDD, ['label', 'words'])
    #
    #     # Extract TF features, calculate tf feature vector for each text
    #     tfFeatureDf = tfModel.transform(testDf).select('label', 'tfFeatures')
    #
    #     # Extract IDF features
    #     # calculate tf-idf feature vector for each text
    #     tfIdfFeatureDf = idfModel.transform(tfFeatureDf).select('label', 'features')
    #
    #     return tfIdfFeatureDf

    @staticmethod
    def labeledLineSplit(line):
        """
            Split line into label and text.
        :param line: a string line with the fixed form: label \t text (no blank near \t)
        :return: (label, text) as a pair
        """
        pos = line.index('\t')
        label = line[0:pos]
        text = line[pos+1:]
        return (float(label), text)

    @staticmethod
    def cnTokenizer(text):
        """
            Tokenize chinese text with a python lib jieba.
        :param text: chinese text without sep
        :return: words: a list of tokenized words
        """
        tokenizer = jieba.cut(text)
        words = [word for word in tokenizer]
        return words
