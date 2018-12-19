#!/usr/bin/env python

# The project is a lab of Web Data Mining Course in UCAS.
# We aim to train a spam classifier for the given dataset with different models such as perceptron, SVM, etc.
# This file is the entry point of ensembling system including two major parts:
#     SpamClassifier: the back-end classifier to identify if the input is a spam
#     OfflineDemo: the front-end demonstration system to support an interactive UI
# Date: 11/16/2018
# Writen by: ssy

from SpamClassifier.SpamClassifier import SpamClassifier
from SpamClassifier.SpamClassifier import ClassifierAssistant

if __name__ == "__main__":
    # Train LR
    #SC = SpamClassifier(type=ClassifierAssistant.LR, existed=False, bfBucketNum = 10000)
    #SC.train()
    # Test LR
    #SC = SpamClassifier(type=ClassifierAssistant.LR, existed=True)
    #SC.test()

    # Train MP
    #SC = SpamClassifier(type=ClassifierAssistant.MP, existed=False, bfBucketNum = 1000)
    #SC.train()
    # Test LP
    #SC = SpamClassifier(type=ClassifierAssistant.MP, existed=True)
    #SC.test()

    # Train SVM
    #SC = SpamClassifier(type=ClassifierAssistant.SVM, existed=False, bfBucketNum = 10000)
    #SC.train()
    # Test SVM
    #SC = SpamClassifier(type=ClassifierAssistant.SVM, existed=True)
    #SC.test()

    # Train DT
    #SC = SpamClassifier(type=ClassifierAssistant.DT, existed=False, bfBucketNum = 10000)
    #SC.train()
    # Test DT
    #SC = SpamClassifier(type=ClassifierAssistant.DT, existed=True)
    #SC.test()

    # Train GBDT
    #SC = SpamClassifier(type=ClassifierAssistant.GBDT, existed=False, bfBucketNum = 10000)
    #SC.train()
    # Test GBDT
    #SC = SpamClassifier(type=ClassifierAssistant.GBDT, existed=True)
    #SC.test()

    SCLR = SpamClassifier(type=ClassifierAssistant.LR)
    SCMP = SpamClassifier(type=ClassifierAssistant.MP)
    SCSVM = SpamClassifier(type=ClassifierAssistant.SVM)
    result = []
    result.append(SCLR.predict("以上比赛规则由江苏科技大学教职工摄影协会负责解释"))
    result.append(SCMP.predict("以上比赛规则由江苏科技大学教职工摄影协会负责解释"))
    result.append(SCSVM.predict("以上比赛规则由江苏科技大学教职工摄影协会负责解释"))
    print(result)
    #result.append(SC.predict("以上比赛规则由江苏科技大学教职工摄影协会负责解释脑残"))
    #result.append(SC.predict("以上比赛规则由江苏科技大学教职工摄影协会负责解释弱智"))
    print("result: {}".format(result))
