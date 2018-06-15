# -*- coding: utf-8 -*-

from tagger import Tagger
from tagger_lr_chunk import TaggerLR_Chunk
import os
import nltk
import seaborn
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV

class TaggerLR_NER(Tagger):

    def __init__(self):
        Tagger.__init__(self)
        self.chunk = []
        self.upos = []
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('classifier', LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, \
                solver='lbfgs', tol=0.0001, max_iter=100, class_weight=None, n_jobs=1, verbose=0, refit=True, intercept_scaling=1.0, \
                multi_class='ovr', random_state=None))
        ])

    def create_features(self, word, index, idx):
        return {
            'word': word,
            'is_capitalized': word[0].upper() == word[0],
            'is_all_caps': word.upper() == word,
            'is_all_lower': word.lower() == word,
            'prefix-1': word[0],
            'prefix-2': word[:2],
            'prefix-3': word[:3],
            'suffix-1': word[-1],
            'suffix-2': word[-2:],
            'suffix-3': word[-3:],
            'is_numeric': word.isdigit(),
            'capitals_inside': word[1:].lower() != word[1:],
            'upos' : self.upos[idx][index],
            'chunk' : self.chunk[idx][index]
        }

    def calculate_features(self):
        for idx,s in enumerate(self.sentences):
            for index,word in enumerate(s):
                self.features.append(self.create_features(word, index, idx))

    def calculate_upos(self):
        self.upos = []
        for i in self.sentences:
            tmp = nltk.pos_tag(i)
            tmp = list(list(zip(*tmp))[1])
            self.upos = self.upos + [tmp]

    def calculate_chunk(self):
        self.chunk = []
        tagger1 = TaggerLR_Chunk()
        tagger1.load('../models/tagger_lr_chunk.pickle')
        tagger1.sentences = self.sentences
        tagger1.calculate_upos()
        tagger1.calculate_features()
        pred = tagger1.model.predict(tagger1.features)
        first = 0
        final = []
        for s in self.sentences:
            last = first + len(s)
            final = final + [pred[first:last]]
        self.chunk = final


    def train(self, file_name):

        self.read_data(file_name)
        self.calculate_upos()
        self.calculate_chunk()
        self.calculate_features()

        self.model.fit(self.features, self.labels)


    def test(self, file_name, labels_to_remove=[]):

        self.read_data(file_name)
        self.calculate_upos()
        self.calculate_chunk()
        self.calculate_features()

        y_pred = self.model.predict(self.features)

        accuracy_score_ = accuracy_score(self.labels, y_pred)
        confusion_matrix_ = confusion_matrix(self.labels, y_pred)

        indexes = ()
        for idx, label in enumerate(self.labels):
            if label != 'O':
                indexes = indexes + (idx,)

        y_pred = [y_pred[i] for i in indexes]
        self.labels = [self.labels[i] for i in indexes]
        precision_score_ = precision_score(self.labels, y_pred, average='micro')
        recall_score_ = recall_score(self.labels, y_pred, average='micro')
        f1_score_ = f1_score(self.labels, y_pred, average='micro')

        return precision_score_, recall_score_, f1_score_, accuracy_score_, confusion_matrix_, self.tag_sents(self.sentences)

    def tag(self, sentence):
        self.sentences = [sentence]
        self.calculate_upos()
        self.calculate_chunk()
        self.calculate_features()
        pred = self.model.predict(self.features)

        tmp = []
        tmp1 = self.upos[0]
        tmp2 = self.chunk[0]
        for w, p, t, l in zip(sentence, tmp1, tmp2, pred):
            tmp = tmp + [[w, p, t, l]]
        return tmp

    def tag_sents(self, sentences):
        self.sentences = sentences
        self.calculate_upos()
        self.calculate_chunk()
        self.calculate_features()
        pred = self.model.predict(self.features)

        first = 0
        final = []
        for s, u, c in zip(sentences, self.upos, self.chunk):
            sentence = []
            last = first + len(s)
            tmp = pred[first:last]
            for w, p, t, l in zip(s, u, c, tmp):
                sentence = sentence + [[w, p,t, l]]
            final = final + [sentence]
            first = first + last
        return final

    def load(self, file_name):

        with open(file_name, 'rb') as file:
            self.model = pickle.load(file)

    def save(self, file_name):

        with open(file_name, 'wb') as file:
            pickle.dump(self.model, file, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    data_dir = '../data_mini/ner/'
    # data_dir = '../data/ner/'

    tagger = TaggerLR_NER()

    tagger.train(data_dir + 'eng.train.txt')
    print('eval acc:', tagger.evaluate(data_dir + 'eng.train.txt'))

    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'eng.testa.txt')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)

    directory = '../graphics'
    if not os.path.exists(directory):
        os.makedirs(directory)

    seaborn.heatmap(confusion, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g',
                        annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None,
                        square=False, xticklabels=set(tagger.labels), yticklabels=set(tagger.labels), mask=None, ax=None)
    plt.savefig('../graphics/lr_ner_confusion_matrix')
    plt.show()

    directory = '../models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    tagger.save('../models/tagger_lr_ner.pickle')

    '''
    --    :    B-NP    O
    Brussels    NNP    I-NP    B-ORG
    Newsroom    NNP    I-NP    I-ORG
    32    CD    I-NP    O
    2    CD    I-NP    O
    287    CD    I-NP    O
    6800    CD    I-NP    O

    There    EX    B-NP    O
    was    VBD    B-VP    O
    no    DT    B-NP    O
    Bundesbank    NNP    I-NP    B-ORG
    intervention    NN    I-NP    O
    .    .    O    O
    '''
    tagger = TaggerLR_NER()
    tagger.load('../models/tagger_lr_ner.pickle')
    print()
    print(tagger.tag(['--', 'Brussels', 'Newsroom', '32', '2', '287', '6800']))
    print()
    print(tagger.tag_sents([
        ['--', 'Brussels', 'Newsroom', '32', '2', '287', '6800'],
        ['There', 'was', 'no', 'Bundesbank', 'intervention', '.']]))