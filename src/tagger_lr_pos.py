# -*- coding: utf-8 -*-

from tagger import Tagger
import nltk
import seaborn
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV

class TaggerLR_POS(Tagger):

    def __init__(self):
        Tagger.__init__(self)
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
            'upos' : self.upos[idx][index]
        }

    def calculate_features(self):
        for idx,s in enumerate(self.sentences):
            for index,word in enumerate(s):
                self.features.append(self.create_features(word, index, idx))

    def calculate_upos(self):
        self.upos = []
        for i in self.sentences:
            tmp = nltk.pos_tag(i,tagset='universal')
            tmp = list(list(zip(*tmp))[1])
            self.upos = self.upos + [tmp]

    def train(self, file_name):

        self.read_data(file_name)
        self.calculate_upos()
        self.calculate_features()

        self.model.fit(self.features, self.labels)


    def test(self, file_name, labels_to_remove=[]):

        self.read_data(file_name)
        self.calculate_upos()
        self.calculate_features()

        y_pred = self.model.predict(self.features)
        precision_score_ = precision_score(self.labels, y_pred, average='micro')
        recall_score_ = recall_score(self.labels, y_pred, average='micro')
        f1_score_ = f1_score(self.labels, y_pred, average='micro')
        accuracy_score_ = accuracy_score(self.labels, y_pred)
        confusion_matrix_ = confusion_matrix(self.labels, y_pred)

        return precision_score_, recall_score_, f1_score_, accuracy_score_, confusion_matrix_, self.tag_sents(self.sentences)

    def tag(self, sentence):
        self.sentences = [sentence]
        self.calculate_upos()
        self.calculate_features()
        pred = self.model.predict(self.features)

        tmp = []
        tmp1 = self.upos[0]
        for w, p, l in zip(sentence, tmp1, pred):
            tmp = tmp + [[w, p, l]]
        return tmp

    def tag_sents(self, sentences):
        self.sentences = sentences
        self.calculate_upos()
        self.calculate_features()
        pred = self.model.predict(self.features)

        first = 0
        final = []
        for s, u in zip(sentences, self.upos):
            sentence = []
            last = first + len(s)
            tmp = pred[first:last]
            for w, p, l in zip(s, u, tmp):
                sentence = sentence + [[w, p, l]]
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

    # data_dir = 'data_mini/pos/'
    data_dir = 'data/pos/'

    tagger = TaggerLR_POS()

    tagger.train(data_dir + 'en-ud-train.conllu')
    print('eval acc:', tagger.evaluate(data_dir + 'en-ud-train.conllu'))

    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'en-ud-dev.conllu')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)
    print()

    # print(tagged_sents)

    seaborn.heatmap(confusion, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g',
                        annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None,
                        square=False, xticklabels=set(tagger.labels), yticklabels=set(tagger.labels), mask=None, ax=None)

    plt.show()

    tagger.save('models/tagger_lr_pos.pickle')
    del tagger
    try:
        print(tagger)
    except Exception as e:
        print(e)
    '''
    I   PRON    PRP
    do  AUX VBP
    n't PART    RB
    think   VERB    VB
    it  PRON    PRP
    matters VERB    VBZ

    Gets    VERB    VBZ
    the DET DT
    Job NOUN    NN
    Done    ADJ JJ
    '''
    tagger = TaggerLR_POS()
    tagger.load('models/tagger_lr_pos.pickle')
    print()
    print(tagger.tag(['I', 'do', 'n\'t', 'think', 'it', 'matters']))
    print()
    print(tagger.tag_sents([
        ['I', 'do', 'n\'t', 'think', 'it', 'matters'],
        ['Gets', 'the', 'Job', 'Done']]))