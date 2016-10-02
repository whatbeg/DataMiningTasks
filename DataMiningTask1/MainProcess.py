#! usr/bin/python
# coding=utf-8
"""
File Name: MainProcess.py
Description: Statistics the TF-IDF of ICML papers with 15 classes.
Date: 2016-09-15
Author: QIU HU
Student ID: MG1633031
"""

import os
import re
import collections
import math
from nltk import word_tokenize
import nltk.stem
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Statistic(object):

    def __init__(self):

        self.cwd = os.getcwd()    # 当前工作目录
        self.data_dir = os.path.join(self.cwd, 'ICML')  # F:\Python Project\DataMiningTask1\ICML
        self.pattern_1 = re.compile('[^A-Za-z]')
        self.stop_words = collections.defaultdict(int)
        self.token_list = []
        self.dir_list = []
        self.TF = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        self.record_IDF = collections.defaultdict(lambda: collections.defaultdict(int))
        self.IDF = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        self.TFIDF = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
        self.work()

    def join_all_articles(self, data_dir):

        file_list = []
        for name in os.listdir(data_dir):   # name is a dir or a file

            full_path = os.path.join(data_dir, name)
            if os.path.isdir(full_path):
                file_list.extend(self.join_all_articles(full_path))
                self.dir_list.append(full_path)        # dir is like 'F:\Python Project\DataMiningTask1\ICML'
            else:
                file_list.append(full_path)

        return file_list

    def load_stop_words(self):

        path = os.path.join(self.cwd, 'stopwords.txt')
        with open(path) as f:
            for line in f.readlines():
                word = line.strip()
                self.stop_words[word] = 1

    def token_analyser(self, all_tokens, token, dir_name, article):

        if len(token) > 2:
            token = token.lower()   # all in lowercase
            all_tokens[token] = 1
            self.TF[dir_name][article][token] += 1    # in a class, this article this token showed 1 time
            self.record_IDF[token][article] = 1   # this token showed in this article, all of articles

    def find_all_token(self, all_articles):

        all_tokens = collections.defaultdict(int)
        for article in all_articles:
            dir_name = os.path.dirname(article)       # full path of the dir of the article
            with open(article, 'r') as f:             # open article file with it's path
                for line in f.readlines():            # every line of this article
                    line = self.pattern_1.sub(' ', line.strip())  # drop invalid characters
                    tokens = word_tokenize(line.strip())          # word tokenize using NLTK tools
                    stemmer = nltk.stem.SnowballStemmer('english')  # stem the word
                    for token in tokens:              # for tokens in this line
                        token = stemmer.stem(token)
                        self.token_analyser(all_tokens, token, dir_name, article)  # article is full path
        return all_tokens

    def drop_stop_words(self, all_tokens):

        del_key = []
        for key in all_tokens.keys():
            if key in self.stop_words:
                del_key.append(key)
        print("TOTAL STOP WORDS: %d" % len(del_key))
        for key in del_key:
            all_tokens.pop(key)
        return all_tokens

    def sorted_tokens(self, all_tokens):

        self.token_list = sorted(all_tokens)    # word feature vector

    def calc(self):

        article_num = 0
        for clas in self.dir_list:
            article_num += len(os.listdir(clas))  # article_num: numbers of all articles
        for clas in self.dir_list:            # clas is a dir of a class
            result_file = os.path.join(os.path.dirname(clas), str(clas+"_RESULT.txt"))
            res_f = open(result_file, 'w')
            for article in os.listdir(clas):  # article is basename of an article
                res_f.write(article)
                res_f.write('\n')
                article = os.path.join(clas, article)
                total_words = 0
                for token in self.TF[clas][article]:
                    total_words += self.TF[clas][article][token]
                for token in self.TF[clas][article]:
                    self.TF[clas][article][token] = float(self.TF[clas][article][token]) / total_words
                    self.IDF[clas][article][token] = math.log(float(article_num) / len(self.record_IDF[token]))
                    self.TFIDF[clas][article][token] = self.TF[clas][article][token] * self.IDF[clas][article][token]
                result = []
                idx = 1
                for token in self.token_list:
                    if self.TFIDF[clas][article][token] and self.TFIDF[clas][article][token] != 0.0:
                        res = "%d: %.5f" % (idx, self.TFIDF[clas][article][token])
                        result.append(res)
                    idx += 1
                res_f.write(str(result))
                res_f.write('\n\n')

            res_f.close()

    def print_list(self, lis):

        for li in lis:
            print(li)

    def store_word_vector(self, token_list):

        with open('word_vector.txt', 'w') as f:
            for token in token_list:
                f.write(token)
                f.write('\n')

    def work(self):

        all_articles = self.join_all_articles(self.data_dir)
        # self.print_list(self.dir_list)
        all_tokens = self.find_all_token(all_articles)
        self.load_stop_words()         # load 890+ stop words
        all_tokens = self.drop_stop_words(all_tokens)
        print("ALL TOKENS: %d" % len(all_tokens))
        # self.print_list(all_tokens.keys()[200:300])
        self.sorted_tokens(all_tokens)
        # self.print_list(self.token_list[1:386])
        self.store_word_vector(self.token_list)
        self.calc()

if __name__ == '__main__':

    stat = Statistic()
