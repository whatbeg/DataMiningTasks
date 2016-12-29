#! usr/bin/python
# coding=utf-8
"""
File Name: Apriori.py
Description: Implementation of Apriori Algorithm
Date: 2016-09-19
Author: QIU HU
"""
import collections


class Apriori(object):

    def __init__(self, path):

        self.min_sup = 2
        self.dataset = []
        self.frequent_itemset = {}
        self.path = path

    def load_dataset(self):

        with open(self.path, 'r') as f:
            for line in f.readlines():
                t = line.strip().split('\t')
                self.dataset.append(t[1:])

    def find_frequent_one_itemsets(self):  # return a dictionary like dic[(I1, )] = 3

        itemsets = collections.defaultdict(int)
        for transaction in self.dataset:
            for item in transaction:
                itemsets[(item, )] += 1
        del_list = []
        for its in itemsets:
            if itemsets[its] < self.min_sup:
                del_list.append(its)
        for its in del_list:
            itemsets.pop(its)
        return itemsets

    def apriori_gen(self, fkm):
        # fkm: Frequent (k-1)item-set, represented by a dictionary which keys contains item-set
        Ck = {}
        for l1 in fkm.keys():
            for l2 in fkm.keys():

                k = len(l1)
                if l1[k-1] < l2[k-1] and l1[:k-1] == l2[:k-1]:
                    c = l1[:] + (l2[k-1], )
                    if not self.has_infrequent_subset(c, fkm):
                        Ck[c] = 0
        return Ck

    def has_infrequent_subset(self, c, fkm):  # c is a list

        for s in self.subset(c):
            if s not in fkm.keys():
                return True
        return False

    def subset(self, c):

        for i in range(len(c)):
            yield c[:i] + c[i+1:]

    def union(self, fkm):

        for its in fkm:
            self.frequent_itemset[its] = fkm[its]

    def work(self):

        self.load_dataset()
        fkm = self.find_frequent_one_itemsets()  # Frequent (k-1)item-set
        while len(fkm) > 0:
            print(fkm)
            self.union(fkm)
            Ck = self.apriori_gen(fkm)
            for transaction in self.dataset:
                for c in Ck:
                    if set(c).issubset(set(transaction)):
                        Ck[c] += 1
            del_list = []
            for c in Ck:
                if Ck[c] < self.min_sup:
                    del_list.append(c)
            for c in del_list:
                Ck.pop(c)
            fkm = Ck

if __name__ == '__main__':

    apriori = Apriori('data.txt')
    apriori.work()
    print(apriori.frequent_itemset)
