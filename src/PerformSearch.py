#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from six.moves import xrange
import math
import random
import numpy as np
import pickle
import numpy.linalg as matlib


class DeepSearch:
	def __init__(self,dir):
 		# load objects, embedding matrix, word2Int dic
		with open(dir, "rb") as f:
    			collections = pickle.load(f)
			self.embedding = collections['embeddings']
			self.lookUpTable = collections['word2Int']
	
	def convertParagraphToIntList(self, parag):
		data_int = []
		for word in parag:
			data_int.append(self.lookUpTable[word])
		return data_int

	def vanillaParagraphEncoding(self,list):
		hot_encoding = np.zeros(self.embedding.shape[1])
		for item in list:
			index = self.word2Int[item]
			hot_encoding[index] = 1
		paragraph_raw_embedding = np.matmul(hot_encoding,self.embedding)
		paragraph_embedding = paragraph_raw_embedding/matlib.norm(paragraph_raw_embedding)
		return paragraph_embedding	

	def computeSimilarity(self,vector1,vector2):
		# assumes vectors have unit norm then dot prod simply corresponds to cosine similarity
		return np.dot(vector1,vector2)

	def performSearch(self,parag1, parag2):
		# Perform encoding on paragraph 1
		pg1_data = self.convertParagraphToIntList(parag1)
		pg1_encoding = self.vanillaParagraphEncoding(pg1_data)
		# Perform encoding on paragraph 2
                pg2_data = self.convertParagraphToIntList(parag2)
                pg2_encoding = self.vanillaParagraphEncoding(pg2_data)
		sim = self.computeSimilarity(pg1_encoding,pg2_encoding)
		return sim

							
