#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import zipfile
import collections



class PreProcess:

	
	def readDataFromFileLists(self,filename):
		filelistReader = open(filename,'r')
		filelist = [line.rstrip('\n') for line in filelistReader]
		filelistReader.close()
		corpus = []
		for file in filelist:
			filehandler = open(file,'r')
			file_data = filehandler.read().replace('\n',' ')
			corpus.extend(file_data.split())
			filehandler.close()			
		
		self.corpus = corpus
			
	

	def readDataFromZipfile(self,filename):
		if not os.path.exists(filename):
			raise Exception(' No zip file of the name ' + filename + ' has been found')

		zipfileHandler = zipfile.ZipFile(filename)
		corpus = []
		for file in zipfileHandler.namelist():
			file_data =  tf.compat.as_str(zipfileHandler.read(file)).split()
			corpus.extend(file_data)

		self.corpus = corpus	 

  
	def processCorpus(self,n_words):
		""" Map words to integers """
		word2Int ={}
		int2Word ={}
		data = list()
		unk_count = 0
		# create list of word and frequencies
		words = []
		for word in self.corpus:
			if word != '.':
				words.append(word)
		count = [['UNK', -1]]
		count.extend(collections.Counter(words).most_common(n_words- 1))
		#create 1 to K  hot encoding
		for word,_ in count:
			word2Int[word] = len(word2Int)
		# get Vocabulary size
		self.VocabularySize = len(word2Int)
		for word in self.corpus:
			if word in word2Int:
				index = word2Int[word]
			else:
				index = 0
				unk_count+=1
			data.append(index)
		int2Word = dict(zip(word2Int.values(),word2Int.keys()))
		return data,word2Int,int2Word
	
	def getVocabSize(self):
		return self.VocabularySize	

	
	def __init__(self,filepath,fileReadMode='zip'):
		self.corpus = []
		self.VocabularySize = 0
		if fileReadMode == 'zip':
			self.readDataFromZipfile(filepath)
		else:
			self.readDataFromFileLists(filepath)			