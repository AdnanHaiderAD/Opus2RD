#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import collections
from six.moves import xrange
import math
import random
import numpy as np
# pylint: disable=g-import-not-at-top
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from PreProcess import PreProcess


class Word2Vec:
	data_index = 0

	def __init__(self,filepath,fileReadMode='zip',vocabSize =20000):
		self.PreProcessObj = PreProcess(filepath,fileReadMode)
		data,word2Int,int2Word = self.PreProcessObj.processCorpus(vocabSize);
		self.data = data
		self.word2Int = word2Int
		self.int2Word = int2Word
		self.vocabSize = self.PreProcessObj.getVocabSize()	

	def  configure(self,training_steps=200000,batch_size=128,valid_size=70,validSampStInd = 100,validSampEndInd=200,embedding_size=128,skip_window=3,num_skips=2,lossfunction='nce',optimiser='gradient_descent',learning_rate=1.0,num_sampled_nce=64):
		self.training_steps = training_steps
		self.batch_size = batch_size
		self.valid_size = valid_size
		self.validSampStIndex = validSampStInd
		self.validSampEndIndex = validSampEndInd
		self.embedding_size = embedding_size  # Dimension of the embedding vector.
		self.skip_window = skip_window     # How many words to consider left and right.
		self.num_skips = num_skips
		self.vocabSize =  self.PreProcessObj.getVocabSize()
		self.lossfunction = lossfunction
		self.optimiser = optimiser
		self.learning_rate = learning_rate
		self.num_sampled_nce = num_sampled_nce
	

		
	###############################################################################
	
	def generate_batch(self):
		""" Function to generate a training batch for the skip-gram model."""
		assert self.batch_size % self.num_skips == 0
		assert self.num_skips <= 2 * self.skip_window
		batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
		labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
		span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
		
		# create a buffer to hold the current context
		buffer = collections.deque(maxlen=span)
		
		for _ in range(span):
			buffer.append(self.data[Word2Vec.data_index])
			Word2Vec.data_index = (Word2Vec.data_index+1)%len(self.data)
		
		for i in range(self.batch_size // self.num_skips):
			target = self.skip_window  # target label at the center of the buffer
			targets_to_avoid = [self.skip_window]
			for j in range(self.num_skips):
				while target in targets_to_avoid:
					target = random.randint(0, span - 1)
				targets_to_avoid.append(target)
				batch[i * self.num_skips + j] = buffer[self.skip_window]
				labels[i * self.num_skips + j, 0] = buffer[target]
			buffer.append(self.data[Word2Vec.data_index])
			Word2Vec.data_index = (Word2Vec.data_index+1)%len(self.data)   
		# Backtrack a little bit to avoid skipping words in the end of a batch
		Word2Vec.data_index = (Word2Vec.data_index + len(self.data) - span) % len(self.data)
		return batch, labels	
		

	def createValidationSet(self):
		    # Random set of words to evaluate similarity on.
		valid_window =  range(self.validSampStIndex ,self.validSampEndIndex)  # Only pick dev samples in the head of the distribution.
		self.valid_examples = np.random.choice(valid_window, self.valid_size, replace=False)
		self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
					
	##################################################################################

	def createEmbeddingMatrix(self):
		return tf.Variable(tf.random_uniform([self.vocabSize, self.embedding_size], -1.0, 1.0), name='Embedding')


	def weight_variable(self):
		return tf.Variable(tf.truncated_normal([self.vocabSize, self.embedding_size],stddev=1.0 / math.sqrt(self.embedding_size)))

	def bias_variable (self):
		return tf.Variable(tf.zeros([self.vocabSize])) 	


	def lossFunction(self,embeddings,train_inputs,train_labels,num_sampled_nce=64):
		if self.lossfunction == 'CE':
			train_one_hot = tf.one_hot(self.train_labels, self.vocabSize)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.embedding_size, labels=train_one_hot))
		else:
			nce_weights = self.weight_variable()
			nce_biases  = self.bias_variable()
			embed = tf.nn.embedding_lookup(embeddings, train_inputs)
			loss = tf.reduce_mean(
					tf.nn.nce_loss(weights=nce_weights,
										 biases=nce_biases,
										 labels=train_labels,
										 inputs=embed,
										 num_sampled=num_sampled_nce,
										 num_classes=self.vocabSize))
		return loss	

	def chooseOptimiser(self, loss,mode='gradient_descent'):
		if mode =='gradient_descent':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
		else:
			optimizer =	tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
		return optimizer		

	
	#########################################################################


	def writeLookUpTableToFile(self,dirToSave):
		file= open(dirToSave+'/metadata.tsv','w')
		file.write('Word\tID\n')
		for id,word in self.int2Word.items():
			file.write(word+'\t'+str(id)+'\n')
		file.close()		

	def setUpTensorBoard(self,dirToSave  ,final_embeddings):
		from tensorflow.contrib.tensorboard.plugins import projector
		summary_writer = tf.summary.FileWriter(dirToSave)
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = final_embeddings.name
		# Link this tensor to its metadata file (e.g. labels).
		embedding.metadata_path = os.path.join(dirToSave, 'metadata.tsv')
		# Saves a configuration file that TensorBoard will read during startup.
		projector.visualize_embeddings(summary_writer, config)
	
	def trainWord2Vec(self, dirToSave,modelToLoad='',saveIntermediateModels = False):
		###Create a dataflow graph
		graph = tf.Graph()
		with graph.as_default():
			train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
			train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
			self.createValidationSet()
			# Ops and variables pinned to the CPU because of missing GPU implementation
			embeddings = self.createEmbeddingMatrix()
			if self.lossfunction == 'nce':
				with tf.device('/cpu:0'):
					# Look up embeddings for inputs.
					loss = self.lossFunction(embeddings,train_inputs, train_labels,num_sampled_nce=self.num_sampled_nce)
			else:
				with tf.device('/gpu:0'):
					# Look up embeddings for inputs.
					loss = self.lossFunction(embeddings,train_inputs, train_labels,num_sampled_nce=self.num_sampled_nce)
			
			optimiser = self.chooseOptimiser(loss,mode=self.optimiser)
			# Compute the cosine similarity between minibatch examples and all embeddings.
			norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
			normalized_embeddings = embeddings / norm
			valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)
			similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
			# Add variable initializer.
			init = tf.global_variables_initializer()
			
		# Link Python program to C++ interface and execute operations on the graph 
		num_steps = self.training_steps
		with graph.as_default():
			saver = tf.train.Saver()
		with tf.Session(graph=graph) as session:
		# We must initialize all variables before we use them.
			init.run()
			if(modelToLoad !=''):
				saver.restore(session,modelToLoad)	
				print('Initialized from Intermediate model')
			average_loss = 0
			for step in xrange(num_steps):
				batch_inputs, batch_labels = self.generate_batch()
				feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
				_, loss_val = session.run([optimiser, loss], feed_dict=feed_dict)
				average_loss += loss_val
				
				if step % 2000 == 0:
					if step > 0:
						average_loss /= 2000
					# The average loss is an estimate of the loss over the last 2000 batches.
					print('Average loss at step ', step, ': ', average_loss)
					average_loss = 0
				# Note that this is expensive (~20% slowdown if computed every 500 steps)
				if step % 10000 == 0:
					sim = similarity.eval()
					for i in xrange(self.valid_size):
						valid_word = self.int2Word[self.valid_examples[i]]
						top_k = 8  # number of nearest neighbors
						nearest = (-sim[i, :]).argsort()[1:top_k + 1]
						log_str = 'Nearest to %s:' % valid_word
						for k in xrange(top_k):
							close_word = self.int2Word[nearest[k]]
							log_str = '%s %s,' % (log_str, close_word)
						print(log_str)
						if (saveIntermediateModels):
							saver.save(session,os.path.join(dirToSave,'model_intermediate.ckpt'))
			final_embeddings = normalized_embeddings.eval()
			##Save model 
			self.writeLookUpTableToFile(dirToSave)
			saver.save(session,os.path.join(dirToSave,'model.ckpt'))
			self.setUpTensorBoard(dirToSave,embeddings)		
		return final_embeddings,self.word2Int	  
	
	def plot_with_labels(self,low_dim_embs, labels, filename='tsne.png'):
		assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
		plt.figure(figsize=(18, 18))  # in inches
		for i, label in enumerate(labels):
			x, y = low_dim_embs[i, :]
			plt.scatter(x, y)
			plt.annotate(label,
								 xy=(x, y),
								 xytext=(5, 2),
								 textcoords='offset points',
								 ha='right',
								 va='bottom')

		plt.savefig(filename)

	def displayResults(self,embeddings,validSampEndInd):
		try:
			tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
			#plot_only = 500
			low_dim_embs = tsne.fit_transform(embeddings[:validSampEndInd+500, :])
			labels = [self.int2Word[i] for i in xrange(validSampEndInd+500)]
			self.plot_with_labels(low_dim_embs, labels)
		
		except ImportError:
			print('Please install sklearn, matplotlib, and scipy to show embeddings.')
			

	
			
#########################################################################################################################################
