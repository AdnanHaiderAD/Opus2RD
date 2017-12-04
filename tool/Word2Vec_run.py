import sys
import os
sys.path.append('../src/')
from Word2Vec import Word2Vec


batch_size = 256      # decision on the size of the batch impacts the choice of learning rate
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
training_steps = 600000 # Number of updates
# We pick a random validation set to sample nearest neighbors. 
valid_size = 49       # Size of validation set 
wordIdStart = 100     # Consider wordids whose index is between this range
wordIdEnd = 150 
num_sampled = 64      # This is feature specific to NCE training, it determines how words to samples from uniform distribution
VocabularySize = 10000 # Number of words /size of vocabulary
loss = 'nce'           # the other criterion is 'CE' and it expects the machine to have gpu or else error
optimiser ='gradient_descent' # the other optimiser is ADAM and can be called by 'adam'
learning_rate = 1.0
dir_to_Save = '../models'


Word2Vec_model = Word2Vec(filepath ='../data/text8.zip',fileReadMode='zip',vocabSize=VocabularySize)
Word2Vec_model.configure(
        training_steps=training_steps,
	batch_size=batch_size,
	valid_size=valid_size,
	validSampStInd = wordIdStart,
	validSampEndInd=wordIdEnd,
	embedding_size=embedding_size,
	skip_window=skip_window,
	num_skips=num_skips,
	lossfunction = loss,
	optimiser=optimiser,
	learning_rate=learning_rate,
	num_sampled_nce=num_sampled,
	)
final_embeddings = Word2Vec_model.trainWord2Vec(dir_to_Save)	
Word2Vec_model.displayResults(embeddings=final_embeddings, validSampEndInd=wordIdEnd)


# Note : When running Tensorboard, for good visualization TSNE is recommended with setting TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000) 
