import sys
import os
sys.path.append('/home/dawna/mah90/Opus2/Project/src/')
from Word2Vec import Word2Vec


batch_size = 256
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2     # How many words to consider left and right.
num_skips = 2       # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. 
valid_size = 49     # Random set of words to evaluate similarity on.
wordIdStart = 100  # Only pick dev samples in the head of the distribution.
wordIdEnd = 150 
num_sampled = 64    #
VocabularySize = 10000
loss = 'nce'
optimiser ='gradient_descent'
learning_rate = 0.1
dir_to_Save = '/home/dawna/mah90/Opus2/Project/models'


Word2Vec_model = Word2Vec(filepath ='/home/dawna/mah90/Opus2/src/text8.zip',fileReadMode='zip',vocabSize=VocabularySize)
Word2Vec_model.configure(
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
Word2Vec_model.displayResults(final_embeddings)

