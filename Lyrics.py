import nltk
from nltk import data
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from pprint import pprint
from gensim.models import LdaModel, LdaMulticore
from gensim.models import LsiModel

# open file
filename = 'rhapsody_dataset_precleaned.txt'
file = open(filename, 'r+')
text = file.read()
file.close()

#PRE-PROCESSING

# split into words by white space
words = text.split()

# convert to lower case
words = [word.lower() for word in words]

# remove stopwords
stoplist = stopwords.words('english')
clean = [word for word in words if word not in stoplist]

# remove numbers
regex = re.compile("[0-9]")
no_numbers = filter(lambda i: not regex.search(i), clean)
no_numbers = [i for i in clean if not regex.search(i)]

# remove punctuation
no_numbers = [''.join(char for char in s if char not in string.punctuation) for s in no_numbers]

# tokenization of the pre-cleaned text
tokenized_sents = [word_tokenize(i) for i in no_numbers]
print(tokenized_sents)

# TOPIC MODELLING

# create dictionary
dictionary = corpora.Dictionary(tokenized_sents)
print(dictionary)

# create corpus
corpus = [dictionary.doc2bow(word) for word in tokenized_sents]
print(corpus)

# train the LDA Model
lda_model = LdaMulticore(corpus = corpus, id2word = dictionary, num_topics = 8)
lda_model.save('lda_model.model')
lda_model.print_topics(-1)
print(lda_model.print_topics(-1))

# create a LSI topic model
lsi_model = LsiModel(corpus = corpus, id2word = dictionary, num_topics = 8, decay = 0.5)
pprint(lsi_model.print_topics(-1))
