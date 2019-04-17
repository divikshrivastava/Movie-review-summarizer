
# coding: utf-8

# In[3]:


from nltk.corpus import movie_reviews


# In[4]:


import nltk
nltk.download('movie_reviews')


# In[5]:


movie_reviews.categories()


# In[6]:


movie_reviews


# In[7]:


movie_reviews.fileids('neg')


# In[8]:


movie_reviews.words(movie_reviews.fileids()[5])


# In[9]:


documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))
documents[0:5]


# In[10]:


import random
random.shuffle(documents)
documents[0:5] # SO THAT NEG POS GET SHUFFLED


# In[11]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[12]:


from nltk.corpus import wordnet
def get_simple_pos(tag):
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[13]:


from nltk import pos_tag
w = "better"
pos_tag([w])


# In[14]:


from nltk.corpus import stopwords
import string
stops = set(stopwords.words('english'))
punctuations = list(string.punctuation)
stops.update(punctuations)
stops, string.punctuation


# In[15]:


def clean_review(words):
    output_words = []
    for w in words:
        if w.lower() not in stops:
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w, pos = get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words


# In[17]:


documents = [(clean_review(document), category) for document, category in documents]


# In[18]:


documents[0]


# In[19]:


training_documents = documents[0:1500]
testing_documents = documents[1500:]


# In[20]:


a = [1,2]
b = [3,4]
a += b
a


# In[21]:


all_words = []
for doc in training_documents:
    all_words += doc[0]


# In[22]:


import nltk


# In[23]:


freq = nltk.FreqDist(all_words)
common = freq.most_common(3000)
features = [i[0] for i in common]


# In[24]:


features


# In[25]:


def get_feature_dict(words):
    current_features = {}
    words_set = set(words)
    for w in features:
        current_features[w] = w in words_set
    return current_features


# In[26]:


output = get_feature_dict(training_documents[0][0])
output


# In[27]:


training_data = [(get_feature_dict(doc), category) for doc, category in training_documents]
testing_data = [(get_feature_dict(doc), category) for doc, category in training_documents]


# In[28]:


training_data[0]


# In[29]:


from nltk import NaiveBayesClassifier


# In[30]:


classfier = NaiveBayesClassifier.train(training_data)


# In[31]:


nltk.classify.accuracy(classfier, testing_data)


# In[32]:


classfier.show_most_informative_features(15)


# In[33]:


from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier


# In[34]:


svc = SVC()
classifier_sklearn = SklearnClassifier(svc)


# In[35]:


classifier_sklearn.train(training_data)


# In[36]:


nltk.classify.accuracy(classifier_sklearn, testing_data)


# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


rfc = RandomForestClassifier()
classifier_sklearn1 = SklearnClassifier(rfc)


# In[39]:


classifier_sklearn1.train(training_data)


# In[40]:


nltk.classify.accuracy(classifier_sklearn1, testing_data)

