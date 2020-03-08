from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from itertools import chain
from nltk import NaiveBayesClassifier

words = pd.read_csv("final_word_data.csv")
words1 = pd.DataFrame(words,columns=['content','type'])
print(words1)

l = []

# for i in range(words1.shape[0]):
# 	l.append([(words1['content'], words1['type'])])

for i in range(words1.shape[0]):
	#words1['content'][i] = word_tokenize(words1['content'][i])
	l.append((words1['content'][i], words1['type'][i]))
print(l)

# df_x_train, df_x_test= train_test_split(words1, test_size=0.2)
# print(df_x_train)

# all_words = set(word for passage in words1 for word in word_tokenize(passage[0]))
t = [({word: (word in x[0]) for word in words1}, x[1]) for x in l]

clf = NaiveBayesClassifier.train(t)
clf.show_most_informative_features()

# words = pd.read_csv("final_word_data.csv")

# #token = RegexpTokenizer(r'[a-zA-Z0-9]+')
# cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1))
# df_x= cv.fit_transform(words['content'])

# df_y = words['type']

# df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2)

# clf = MultinomialNB().fit(df_x_train, df_y_train)