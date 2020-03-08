import pandas as pd
import random
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.stem import PorterStemmer
import string

p = 0.003  # 1% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
df = pd.read_csv(
         "news_cleaned_2018_02_13.csv",
         header=0, 
         skiprows=lambda i: i>0 and random.random() > p,
         na_filter=False)

df_1 = df.drop(['id', 'domain', 'url', 'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors', 'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary', 'source'], axis=1)

df_1.loc[df_1['type'] == 'fake', 'type'] = 0
df_1.loc[df_1['type'] == 'bias', 'type'] = 0
df_1.loc[df_1['type'] == 'satire', 'type'] = 0
df_1.loc[df_1['type'] == 'conspiracy', 'type'] = 0
df_1.loc[df_1['type'] == 'junksci', 'type'] = 0
df_1.loc[df_1['type'] == 'hate', 'type'] = 0
df_1.loc[df_1['type'] == 'unreliable', 'type'] = 0

df_1.loc[df_1['type'] == 'state', 'type'] = 1
df_1.loc[df_1['type'] == 'clickbait', 'type'] = 1
df_1.loc[df_1['type'] == 'political', 'type'] = 1
df_1.loc[df_1['type'] == 'reliable', 'type'] = 1

df_y = df_1['type']
df_x = df_1.drop(['type'], axis=1)

#print(df_x['content'][2])

#preprocessing

#create a string with all possible punctuations
punctuations = "~`!@#$%^&*()_-+={}[]|\\:;<>,.?/\"\'"
# numbers = "0123456789"
#print(type(punctuations))

stemmer = PorterStemmer()
words = stopwords.words('english')

#traverse through the data
count = 0
for x in df_1['content']:
	
	x = x.lower()

	tokens = word_tokenize(x)
	for z in words:
		if z in tokens:
			x = x.replace(" " + z + " ", " ")
		x = x.replace(" " + z + " ", " " + stemmer.stem(z) + " ")

	df_1['content'][count] = x
	print(df_x['content'][count])
	count += 1

df_1.to_csv("word_data.csv")

#print(df_x['content'][2])

# df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2)
