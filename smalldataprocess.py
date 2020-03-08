import pandas as pd
import numpy as np

words = pd.read_csv("word_data.csv")

def isNaN(num):
    return num != num

for i in range(words.shape[0]):
	if words['type'][i] == "0" or words['type'][i] == "rumor":
		words['type'][i] = int(0)
	if words['type'][i] == "1":
		words['type'][i] = int(1)
	if words['type'][i] != 0 and words['type'][i] != 1:
		words.drop(i)
	if isNaN(words['type'][i]):
		words.drop(i)

words.to_csv("final_word_data.csv")