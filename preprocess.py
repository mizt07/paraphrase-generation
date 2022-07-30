import csv
from distutils.command.build import build
import pickle
from nltk.tokenize import word_tokenize
from copy import deepcopy
import json
import sys
import numpy as np
import h5py
from collections import Counter
import matplotlib.pyplot as plt

def paws_split(filename, output_name): # reads from .tsv and transfers data to .json
	final_output = []
	fieldnames = ["id", "sentence1", "sentence2", "label"]
	csvin = open(filename, "r")
	reader = csv.DictReader(csvin, fieldnames=fieldnames, delimiter="\t")
	count = 0
	added = 0
	for row in reader:
		count += 1
		if count == 1: continue
		if row["label"] == "1": #paraphrase only
			added += 1
			dict_obj = {}
			s1 = row["sentence1"]
			s2 = row["sentence2"]
			# print("======")
			# print("q1:", s1)
			# print("q2:", s2)
			dict_obj["sentence"] = s1
			dict_obj["paraphrase"] = s2
			# if added > 10: break
			final_output.append(dict_obj)
			# tok = word_tokenize(s1)
			# tok = tok = tok[:len(tok)-1] #remove period
			# print(tok)
	with open(output_name, "w") as f:
		json.dump(final_output, f)

	print(f"Added {added} items.")

class Preprocess:
	def __init__(self, corpus):
		self.data = corpus
		self.itow = {} #31217 length
		self.wtoi = {} # 31217 length
		self.word_freq = {}

	def get_data(self):
		return self.data
	
	def get_itow(self):
		return self.itow
	
	def get_wtoi(self):
		return self.wtoi

	def add_tokenized(self, key):
		for i, x in enumerate(self.data): 
			if key=="processed_tokens":
				txt = word_tokenize(x["sentence"])
				txt = txt[:len(txt)-1] 
				x[key] = txt
			elif key=="processed_tokens_duplicate":
				txt = word_tokenize(x["paraphrase"])
				txt = txt[:len(txt)-1]
				x[key] = txt

			if i < 15: 
				print("Tokens Preview: ", txt)
			
			if i % 1000 == 0:
				sys.stdout.write("Tokenizing senteneces.. %d/%d (%.2f%% done)   \r" %  (i, len(self.data), i*100.0/len(self.data)) )
				sys.stdout.flush() 
	
	def build_vocab(self):
		counts = {}	
		for x in self.data:
			for word in x["processed_tokens"]:
				counts[word] = counts.get(word, 0) + 1 # https://realpython.com/python-counter/

		self.word_freq = deepcopy(counts)
		#self.word_freq = sorted([(count,w) for w,count in counts.items()], reverse=True) 
		# print("top words and their counts:")
		# print("\n".join(map(str,top_words[:20])))
		vocab = [w for w, _ in counts.items()]
		vocab.append("UNK")

		self.itow = {i+1:w for i,w in enumerate(vocab)} 
		self.wtoi = {w:i+1 for i,w in enumerate(vocab)}
		
		for x in self.data: #check for tokens less than 0 (no word = UNK [unknown])
			x["processed_tokens"] = [w if counts.get(w,0) > 0 else "UNK" for w in x["processed_tokens"]]
			x["processed_tokens_duplicate"] = [w if counts.get(w,0) > 0 else "UNK" for w in x["processed_tokens_duplicate"] ]  
		return vocab			

	def encode(self):
		max_length = 30 
		count = 0
		N = len(self.data)

		original_arr = np.zeros((N, max_length), dtype="uint32")
		original_len = np.zeros(N, dtype="uint32")
		paraphrase_arr = np.zeros((N, max_length), dtype="uint32") 
		paraphrase_len = np.zeros(N, dtype="uint32")
		
		for i,x in enumerate(self.data):
			original_len[count] = min(max_length, len(x["processed_tokens"]))
			paraphrase_len[count] = min(max_length, len(x["processed_tokens_duplicate"])) 
			count += 1
			for k,w in enumerate(x["processed_tokens"]):
				if k < max_length:
					#print("original: ", w, "index: ", wtoi[w])
					original_arr[i,k] = self.wtoi[w]
			for k,w in enumerate(x["processed_tokens_duplicate"]):        
				if k < max_length:
					paraphrase_arr[i,k] = self.wtoi[w]   
			# convert each word in sentence to vocab index 
		#print(original_arr)    
		return original_arr, original_len, paraphrase_arr, paraphrase_len

	# def make_vocab_charts(self):
	# 	fig = plt.figure()
	# 	plt.title('Token Frequency Distribution')
	# 	plt.xlabel(xlabel='Token ID (sorted by frequency high to low)')
	# 	plt.ylabel(ylabel='Frequency')
	# 	plt.yscale('log')
	# 	plt.plot(np.arange(len(self.word_freq)), np.array(list(self.word_freq.values())), label='word frequencies')
	# 	plt.axhline(y=50, c='red', linestyle='dashed', label="freq=50")
	# 	plt.legend()
	# 	# plt.show()
	# 	plt.savefig('tfd.png')

	# 	training_tokens = sum(self.word_freq.values())
	# 	cutoff = 0
	# 	occurrences = []
	# 	for word in self.word_freq.keys():
	# 		occurrences.append(cutoff / training_tokens)
	# 		if 0.91 < (cutoff / training_tokens) < 0.92:
	# 			print(cutoff / training_tokens, self.word_freq[word], self.wtoi[word])
	# 		cutoff += self.word_freq[word]
	# 	fig = plt.figure()
	# 	plt.title('Cumulative Fraction Covered')
	# 	plt.xlabel(xlabel='Token ID (sorted by frequency high to low)')
	# 	plt.ylabel(ylabel='Fraction of Token Occurrences Covered')
	# 	plt.plot(np.arange(len(self.word_freq)), np.array(occurrences), label='cutoff curve')
	# 	plt.axvline(x=6530, c='red', linestyle='dashed', label="y=0.91")
	# 	plt.legend()
	# 	plt.show()
	# 	plt.savefig('cfc.png')

if __name__ == "__main__":
	export_h5 = "data/0_data_prepo_new.h5"
	export_json = "data/0_data_index_word.json"
	export_vocab = "data/0_vocab_lel.txt"
	train_data = json.load(open("data/paws/PAWS_train.json", "r"))

	preprocess = Preprocess(train_data)
	preprocess.add_tokenized("processed_tokens")
	preprocess.add_tokenized("processed_tokens_duplicate")
	vocab = preprocess.build_vocab()

	# with open(export_vocab, "w") as vocab_save:
	# 	for x in vocab:
	# 		vocab_save.write(str(x) + "\n")
	# 	vocab_save.close() #just to see and test

	with open("vocab.txt", "wb") as vocab_save:
		pickle.dump(vocab, vocab_save)
		vocab_save.close()
	
	sentence_arr, sentence_arr_len , paraphrase_arr, paraphrase_arr_len = preprocess.encode()
	print("Sentence Train shape: ", sentence_arr.shape) # (21829, 30)

	f = h5py.File(export_h5, "w")
	f.create_dataset("sentences_array", dtype="uint32", data=sentence_arr)
	f.create_dataset("sentences_length", dtype="uint32", data=sentence_arr_len)
	f.create_dataset("paraphrases_array", dtype="uint32", data=paraphrase_arr)
	f.create_dataset("paraphrases_length", dtype="uint32", data=paraphrase_arr_len)

	f.close()
	print("h5py file generated: ", export_h5)
	
	out = {}
	out["index_to_word"] = preprocess.get_itow()
	json.dump(out, open(export_json, "w"))
	print ("json file generated:  ", export_json)
	