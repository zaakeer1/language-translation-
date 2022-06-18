import pickle
import sys
import numpy as np
import re
import string

from pickle import dump
from pickle import load
from unicodedata import normalize
from numpy import array
from numpy.random import rand
from numpy.random import shuffle



def ind_to_word_vecs(inds, parameters):
    
    m = inds.shape[1]
    WRD_EMB = parameters['WRD_EMB']
    word_vec = WRD_EMB[inds.flatten(), :].T
    
    assert(word_vec.shape == (WRD_EMB.shape[1], m))
    
    return word_vec

def linear_dense(word_vec, parameters):
   
    m = word_vec.shape[1]
    W = parameters['W']
    Z = np.dot(W, word_vec)
    
    assert(Z.shape == (W.shape[0], m))
    
    return W, Z



def forward_propagation(inds, parameters):
    word_vec = ind_to_word_vecs(inds, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)
    
    caches = {}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z
    
    return softmax_out, caches

def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def rnn_cell_forward_enc(xt, a_prev, parameters):
    
   
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba = parameters["ba"]
  
    waa = np.array(Waa)
    a_prev = np.array(a_prev)
    ba = np.array(ba)
    Wax = np.array(Wax)
    xt = np.array(xt)
    
   
    a_next = np.tanh(np.dot(Wax,xt)+np.dot(Waa,a_prev)+ba)            
    
   
    return a_next

def rnn_cell_forward_dec(a_prev, parameters):
   
    

    Ws = parameters["Ws"]
    Whh = parameters["Whh"]
   
   
    Whh = np.array(Whh)
 
    Ws = np.array(Ws)
   
    
            
    h_next = np.dot(Whh,a_prev)
    h_next = np.reshape(h_next, (1, 50))
    
    yt = softmax(np.dot(Ws,h_next)) 
    
    return h_next, yt

def load_doc(filename):
	
	file = open(filename, mode='rt', encoding='utf-8')
	
	text = file.read()
	
	file.close()
	return text


def tokenize(text):
 
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split(' ') for line in  lines]
	return pairs

def clean_pairs(lines):
	cleaned = list()
	
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	
	table = str.maketrans('', '', string.punctuation)
	
	for pair in lines:
		clean_pair = list()
		for line in pair:
			
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			
			line = line.split()
			
			line = [word.lower() for word in line]
			
			line = [word.translate(table) for word in line]
			
			line = [re_print.sub('', w) for w in line]
			
			line = [word for word in line if word.isalpha()]
			
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	
	return cleaned
	
with open("mySavedDict.txt", "rb") as myFile:
    myNewPulledInDictionary = pickle.load(myFile)


filename = '/home/zak/ml/deu3.txt'

doc = load_doc(filename)


tokens = tokenize(doc)


word_to_id, id_to_word = mapping(tokens)
print(word_to_id)



np.random.seed(1)

a_prev = np.random.randn(1,50)
Waa = np.random.randn(1)
Wax = np.random.randn(1)
Wya = np.random.randn(1,1)
ba = np.random.randn(1)

parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba}


pairs = to_pairs(doc)

clean_data = clean_pairs(pairs)


w1=clean_data[50][0]

w2=clean_data[50][1]


for i in range(191,192):

    length = len(clean_data[i]) 
    for j in range(length):
        no=word_to_id[clean_data[i][j]]
     
        arr_2d = np.reshape(myNewPulledInDictionary["WRD_EMB"][no], (1, 50))
   
        if j==0:
            
            a_next= rnn_cell_forward_enc(arr_2d, a_prev, parameters)
     
            a_next = np.reshape(a_next, (1, 50))
        else:
       
            a_next= rnn_cell_forward_enc(arr_2d, a_next, parameters)
        


div=a_next

filename = '/home/zak/ml/deu4.txt'

doc2 = load_doc(filename)


tokens2 = tokenize(doc2)

word_to_id2, id_to_word2 = mapping(tokens2)
print(word_to_id2)



with open("mySavedDictg.txt", "rb") as myFile:
   
    myNewPulledInDictionary2 = pickle.load(myFile)




pairs2 = to_pairs(doc2)

clean_data2 = clean_pairs(pairs2)


Whh = np.random.randn(1)
Ws = np.random.randn(1)
parameters = {"Whh": Whh, "Ws": Ws}


div = np.reshape(div, (1, 50))

vocab_size = len(id_to_word2)
X_test = np.arange(vocab_size)
X_test = np.expand_dims(X_test, axis=0)
softmax_test, _ = forward_propagation(X_test, myNewPulledInDictionary2 )
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]


for i in range(191,192):
    
    length = len(clean_data2[i]) 
    for j in range(length):
       
        if j==0:
           
            h_next ,outputvec= rnn_cell_forward_dec(div, parameters)
            sno = 0
         
            subt=np.subtract(outputvec,myNewPulledInDictionary2["WRD_EMB"][0])
            minm=sum(subt)
            print(sum(subt))
            for k in range(1,vocab_size):
                subt=np.subtract(outputvec,myNewPulledInDictionary2["WRD_EMB"][k])
                if sum(subt) < minm:
                    minm=sum(subt)
                
            for k in range(vocab_size):
                subt=np.subtract(outputvec,myNewPulledInDictionary2["WRD_EMB"][k])
                if sum(subt) == minm:
                    sno=k
           
            print(id_to_word2[sno])

            h_next = np.reshape(h_next, (1, 50))
        else:
            print(j)
            a_next , outputvec= rnn_cell_forward_dec(h_next, parameters)
        
            subt=np.subtract(outputvec,myNewPulledInDictionary2["WRD_EMB"][0])
            minm=sum(subt)
   
            for k in range(1,vocab_size):
                subt=np.subtract(outputvec,myNewPulledInDictionary2["WRD_EMB"][k])
                if sum(subt) < minm:
                    minm=sum(subt)
                    
            for k in range(vocab_size):
                subt=np.subtract(outputvec,myNewPulledInDictionary2["WRD_EMB"][k])
                if sum(subt) == minm:
                    sno=k
           
            print(id_to_word2[sno])





