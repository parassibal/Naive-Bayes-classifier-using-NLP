import os
import sys
import glob as g
import re
import numpy as np
import ast as te
from collections import Counter
import string
import math as m
import pandas as pd

global stop_words,st
st='words_nan'
stop_words=['ad','added','ae','affected','after','afterwards','again','against','ah','al','all','allow','allows','hotel','room','wife','wifes','wives','husband','husbands', 'a', 'amongst', 'amoungst', 'becomes', 'eg', 'fify', 'formerly', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'latterly', 'ltd', 'namely', 'nevertheless', 'sixty', 'thence', 'thereafter', 'thereby', 'therein', 'thereupon', 'thickv', 'twelve', 'whence', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'whither', 'whose', 'yourselves', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'act', 'actually','almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'are', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'au', 'av', 'away', 'b', 'back', 'be', 'became', 'because', 'become', 'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'bottom', 'brief', 'briefly', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'cd', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'could', 'couldn', 'couldnt', 'course', 'cry', 'currently', 'd', 'date', 'dc', 'de', 'definitely', 'describe', 'described', 'despite', 'detail', 'did', 'didn', 'different', 'dj', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'dr', 'due', 'during', 'e', 'each', 'effect', 'eight', 'either', 'el', 'eleven', 'else', 'elsewhere', 'em', 'end', 'enough', 'entirely', 'especially', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'few', 'fi', 'fifteen', 'fifth', 'fill', 'find', 'fire', 'first', 'five', 'fix', 'fl', 'followed', 'following', 'for', 'former', 'forth', 'forty', 'found', 'four', 'from', 'front', 'ft', 'full', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'hadn', 'happens', 'has', 'hasn', 'hasnt', 'have', 'haven', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'herself', 'hi', 'hid', 'him', 'himself', 'his', 'ho', 'hopefully', 'how', 'however', 'hr', 'http', 'hundred', 'i', 'ie', 'if', 'il', 'im', 'immediate', 'immediately', 'important', 'in', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'instead', 'interest', 'into', 'ip', 'is', 'isn', 'it', 'its', 'itself', 'j', 'jr', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'l', 'la', 'largely', 'last', 'lately', 'later', 'latter', 'lb', 'le', 'least', 'les', 'less', 'let', 'lets', 'like', 'likely', 'line', 'little', 'll', 'lo', 'look', 'looking', 'looks', 'los', 'm', 'ma', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'might', 'mill', 'million', 'mine', 'miss', 'ml', 'mo', 'more', 'moreover', 'most', 'mostly', 'move', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'name', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needn', 'needs', 'never', 'new', 'next', 'nine', 'nobody', 'non', 'none', 'nonetheless', 'nor', 'normally', 'noted', 'nothing', 'now', 'nowhere', 'nt', 'ny', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'oj', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'ow', 'own', 'p', 'pages', 'par', 'part', 'particular', 'particularly', 'past', 'pc', 'per', 'perhaps', 'pl', 'placed', 'please', 'plus', 'pm', 'possible', 'possibly', 'potentially', 'present', 'presumably', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'ps', 'put', 'q', 'que', 'quickly', 'quite', 'ran', 'rather', 'rd', 're', 'really', 'reasonably', 'recent', 'recently', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'resulted', 'resulting', 'results', 'right', 'rm', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'se', 'second', 'secondly', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'sent', 'serious', 'seriously', 'seven', 'several', 'sf', 'shall', 'she', 'shed', 'should', 'shouldn', 'show', 'showed', 'shown', 'shows', 'si', 'side', 'significant', 'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'sp', 'specifically', 'sq', 'st', 'still', 'stop', 'strongly', 'sub', 'successfully', 'such', 'sup', 'sure', 'system', 't', 'take', 'taken', 'taking', 'tell', 'ten', 'tends', 'th', 'than', 'thank', 'thanks', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'therefore', 'thereof', 'theres', 'these', 'they', 'thin', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'tv', 'twenty', 'twice', 'two', 'u', 'uk', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'until', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefulness', 'uses', 'using', 'usually', 'v', 'value', 'various', 've', 'very', 'via', 'vs', 'w', 'want', 'wants', 'was', 'wasn', 'wasnt', 'way', 'we', 'welcome', 'well', 'went', 'were', 'weren', 'what', 'whatever', 'when', 'whenever', 'where', 'wherever', 'whether', 'which', 'while', 'whim', 'who', 'whoever', 'whole', 'whom', 'why', 'wi', 'will', 'willing', 'wish', 'with', 'within', 'won', 'wonder', 'wont', 'words', 'world', 'would', 'wouldn', 'wouldnt', 'x', 'yet', 'you', 'your', 'youre', 'yours', 'yourself', 'yr']

def rem_stop_words(wordlist):
    refined_words=[]
    for word in wordlist:
        if word not in stop_words:
            refined_words.append(word)
    return(refined_words)

def data_tokenization_prior(p_d_prior,p_t_prior,n_d_prior,n_t_prior):
    p_d_token=m.log(p_d_prior)
    p_t_token=m.log(p_t_prior)
    n_d_token=m.log(n_d_prior)
    n_t_token=m.log(n_t_prior)
    return(p_d_token,p_t_token,n_d_token,n_t_token)

def train_tokenization(train_read_data):
    data_store={}
    for text in train_read_data:
        (data1,data2)=text.split('=')
        data_store[data1.strip()]=data2.strip()
    return(data_store)

def test_tokenization(text_file):
    text_file=text_file.lower()
    text_file=re.sub(r"[^a-zA-Z0-9]+",' ',text_file)
    text_file_ds=text_file.split()
    text_data_word=rem_stop_words(text_file_ds)
    return(text_data_word)

def nbclassifition_algorithm(p_d,p_t,n_d,n_t):
    if(p_d>p_t and p_d>n_d and p_d>n_t):
        return(['deceptive','positive'])
    elif(n_d>p_t and n_d>n_t and n_d>p_d):
        return(['deceptive','negative'])
    elif(n_t>p_t and n_t>n_d and n_t>p_d):
        return(['truthful','negative'])
    else:
        return(['truthful','positive'])

def get_prior_prob(data_store):
    p_d_prior_prob=data_store.get('p_d_prior_prob')
    p_t_prior_prob=data_store.get('p_t_prior_prob')
    n_d_prior_prob=data_store.get('n_d_prior_prob')
    n_t_prior_prob=data_store.get('n_t_prior_prob')
    return(p_d_prior_prob,p_t_prior_prob,n_d_prior_prob,n_t_prior_prob)

def get_conditional_prob(data_store): 
    p_d_conditional_prob=te.literal_eval(data_store.get('p_d_conditional_prob'))
    p_t_conditional_prob=te.literal_eval(data_store.get('p_t_conditional_prob'))
    n_d_conditional_prob=te.literal_eval(data_store.get('n_d_conditional_prob'))
    n_t_conditional_prob=te.literal_eval(data_store.get('n_t_conditional_prob'))
    return( p_d_conditional_prob,p_t_conditional_prob,n_d_conditional_prob,n_t_conditional_prob)

def test_data_find():
    test_data_file=sys.argv[1]
    temp1={}
    temp=[]
    os_path=os.path.join(test_data_file,'*/*/*/*.txt')
    data_collect=g.glob(os_path)
    for data in data_collect:
        for text in data.split('\n'):
            text.strip()
            eq=text.find("=")
            temp.append(eq)
        temp1[data[:]]=temp
    return(data_collect)

def token_punctuation(text):
    data_pun={}
    for pun in string.punctuation:
        data_pun[pun]=""
    return(text.translate(str(data_pun.keys())))

def solve_prob(hotel_review,prob_result1,prob_result2):
    temp=prob_result1
    for text in hotel_review:
        if text in prob_result2.keys():
            temp=temp+prob_result2.get(text)
        else:
            temp=temp+prob_result2.get(st)
    return(temp)

def features_refine(hotel_review_rext):
    features=[]
    for text in hotel_review_rext.split():
        if(text.lower() and text not in stop_words and not text.isalpha()):
            features.append(text)
    return(features)


def main():
    train_file=open("nbmodel.txt",'r')
    train_read_data=train_file.readlines()
    data_store=train_tokenization(train_read_data)
    p_d_prior_prob,p_t_prior_prob,n_d_prior_prob,n_t_prior_prob=get_prior_prob(data_store)
    p_d_conditional_prob,p_t_conditional_prob,n_d_conditional_prob,n_t_conditional_prob=get_conditional_prob(data_store)
    test_file=open("nboutput.txt",'w')
    data_collect=test_data_find()
    for data in data_collect:
        with open(data,'r') as folder_file:
            text_file=folder_file.readline()
        text_data_word=test_tokenization(text_file)
        p_d=solve_prob(text_data_word,float(p_d_prior_prob),p_d_conditional_prob)
        p_t=solve_prob(text_data_word,float(p_t_prior_prob),p_t_conditional_prob)
        n_d=solve_prob(text_data_word,float(n_d_prior_prob),n_d_conditional_prob)
        n_t=solve_prob(text_data_word,float(n_t_prior_prob),n_t_conditional_prob)
        result=nbclassifition_algorithm(p_d,p_t,n_d,n_t)
        text_data_word=test_tokenization(text_file)
        test_file.write(result[0]+' '+result[1]+' '+data+'\n')
    test_file.close()

if __name__ == "__main__":
    main()

