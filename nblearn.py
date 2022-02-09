import os
import sys
import glob as g
import re
import numpy as np
from collections import Counter
import math as m
import pandas as pd

global stop_words
stop_words=['ad','added','ae','affected','after','afterwards','again','against','ah','al','all','allow','allows','hotel','room','wife','wifes','wives','husband','husbands', 'a', 'amongst', 'amoungst', 'becomes', 'eg', 'fify', 'formerly', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'latterly', 'ltd', 'namely', 'nevertheless', 'sixty', 'thence', 'thereafter', 'thereby', 'therein', 'thereupon', 'thickv', 'twelve', 'whence', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'whither', 'whose', 'yourselves', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'act', 'actually','almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'are', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'au', 'av', 'away', 'b', 'back', 'be', 'became', 'because', 'become', 'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'bottom', 'brief', 'briefly', 'but', 'by', 'c', 'came', 'can', 'cannot', 'cant', 'cause', 'cd', 'certain', 'certainly', 'changes', 'clearly', 'co', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'could', 'couldn', 'couldnt', 'course', 'cry', 'currently', 'd', 'date', 'dc', 'de', 'definitely', 'describe', 'described', 'despite', 'detail', 'did', 'didn', 'different', 'dj', 'do', 'does', 'doesn', 'doing', 'don', 'done', 'down', 'dr', 'due', 'during', 'e', 'each', 'effect', 'eight', 'either', 'el', 'eleven', 'else', 'elsewhere', 'em', 'end', 'enough', 'entirely', 'especially', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'few', 'fi', 'fifteen', 'fifth', 'fill', 'find', 'fire', 'first', 'five', 'fix', 'fl', 'followed', 'following', 'for', 'former', 'forth', 'forty', 'found', 'four', 'from', 'front', 'ft', 'full', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', 'hadn', 'happens', 'has', 'hasn', 'hasnt', 'have', 'haven', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'herself', 'hi', 'hid', 'him', 'himself', 'his', 'ho', 'hopefully', 'how', 'however', 'hr', 'http', 'hundred', 'i', 'ie', 'if', 'il', 'im', 'immediate', 'immediately', 'important', 'in', 'inc', 'indeed', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'instead', 'interest', 'into', 'ip', 'is', 'isn', 'it', 'its', 'itself', 'j', 'jr', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'l', 'la', 'largely', 'last', 'lately', 'later', 'latter', 'lb', 'le', 'least', 'les', 'less', 'let', 'lets', 'like', 'likely', 'line', 'little', 'll', 'lo', 'look', 'looking', 'looks', 'los', 'm', 'ma', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'might', 'mill', 'million', 'mine', 'miss', 'ml', 'mo', 'more', 'moreover', 'most', 'mostly', 'move', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'name', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needn', 'needs', 'never', 'new', 'next', 'nine', 'nobody', 'non', 'none', 'nonetheless', 'nor', 'normally', 'noted', 'nothing', 'now', 'nowhere', 'nt', 'ny', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'oj', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'ow', 'own', 'p', 'pages', 'par', 'part', 'particular', 'particularly', 'past', 'pc', 'per', 'perhaps', 'pl', 'placed', 'please', 'plus', 'pm', 'possible', 'possibly', 'potentially', 'present', 'presumably', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'ps', 'put', 'q', 'que', 'quickly', 'quite', 'ran', 'rather', 'rd', 're', 'really', 'reasonably', 'recent', 'recently', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'resulted', 'resulting', 'results', 'right', 'rm', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'se', 'second', 'secondly', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'sent', 'serious', 'seriously', 'seven', 'several', 'sf', 'shall', 'she', 'shed', 'should', 'shouldn', 'show', 'showed', 'shown', 'shows', 'si', 'side', 'significant', 'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'sp', 'specifically', 'sq', 'st', 'still', 'stop', 'strongly', 'sub', 'successfully', 'such', 'sup', 'sure', 'system', 't', 'take', 'taken', 'taking', 'tell', 'ten', 'tends', 'th', 'than', 'thank', 'thanks', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'therefore', 'thereof', 'theres', 'these', 'they', 'thin', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'tv', 'twenty', 'twice', 'two', 'u', 'uk', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'until', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefulness', 'uses', 'using', 'usually', 'v', 'value', 'various', 've', 'very', 'via', 'vs', 'w', 'want', 'wants', 'was', 'wasn', 'wasnt', 'way', 'we', 'welcome', 'well', 'went', 'were', 'weren', 'what', 'whatever', 'when', 'whenever', 'where', 'wherever', 'whether', 'which', 'while', 'whim', 'who', 'whoever', 'whole', 'whom', 'why', 'wi', 'will', 'willing', 'wish', 'with', 'within', 'won', 'wonder', 'wont', 'words', 'world', 'would', 'wouldn', 'wouldnt', 'x', 'yet', 'you', 'your', 'youre', 'yours', 'yourself', 'yr']

def rem_stop_words(wordlist):
    refined_words=[]
    for word in wordlist:
        if word not in stop_words:
            refined_words.append(word)
    return(refined_words)

def data_path_find():
    data_path=os.path.join(sys.argv[1],'*/*/*/*.txt')
    data_file=g.glob(data_path)
    return(data_file)

global tr,dec,pos,neg,l_dic,prob
l_dic={}
def label_counter(label):
    pos,neg,tr,dec=0
    label_extract=l_dic.get(label)
    if(label_extract=="positive"):
        pos=pos+1
    if(label_extract=="deceptive"):
        dec=dec+1
    if(label_extract=="truthful"):
        tr=tr+1
    if(label_extract=="negative"):
        neg=neg+1
    
def learning_data_dict_find(p_d,p_t,n_d,n_t):
    p_d_file=data_dict(p_d)
    p_t_file=data_dict(p_t)
    n_d_file=data_dict(n_d)
    n_t_file=data_dict(n_t)
    return(p_d_file,p_t_file,n_d_file,n_t_file)

def tokenization(hotel_review):
    hotel_review=hotel_review.lower()
    hotel_review=re.sub(r"[^a-zA-Z]+",' ',hotel_review)
    wordlist=hotel_review.split()
    return(wordlist)

def data_dict(label_data):
    count=Counter()
    for data in label_data:
        file=open(data,'r')
        hotel_review=file.read()
        wordlist=tokenization(hotel_review)
        words=Counter(rem_stop_words(wordlist))
        count=count+words
    return(count)

def get_prior_prob(p_d,p_t,n_d,n_t):
    p_d_prior_prob=m.log(len(p_d))-(m.log(len(p_d)+len(p_t)+len(n_d)+len(n_t)))
    p_t_prior_prob=m.log(len(p_t))-(m.log(len(p_d)+len(p_t)+len(n_d)+len(n_t)))
    n_d_prior_prob=m.log(len(n_d))-(m.log(len(p_d)+len(p_t)+len(n_d)+len(n_t)))
    n_t_prior_prob=m.log(len(n_t))-(m.log(len(p_d)+len(p_t)+len(n_d)+len(n_t)))
    return(p_d_prior_prob,p_t_prior_prob,n_d_prior_prob,n_t_prior_prob)

prob={}
def prior_prob(pos,neg,tr,dec):
    prob['positive']=pos/pos+neg
    prob['truthful']=tr/tr+dec
    prob['negative']=neg/pos+neg
    prob['deceptive']=dec/tr+dec
    return(prob)

def learning_data_find(data_file):
    pos_dec_data_file=data_find("positive","deceptive",data_file)
    pos_tr_data_file=data_find("positive","truthful",data_file)
    neg_dec_data_file=data_find("negative","deceptive",data_file)
    neg_tr_data_file=data_find("negative","truthful",data_file)
    return(pos_dec_data_file,pos_tr_data_file,neg_dec_data_file,neg_tr_data_file)

def data_find(learning1,learning2,data_file):
    find_data=[]
    for text_data in data_file:
        if(learning1 in text_data):
            if(learning2 in text_data):
                find_data.append(text_data)
    return(find_data)

def calculate_conditional_prob(words_len,data,prior_prob_data):
    prob_p2=m.log(1)
    con_prob_data={}
    total_p=prior_prob_data+words_len
    prob_p2=m.log(1)
    prob_p=m.log(total_p)
    for key,val in data.items():
        temp=val+1
        data_prob=m.log(temp)
        con_prob_data[key]=data_prob-prob_p
    con_prob_data['words_nan']=prob_p2-prob_p
    return(con_prob_data)

def learning_data_find(data_file):
    pos_dec_data_file=data_find("positive","deceptive",data_file)
    pos_tr_data_file=data_find("positive","truthful",data_file)
    neg_dec_data_file=data_find("negative","deceptive",data_file)
    neg_tr_data_file=data_find("negative","truthful",data_file)
    return(pos_dec_data_file,pos_tr_data_file,neg_dec_data_file,neg_tr_data_file)

def get_cond_prob(p_d,p_t,n_d,n_t):
    total_words=p_d+p_t+n_d+n_t
    prior_prob_data=len(total_words)
    p_d_cond_prob=calculate_conditional_prob(sum(p_d.values()),p_d,prior_prob_data)
    p_t_cond_prob=calculate_conditional_prob(sum(p_t.values()),p_t,prior_prob_data)
    n_d_cond_prob=calculate_conditional_prob(sum(n_d.values()),n_d,prior_prob_data)
    n_t_cond_prob=calculate_conditional_prob(sum(n_t.values()),n_t,prior_prob_data)
    return(p_d_cond_prob,p_t_cond_prob,n_d_cond_prob,n_t_cond_prob)


def main():
    data_file=data_path_find()
    p_d,p_t,n_d,n_t=learning_data_find(data_file)
    p_d_prior,p_t_prior,n_d_prior,n_t_prior=get_prior_prob(p_d,p_t,n_d,n_t)
    p_d,p_t,n_d,n_t=learning_data_dict_find(p_d,p_t,n_d,n_t)
    p_d_con,p_t_con,n_d_con,n_t_con=get_cond_prob(p_d,p_t,n_d,n_t)
    w_file=open("nbmodel.txt",'w')
    w_file.write("p_d_prior_prob="+str(p_d_prior))
    w_file.write("\n")
    w_file.write("p_t_prior_prob="+str(p_t_prior))
    w_file.write("\n")
    w_file.write("n_d_prior_prob="+str(n_d_prior))
    w_file.write("\n")
    w_file.write("n_t_prior_prob="+str(n_t_prior))
    w_file.write("\n")
    w_file.write("p_d_conditional_prob="+str(dict(p_d_con)))
    w_file.write("\n")
    w_file.write("p_t_conditional_prob="+str(dict(p_t_con)))
    w_file.write("\n")
    w_file.write("n_d_conditional_prob="+str(dict(n_d_con)))
    w_file.write("\n")
    w_file.write("n_t_conditional_prob="+str(dict(n_t_con)))
    w_file.write("\n")

if __name__ == "__main__":
    main()