import pandas as pd
import pickle
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from itertools import chain

import spacy
spacy_model_sm = spacy.load('en_core_web_sm')
#Stopwords

from nltk.corpus import stopwords
stopwords_file = list(w.rstrip() for w in open('./static/model/stopwords.txt'))
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can', 'defendants', 'defendant']
stopwords_other = ['one', 'mr', 'ms', 'introduction', 'background', 'lin', 'lee', 'koh', 'amin', 'leong', 'chua', 'lun', 'tan', 'wang', 'loh', 'yeoh', 'ng', 'lau']
my_stopwords = stopwords.words('English') + stopwords_verbs + stopwords_other + stopwords_file
stopwords = list(set(my_stopwords))

from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
lemmatizer = WordNetLemmatizer()

def removeNER(a, b):
    wordlist = []
    for token in a:
        if token.lower() not in b:
            wordlist.append(token.lower())
    return wordlist

def gettokenname_spacy(rawtext):
    namelist = []
    fullname = []
    doc = spacy_model_sm(rawtext)
    for x in doc.ents:
        if (x.label_ == 'PERSON') and (x.text.lower() not in fullname) and (x.text.lower() not in namelist):
            fullname.append(x.text.lower())
            #print(x.text.lower())
            for name in x.text.split():
                if name.lower() not in namelist:
                    namelist.append(name.lower())
    return(doc, namelist)

#Lemmatized Words
def lemmatizeprocess(word, postag = 'v'):
    morphyword = wordnet._morphy(word, pos=postag)
    #words that contain more than 1 meaning
    if len(morphyword) > 1:
        if len(morphyword[0]) - len(morphyword[1]) == 1:
            #If the 1st morphyword is a plural version of itself
            if (morphyword[0][ len(morphyword[0]) - 1 ] == 's'):
                #Get the 2nd morphyword
                morphyword[0] = morphyword[1]
        #print('there is', len(morphyword), 'meaning for:', word)
        #print(morphyword[0], 'other meaning is:', morphyword[1])
    return(morphyword[0])

def get_verb_noun_lemma(word):
    trfmword = wordnet._morphy(word, pos='v')
    
    #Check if word is a noun
    if len(trfmword) == 0:
        trfmword = wordnet._morphy(word, pos='n')
        
        #Check if a word is a verb
        if len(trfmword) == 0:
            lemmaword = ''
        else:
            lemmaword = lemmatizeprocess(word, 'n')
    else:
        #print(trfmword)
        lemmaword = lemmatizeprocess(word, 'v')
    return(lemmaword)

def df_manipulation(dataframe, ner = False):
    if ner:
        dataframe['nameentity'] = [ner]
    else:
        #Get name entities
        dataframe['nameentity'] = dataframe['RawText'].map(lambda x: gettokenname_spacy(x)[1])
        #Saving name entities for later
        detected_name_entities = [name for row in dataframe['nameentity'].tolist() for name in row]
        detected_name_entities = list(set(detected_name_entities))
        with open("./static/model/detected_name_entities_file.txt", "wb") as detected_name_entities_file:
            pickle.dump(detected_name_entities, detected_name_entities_file)
        print('Finished extracting name entities')

    #Convert Raw Text to Sentences 
    dataframe['tokens3'] = dataframe['RawText'].map(sent_tokenize)
    print('Finished converting raw text to sentences')

    #Tokenizing Sentences
    dataframe['tokens3'] = dataframe['tokens3'].map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    print('Finished tokenizing sentences')

    #POS_tagging words
    dataframe['tokens3'] = dataframe['tokens3'].map(lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])
    print('Finished POS tagging words')

    #Lemmatizing each word by its POS Tag
    dataframe['tokens3'] = dataframe['tokens3'].map(
        lambda list_tokens_POS: [
            [
                lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
                if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
            ] 
            for tokens_POS in list_tokens_POS
        ]
    )
    print('Finished 1st lemmatizing')

    #Tokenizing
    dataframe['tokens3'] = dataframe['tokens3'].map(lambda sentences: list(chain.from_iterable(sentences)))
    #Cleaning Tokens
    dataframe['tokens3'] = dataframe['tokens3'].map(lambda tokens: [token.lower() for token in tokens if 
                                                                    token.isalpha() and 
                                                                    token.lower() not in stopwords and 
                                                                    len(token)>2 and
                                                                    not token.isdigit()])
    print('Finished cleaning')

    #Removing name entities from tokens
    dataframe['nonametoken'] = dataframe.apply(lambda row:removeNER(row['tokens3'], row['nameentity']), axis = 1)

    #Get & lemmatized verbs and nouns
    dataframe['tokens3'] = dataframe['nonametoken'].map(lambda words: [get_verb_noun_lemma(word) for word in words])
    print('Finished extracting verbs and nouns')

    dataframe = dataframe.drop(columns = {'nonametoken'})

    return(dataframe)