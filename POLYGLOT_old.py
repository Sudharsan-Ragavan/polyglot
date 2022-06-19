#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install pyttsx3


# In[2]:


import pyttsx3

# pyttsx3 package is imported to that our text data can be coverted to voice so POLYGLOT can speak
engine = pyttsx3.init()
# constuctor is created and the methods of pyttsx3 can be accessed using this engine
# print("hi hello")
# engine.say("hi hello")
# engine.runAndWait()
# this the sample code to see the basic working of the pyuttsx3 package


# #this is the package which is used to covert input string to audio output
# #this package is python's test to speech coverter 
# #this package has many voice and this will give accurate pronounciation of the word according to the situation
# #in which the word is encountered

# In[3]:


import speech_recognition as sr

r = sr.Recognizer()

# In[4]:


# pip install wikipedia


# #this is the package which will give a wikipedia webpage related to given topic 
# #This package has many methods like giving direct webpage or it can produce summary of the page for given number of lines
# #This package need internet connection but this will provide the user a good featur to now general thing while learning a concept

# In[5]:


# pip install translate


# #this package is used translate words, phrases or line of a given language to desired language
# #we can change both from language and to language so that it is versatile 
# #this package helps to give our user a good feature to app 
# #so that interaction user can be increased as chatbot can also translate

# In[6]:


import wikipedia
# package wikipedia is imported so that summarization of a topoic could be done

from translate import Translator

# method called translator is imported from translate package so that functions to translate from and different
# languages can be done

translatorta = Translator(to_lang="ta")
# translatorta, a method created to translate string from default language english to target language tamil
# print(translatorta.translate("basketball"))
# this code shows how the code for traslation works for language tamil
translatores = Translator(to_lang="es")
# translatores, a method created to translate string from default language english to target language spanish
# print(translatores.translate("basketball"))
# this code shows how the code for traslation works for language spanish
translatorja = Translator(to_lang="ja")
# translatorta, a method created to translate string from default language english to target language Japanese
# print(translatorja.translate("basketball"))
# this code shows how the code for traslation works for language japanese
translatorfr = Translator(to_lang="fr")
# translatorta, a method created to translate string from default language english to target language French
# print(translatorfe.translate("basketball"))
# this code shows how the code for traslation works for language french


# In[7]:


# # Meet POLYGLOT : your virtual language assistant
import nltk
# package called nltk is imported so that we can do natural language processing(NLP) like tokenize the sentence
import warnings

warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages
import numpy as np
import random
import string

# to process standard python strings

# For our example,we will be using the Wikipedia page for chatbots as our corpus.
# Copy the contents from the page and place it in a text file named ‘chatbot.txt’.
# However, you can use any corpus of your choice.
# We will read in the chatbot.txt file
f = open('chatbot.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase

nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only

# convert the entire corpus into a list of sentences and a list of words for further pre-processing
sent_tokens = nltk.sent_tokenize(raw)
# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)
# converts to list of words

sent_tokens[:2]
word_tokens[:5]

# sent_tokens[0]

# word_tokens[:5]


# WordNet is a semantically-oriented dictionary of English included in NLTK.
lemmer = nltk.stem.WordNetLemmatizer()


# LemTokens will take as input the tokens and return normalized tokens.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# Checking for greetings
# define a function for a greeting by the bot i.e if a user’s input is a greeting,
# the bot shall return a greeting response.
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# the words need to be encoded as integers or floating point values
# for use as input to a machine learning algorithm, called feature extraction (or vectorization).
from sklearn.feature_extraction.text import TfidfVectorizer

# find the similarity between words entered by the user and the words in the corpus.
# This is the simplest possible implementation of a chatbot.
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
# define a function response which searches the user’s utterance for one or more known keywords
# and returns one of several possible responses. If it doesn’t find the input matching any of the keywords,
# it returns a response:” I am sorry! I don’t understand you”
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    # TF-IDF are word frequency scores that try to highlight words that are more interesting,
    # e.g. frequent in a document but not across documents.
    # The TfidfVectorizer will tokenize documents, learn the vocabulary and
    # inverse document frequency weightings, and allow you to encode new documents.
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')

    # Learn vocabulary and idf, return term-document matrix
    # Returns X : Tf-idf-weighted sparse matrix, [n_samples, n_features]
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # print (tfidf.shape)

    # Cosine similarity is a measure of similarity between two non-zero vectors.
    # Using this formula we can find out the similarity between any two documents d1 and d2.
    # Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||
    vals = cosine_similarity(tfidf[-1], tfidf)

    # function is used to perform an indirect sort along the given axis using the algorithm
    # specified by the kind keyword. It returns an array of indices of the same shape as arr
    # that would sort the array.
    idx = vals.argsort()[0][-2]

    # Returns a new array that is a one-dimensional flattening of this array (recursively).
    # That is, for every element that is an array, extract its elements into the new array.
    # If the optional level argument determines the level of recursion to flatten.
    flat = vals.flatten()

    flat.sort()
    # flat is sorted
    req_tfidf = flat[-2]
    # second element from reverse order is taken as req_tfidf

    if req_tfidf == 0:
        # check if req_tfidf is zero if yes then don the esceptional handling for wikipedia
        try:
            robo_response += wikipedia.summary(user_response, 4)
        except wikipedia.DisambiguationError as e:
            robo_response = robo_response + "I am sorry! Please enter detailed query."
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


flag = True
print("""POLYGLOT: I am POLYGLOT. I am your personal language assistant.
         As i know many things please enter your query precisely.
         If you want to exit, say Bye!""")
engine.say("""I am POLYGLOT. I am your personal language assistant.
As i know many things please enter your query precisely.
If you want to exit, type Bye!""")
engine.runAndWait()
# Intro statement for our chatbot this is the first output statement from ouyr side to the user

while flag:

    with sr.Microphone() as source:
        print("User:", end="")
        audio = r.listen(source)
        try:
            user_response = r.recognize_google(audio)
        except:
            print()
            print("POLYGLOT: I can't recognize what you said, please speak clearly.")
            engine.say("I can't recognize what you said, please speak clearly.")
            engine.runAndWait()
            continue
    user_response = user_response.lower()
    if user_response == 'bhai':
        user_response = 'bye'
    print(user_response)
    if user_response != ('bye'):
        if "translate" in user_response:
            if "tamil" in user_response:
                xy = len(user_response)
                translation = translatorta.translate(user_response[10:xy - 8])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
            if "spanish" in user_response:
                xy = len(user_response)
                translation = translatores.translate(user_response[10:xy - 10])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
            if ("japanese" in user_response):
                xy = len(user_response)
                translation = translatorja.translate(user_response[10:xy - 11])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
            if "french" in user_response:
                xy = len(user_response)
                translation = translatorfr.translate(user_response[10:xy - 10])
                print("POLYGLOT:", translation)
                engine.say(translation)
                engine.runAndWait()
        elif user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("POLYGLOT: You are welcome..")
            engine.say("you are welcome")
            engine.runAndWait

        else:
            if greeting(user_response) is not None:
                print("POLYGLOT: " + greeting(user_response))
                engine.say(greeting(user_response))
            else:
                print("POLYGLOT: ", end="")
                print(response(user_response))

                sent_tokens.remove(user_response)
                engine.say(response(user_response))
                engine.runAndWait()
    else:
        flag = False
        print("POLYGLOT: Bye! take care..")
        engine.say("Bye! take care..")
        engine.runAndWait()

# In[ ]:


# In[ ]:
