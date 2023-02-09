
# Il file contiene funzioni di base utilizzate per il preprocessing


import pandas as pd
import nltk
import string
from nltk import pos_tag
from nltk.corpus import wordnet
import re
from gensim.models import Word2Vec
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models import KeyedVectors


custom_stopwords = ['echo', 'alexa', "alexia", 'dot', "star", 'amazon', 'prime', '2nd', 'generation', "fire", "stick", "firestick", "skype", "facetime", '1st', '3rd', '4th', '5th', "hub", "hulu", 'google', 'netflix', "sony", 'youtube', 'philip', 'tp-link', 'fourth', 'roku', "siri", 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "...", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 'wasn', 'weren', 'won', 'wouldn']


def frequency_cleaning(new_sent_tok, n):

    """Rimuove token con frequenza inferiore a n. Usato quando si lavora con tf-idf e word2vec in contemporanea,
    in quanto entrambe le classi hanno già dei metodi per eliminare parolo con una certa frequenza."""
    
    tot_tokens = []

    for sent in new_sent_tok:
        for tok in sent:
            tot_tokens.append(tok)

    freqs = nltk.FreqDist(tot_tokens)
    cleaned_reviews = []

    for sent in new_sent_tok:
        clean_sent = []
        for tok in sent:
            if freqs[tok] > n:
                clean_sent.append(tok)
        cleaned_reviews.append(clean_sent)

    return cleaned_reviews

def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v).
    This was done in order to have compatibility with the wordnet lemmatizer.
    For example the pos "JJ" is transformed in "a".
    """
    if treebank_tag.startswith('J'):
        return "a"
    elif treebank_tag.startswith('V'):
        return "v"
    elif treebank_tag.startswith('N'):
        return "n"
    elif treebank_tag.startswith('R'):
        return "r"
    else:
        return "n"
        

def negation_handler(sentence):	

    """Handle negations using WordNet. See https://github.com/UtkarshRedd/Negation_handling for a clear explenation."""

    temp = int(0)
    lemmatizer = nltk.WordNetLemmatizer()
    for i in range(len(sentence)):
        if sentence[i-1] in ['not',"n't", "no", "without", "t"]: # se il token precedente è una negazione
            w = sentence[i]
            pos_w = pos_tag([w])[0]
            w_lemm = lemmatizer.lemmatize(pos_w[0], get_wordnet_pos(pos_w[1])) # lemmatizzo il token negato
            antonyms = []
            antonym_found = False
            for syn in wordnet.synsets(w_lemm):
                syns = wordnet.synsets(w_lemm)
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name()) # appendo alla lista gli antomi trovati
                max_dissimilarity = 0
                for ant in antonyms: # per ogni antonimo calcolo la dissimilarità
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(w_lemm)
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity: # se la dissimilarità è più alta della precedente allora sostituisco l'antonimo
                        # all'espressione negata.
                        # Il processo è ripetututo per ogni synset della parola negata
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
                        antonym_found = True
            if not antonym_found:
                sentence[i] = f"not_{w_lemm}"
                sentence[i-1] = ''

    while '' in sentence:
        sentence.remove('')
    return sentence


def tokenize_list_of_text(list_of_text, custom_stopwords = [], pos_tagging = False, token_len = 2):
    """Tokenizza tutte le recensioni, pulisce da stopwords, elimina token <= token_len caratteri e lemmatizza.
    Ritorna sia il la lista tokenizzata ma come stringa sia come lista di tokens, dunque ritorna due elementi"""

    lemmatizer = nltk.WordNetLemmatizer() 
    detokenizer = TreebankWordDetokenizer()

    tokenized_reviews = []
    sent_tokenized_reviews = []
    for review in list_of_text: #pulisce le recensioni
        review = re.sub(r'\d+', '', review) # elimina i numeri
        review = re.sub(r'[^\w\s\']+', ' ', review) # sostituisce la punteggiatura con uno spazio
        tokens = nltk.tokenize.word_tokenize(review, language='english', preserve_line=False) # tokenizza
        tokens = [w.lower() for w in tokens] # mette in minuscolo
        tokens = negation_handler(tokens) # gestisce le negazioni, vedi funzione
        tokens_pos = pos_tag(tokens) # pos tagging
        lemmatized_tokens = [(lemmatizer.lemmatize(w, get_wordnet_pos(pos)), pos) for w, pos in tokens_pos] # lemmatizza in base al pos tag
        if pos_tagging: # se si vuole filtrare per la lista di pos
            clean_tokens = [f"{w}_{pos}" for w, pos in lemmatized_tokens if w not in string.punctuation and len(w)>token_len and w not in custom_stopwords]
        else:
            clean_tokens = [w for w, pos in lemmatized_tokens if w not in string.punctuation and len(w)>token_len and w not in custom_stopwords]
        if clean_tokens == []:
            clean_tokens = tokens
        sent_tokenized_reviews.append(clean_tokens)
        tokenized_reviews.append(detokenizer.detokenize(clean_tokens))
    
    # questo viene fatto solo per estrarre il numero di parole tipo estratte
    n_tokens = []
    for sent in sent_tokenized_reviews:
        for w in sent:
            n_tokens.append(w)
    print("total number of types extracted is:", len(set(n_tokens)))

    return tokenized_reviews,  sent_tokenized_reviews # ritorna una tupla, il primo elemento contiene le recensioni in formato stringa, il secondo le continene tokenizzate


def generate_samples(reviews, n, pre_trained_model):
    
    """Genera recensioni sintetiche. Ciò avviene sostituendo i token delle recensioni in input con i sinonimi più coerenti.
    Prende in input le recensioni tokenizzate e il numero di recensioni da generare (deve essere minore del numero di recensioni in input).
    La generazione usa wordnet per l'estrazione dei sinonimi e addestra un modello di w2v per calcolare la precisione dei sinonimi."""
    
    import random
    from nltk.corpus import wordnet as wn

    random.seed(10)
    reviews_to_sample = random.sample(reviews, n) # si fa un sample di n recensioni
    generated_reviews = []

    w2v = pre_trained_model.wv

    for review in reviews_to_sample:
        new_review = []
        for w in review:
            syns = wn.synsets(w, lang = "eng") # estrae i synsets del token su cui stiamo ciclando
            synonyms = [] 
            for syn in syns: # per ogni synset 
                synonyms.extend(syn.lemma_names()) # appende i sinonimi

            # per ogni sinonimo viene calcolata la similarità con il token originale e si crea un dizionario della forma {sinonimo : score}
            sim_score = {}
            for word_sim in synonyms:
                if word_sim != w and word_sim != "" and "_" not in word_sim: # se la parola è diversa dalla iniziale, se non è vuota e se non è un n-gramma
                    try:
                        score = w2v.similarity(w, word_sim) # lo score è calcolato come la cosine similarity fra il sinonimo e il token originale
                        if score > 0.20: # soglia di similarità per essere considerato sinonimo
                            sim_score[word_sim] = score
                    except KeyError:
                        continue
            
            if sim_score: # se il dizionario è popolato
                new_review.append(max(sim_score, key=sim_score.get)) #si prende il sinonimo col punteggio maggiore
            else:
                new_review.append(w) #altrimenti la parola originale

        if new_review and new_review != review:
            print(review)
            print("-")
            print(new_review)
            print("--------------------------")
            generated_reviews.append(new_review)

    return generated_reviews


