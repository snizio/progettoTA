import pandas as pd
import nltk
import string
from nltk import pos_tag
from nltk.corpus import wordnet
import re
from gensim.models import Word2Vec
from nltk.tokenize.treebank import TreebankWordDetokenizer

custom_stopwords = ['echo', 'alexa', "alexia", 'dot', "star", 'amazon', 'prime', '2nd', 'generation', "fire", "stick", "firestick", "skype", "facetime", '1st', '3rd', '4th', '5th', "hub", "hulu", 'google', 'netflix', "sony", 'youtube', 'philip', 'tp-link', 'fourth', 'roku', "siri", 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "...", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
pos_list = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] # lista di pos utilizzate per eventualmente filtrare durante la tokenizzazione

def negation_handler(sentence):	

    """Handle negations using WordNet. See https://github.com/UtkarshRedd/Negation_handling for a clear explenation."""

    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i-1] in ['not',"n't", "no", "without"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp>max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
    while '' in sentence:
        sentence.remove('')
    return sentence



def undersample_positive(dataset):

    """Bilancia il dataset uniformando il numero di recensioni negative e positive. Prende in input il dataset e ritorna il dataset bilanciato."""

    reviews_1 = list(dataset[dataset["feedback"] == 1]["verified_reviews"])
    reviews_0 = list(dataset[dataset["feedback"] == 0]["verified_reviews"])
    reviews_1.sort(key=len, reverse = True)
    sample_1 = reviews_1[:len(reviews_0)]
    verified_reviews_sample = []
    feedback_sample = []
    verified_reviews_sample.extend(sample_1)
    verified_reviews_sample.extend(reviews_0)
    feedback_sample.extend([1 for i in range(len(sample_1))])
    feedback_sample.extend([0 for i in range(len(reviews_0))])
    dataset = pd.DataFrame({"verified_reviews":verified_reviews_sample, "feedback": feedback_sample})
    print(dataset["feedback"].value_counts())

    return dataset


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
        

def tokenize_list_of_text(list_of_text, custom_stopwords = [], pos_filter = False, pos_list = []):
    """Tokenizza tutte le recensioni, pulisce da stopwords, elimina token <= 2 caratteri e lemmatizza.
    Ritorna sia il la lista tokenizzata ma come stringa sia come lista di tokens, dunque ritorna due elementi"""

    lemmatizer = nltk.WordNetLemmatizer()
    detokenizer = TreebankWordDetokenizer()

    tokenized_reviews = []
    sent_tokenized_reviews = []
    for review in list_of_text: #pulisce le recensioni
        review = re.sub(r'\d+', '', review) # elimina i numeri
        tokens = nltk.tokenize.word_tokenize(review, language='english', preserve_line=False) # tokenizza
        tokens = [w.lower() for w in tokens] # mette in minuscolo
        tokens_pos = pos_tag(tokens) # pos tagging
        lemmatized_tokens = [(lemmatizer.lemmatize(w, get_wordnet_pos(pos)), pos) for w, pos in tokens_pos] # lemmatizza in base al pos tag
        if pos_filter: # se si vuole filtrare per la lista di pos
            clean_tokens = [w for w, pos in lemmatized_tokens if w not in string.punctuation and len(w)>2 and w not in custom_stopwords and pos in pos_list]
        else:
            clean_tokens = [w for w, pos in lemmatized_tokens if w not in string.punctuation and len(w)>2 and w not in custom_stopwords]
        clean_tokens = negation_handler(clean_tokens) # gestisce le negazioni, vedi funzione
        sent_tokenized_reviews.append(clean_tokens)
        tokenized_reviews.append(detokenizer.detokenize(clean_tokens))
    
    # questo viene fatto solo per estrarre il numero di parole tipo estratte
    n_tokens = []
    for sent in sent_tokenized_reviews:
        for w in sent:
            n_tokens.append(w)
    print("total number of types extracted is:", len(set(n_tokens)))

    return tokenized_reviews,  sent_tokenized_reviews # ritorna una tupla, il primo elemento contiene le recensioni in formato stringa, il secondo le continene tokenizzate

def generate_samples(reviews = [], n = 150):

    """Genera recensioni sintetiche. Ciò avviene sostituendo i token delle recensioni in input con i sinonimi più coerenti."
    Prende in input le recensioni tokenizzate e il numero di recensioni da generare (deve essere minore del numero di recensioni in input).
    La generazione usa wordnet per l'estrazione dei sinonimi e addestra un modello di w2v per calcolare la precisione dei sinonimi."""
    
    import random
    from nltk.corpus import wordnet as wn

    random.seed(10)
    reviews_to_sample = random.sample(reviews, n) # si fa un sample di n recensioni
    generated_reviews = []
    # si addestra w2v solo sulle recensioni di input alla funzione
    w2v_model_neg = Word2Vec(reviews, vector_size=300, window = 5, min_count = 0, sg=1, hs = 1, alpha=0.03, min_alpha=0.0007, seed = 5)
    w2v_model_neg.train(reviews, total_examples=len(reviews), epochs=100)

    for review in reviews_to_sample:
        new_review = []
        for w in review:
            syns = wn.synsets(w, lang = "eng") # estrae i synsets
            synonyms = [] 
            for syn in syns:
                synonyms.extend(syn.lemma_names()) # appende i sinonimi

            # per ogni sinonimo viene calcolata la similarità con il token originale e si crea un dizionario della forma {sinonimo : score}
            sim_score = {}
            for word_sim in synonyms:
                if word_sim != w and word_sim != "" and "_" not in word_sim:
                    try:
                        score = w2v_model_neg.wv.similarity(w, word_sim)
                        sim_score[word_sim] = score
                    except KeyError:
                        continue
            
            if sim_score:
                new_review.append(max(sim_score, key=sim_score.get)) #si prende il sinonimo col punteggio maggiore
            else:
                new_review.append(w)

        if new_review:
            generated_reviews.append(new_review)

    return generated_reviews

def drop_positive_duplicates(dataset):

    """Droppa i duplicati delle recensioni positive, insieme alle foto di agosto..."""
        
    df_no_duplicates = dataset.drop_duplicates(subset="verified_reviews")
    index_positive = list(dataset[dataset["feedback"] == 1].index)
    dataset = pd.concat([dataset.drop(labels=index_positive), df_no_duplicates])

    return dataset




# La pipeline standard:
# 1) Droppare le recensioni con 3 stelle in quanto neutre
# 2) Droppare i duplicati nelle recensioni positive con la funzione drop_positive_duplicates()
# 3) tokenize_list_of_text()
# 4) volendo è possibile generare delle recensioni sintetiche
# 5) la parte di rimozione delle parole in base alla frequenza e alle collocazioni(o n_grammi) è affidata ai metodi di sklearn o ad altre librerie usate

