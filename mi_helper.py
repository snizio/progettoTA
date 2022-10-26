import nltk
from nltk.corpus import stopwords
from string import punctuation
import math

def vectors_creator(data_dict, normalize = True):

    stops = stopwords.words('english')
    tk = nltk.tokenize.TweetTokenizer()
    lemmatizer = nltk.WordNetLemmatizer()
    global_tokens = []
    global_bigrams = []
    for k in data_dict:
        text = data_dict[k]["text"]
        sents = nltk.tokenize.sent_tokenize(text, "english")
        tokens_per_sents = []
        total_tokens = []
        for sent in sents:
            tokens = tk.tokenize(sent)
            lower_tokens = [w.lower() for w in tokens]
            clean_tokens = [lemmatizer.lemmatize(w) for w in lower_tokens if w not in punctuation and w not in stops and len(w)>2]
            tokens_per_sents.append(clean_tokens)
            total_tokens.extend(clean_tokens)
            global_tokens.extend(clean_tokens)

        data_dict[k]["tokens"] = total_tokens
        data_dict[k]["sents"] = tokens_per_sents
        freq = nltk.FreqDist(total_tokens)
        data_dict[k]["freq"] = freq

        total_bigrams = []
        for sent in tokens_per_sents:
            bigrams = list(nltk.ngrams(sent, 2))
            total_bigrams.extend(bigrams)
            global_bigrams.extend(bigrams)

        data_dict[k]["bigrams"] = total_bigrams
        data_dict[k]["bigrams_freq"] = nltk.FreqDist(total_bigrams)

    word_scores = {}
    bigram_scores = {}
    global_tokens_freq = nltk.FreqDist(global_tokens)
    global_bigrams_freq = nltk.FreqDist(global_bigrams)

    for k in data_dict:
        word_scores[k] = {}
        bigram_scores[k] = {}
        
        for w in set(data_dict[k]["tokens"]):
            if global_tokens_freq[w] > 3:
                p = data_dict[k]["freq"][w]/len(data_dict[k]["tokens"])
                p_category = data_dict[k]["freq"][w]/global_tokens_freq[w]
                h = -math.log2(p)
                score = h * p_category * p
                word_scores[k][w] = score

        for bigram in set(data_dict[k]["bigrams"]):
            p = data_dict[k]["bigrams_freq"][bigram]/len(data_dict[k]["bigrams"])
            p_category = data_dict[k]["bigrams_freq"][bigram]/global_bigrams_freq.get(bigram, 1)
            h = -math.log2(p)
            score = p_category * p
            bigram_scores[k][bigram] = score

    if normalize:
        
        for label in word_scores:
            values_list = list(word_scores[label].values())
            min_values = min(values_list)
            max_values = max(values_list)
            for k in word_scores[label]:
                val = word_scores[label][k]
                word_scores[label][k] = (val- min_values)/(max_values-min_values)

        for label in bigram_scores:
            values_list = list(bigram_scores[label].values())
            min_values = min(values_list)
            max_values = max(values_list)
            for k in bigram_scores[label]:
                val = bigram_scores[label][k]
                bigram_scores[label][k] = (val- min_values)/(max_values-min_values)
            
    return word_scores, bigram_scores

def predict(test_list, trained_dic_w, trained_dic_bi, add_bigram = True, only_bigram = False):

    tk = nltk.tokenize.TweetTokenizer()
    predicted_list = []
    average_positive_len = trained_dic_w
    average_negative_len = 0

    for review in test_list:
        tokens = tk.tokenize(str(review))
        lemmatizer = nltk.WordNetLemmatizer()
        stops = stopwords.words('english')
        lower_tokens = [w.lower() for w in tokens]
        clean_tokens = [lemmatizer.lemmatize(w) for w in lower_tokens if w not in punctuation and w not in stops and len(w)>2]
        freq = nltk.FreqDist(clean_tokens)
        bigrams = list(nltk.ngrams(clean_tokens, 2))
        bigrams_freq = nltk.FreqDist(bigrams)

        score = {}
        if only_bigram:
            for k in trained_dic_bi:
                score[k] = 0
                for bi in bigrams:
                    val = trained_dic_bi[k].get(bi, 0)
                    #h = - math.log2(bigrams_freq[bi]/len(bigrams))
                    score[k]+=val
        else:
            for k in trained_dic_w:
                score[k] = 0
                for w in clean_tokens:
                    val = trained_dic_w[k].get(w, 0)
                    #h = - math.log2(freq[w]/len(clean_tokens))
                    score[k]+= val
            
            if add_bigram:
                for k in trained_dic_bi:
                    for bi in bigrams:
                        val = trained_dic_bi[k].get(bi, 0)
                        # h = - math.log2(bigrams_freq[bi]/len(bigrams))
                        score[k]+=val

        predicted_label = max(score, key=score.get)
        predicted_list.append(predicted_label)
    
    return predicted_list
    
