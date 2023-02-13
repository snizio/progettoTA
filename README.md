La repository contiene i file per il progetto del corso di Text Analytics 2022/2023.

Descrizione dei file:

- "amazon_alexa.tsv": il file contiene il dataset di recensioni Amazon Alexa sul quale è stata fatta sentiment analysis.

- "DataUnderstanding.ipynb": il file contiene un'analisi quantitativa del dataset.

- "review_scraper" e "scraping_book": i file contengono le funzioni necessarie per lo scraping di recensioni Amazon.

- "data_for_pt.csv": questo file contiene un altro dataset di circa 20000 recensioni Amazon Alexa estratto da noi facendo scraping. Questo ci ha permesso di usufruire di embedding pre-addestrati più coerenti al nostro contesto linguistico rispetto ad altri embedding pre-addestrati che abbiamo testato (i.e., google-300, glove_twitter_25, etc...).

- "w2vPreTrained" e "w2vPreTrainedPOS": questi file contengono i modelli Word2Vec addestrati sul dataset "data_for_pt.csv". I parametri sono: window_size = 10, vector_size = 100 e skipgram. Il secondo file contiene gli embedding addestrati sui token annotati da part of speech (i.e., "love_VB")

- "NRC-Emotion-Lexicon-ForVariousLanguages.txt": il file contiene un lexicon usato nella parte non supervisionata.

- "SelectKBest.ipynb": il file implementa un modello Count, con feature selection tramite K squared, TF-IDF, seguito dai classificatori SVM e MNB.

- "w2v_svm.ipynb": il file implementa un modello word2vec addestrato sul dataset "amazon_alexa.tsv". Vengono vettorizzate le recensioni tramite media ponderata dei vettori pesati con i valori TF-IDF. Il classificatore usato è un SVM.

- "NN.ipynb": il file implementa un semplice multi-layer perceptron. Per vettorizzare le recensioni viene usata la stessa metodologia descritta in "w2v_svm.ipynb".

- "CNN.ipynb": il file implementa un'architettura CNN. Le recensioni vengono vettorizzate tramite gli id dei token (Keras Tokenizer) e tecniche di padding.

- "CNNpt.ipynb": il file implementa un'architettura CNN. Vengono in questo caso utilizzati i vettori pre-addestrati presenti nel file "w2vPreTrained".

- "BERT+Keras": il file contiene il fine-tuning di BERT collegato a un perceptron per la classificazione.

- "w2vNonSup.ipynb": il file contiene un classificatore non supervisionato che sfrutta lo spazio vettoriale per definire dei centroidi del sentimento positivo e negativo. Si usano gli embedding pre-addestrati.

- "Lessici.ipynb": il file implementa l'utilizzo di lessici per la classificazione non supervisionata.

