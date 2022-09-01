# Pre-trained Transformers, Information Extraction and Dialogue System

This project focuses mostly on the implementation of large pre-trained language models. The techniques implemented are used on these domains - information extraction, coreference resolution and dialogue systems.

### Pre-trained BERT
A pre-trained BERT model is turned into a trainable keras layer and applied to Aspect-Based Sentiment Analysis.
* Preprocessing and Tokenization:
    * DistilBERT tokenizer is used
    * Text and topics of tweet are converted to integers after tokenizing and then labels are one-hot encoded and labels are converted to numbers.
* Model 1: Prebuilt Sequence Classification:
    * A sequence classification model based on distillBERT is used.
* Model 2: Neural bag of words using BERT
* Model 3: LSTM with BERT

### Information Extraction 1: Training a Named Entity Resolver
The following things are performed here:
* A two layer Bidirectional GRU and Multi-layer FFNN is created to compute the ner scores for individual tokens.
* The predictions of NER from the IO label is created.

### Information Extraction 2: A Coreference Resolver for Arabic
A coreference system based on the mention-ranking algorithm is built here.
* Embedding dictionary is created first. To prepare the each dataset for the coreference resolution model, variables are created from each document
    * Getting the mentions from the clusters.
    * Turning the sentences into embeddings and the mention indices into vectors.
    * Generating Mention Pairs.
* During pre-processing since it is arabic language, diacritics are also removed to improve overall performance.
* Coreference resolution model is then built by using mention pair classification model. Document is encoded using Bidirectional LSTMs. Finally, Multilayer feed-forward neural network is created to compute the mention-pair scores.
* Coreference Resolution models are evaluated by building coreference clusters and then these clusters are evaluated using CONLL score.

### Dialogue 1: Dialogue Act Tagging
Two different DA classification models are used here.
* Model 1 has arhitecture as - Embedding layer, BLSTM layers, Fully Connected (Dense) layer, Softmax activation
* Model 2 - Balanced Network
    * As the dataset is highly imbalanced, minority classes are weighted up proportionally to their underrepresentation while training.
* Using Context for Dialog Act Classification
    * We expect there is valuable sequential information among the DA tags. So in this section we apply a BiLSTM on top of the sentence CNN representation. The CNN model learns textual information in each utterance for DA classification. Here, we use bidirectional-LSTM (BLSTM) to learn the context before and after the current utterance.
    * This model has architecture as - Word Embedding, CNN, Bidirectional LSTM, Fully-Connected output

### Dialogue 2: A Conversational Dialogue System
* Encoder is implemented which is producing an "output" vector and a "hidden state" vector at each time step. A bidirectional GRU is defined and the embedding is passed into that GRU.
* The decoder with attention is created which allows the decoder to focus on specific parts of the input sequence rather than using the whole set context at each step to deal with information loss. The attention layer is called and GRU is used for decoding.
* The behaviour and the properties of the encoder-decoder network is evaluated.
