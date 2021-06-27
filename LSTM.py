df = pd.read_csv('tweets.csv',sep=',')
#SELECT RELEVNANT COLS
tweet_df = df[['text','airline_sentiment']]

#REMOVE NEUTRAL ROWS
 tweet_df = tweet_df[tweet_df['airline_sentiment']!='neutral']

# CONVERT AIRLINE SENTIMENT TO NUMERIC
sentiment_label = tweet_df.airline_sentiment.factorize()


#PREPROCESSING DATA
tweet= tweet_df.text.values
tokenizer= Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokinizer.word_index)+1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence= pad_sequence(encoded_docs,maxlen = 50)

# BUILD AND COMPILE THE MODEL
class myCallBaclk(callback):
 def on_epoch_end(self,epochs,logs={}):
    if (logs.get('accuracy')>0.95):
       print('cancel')
       self.model.stop_training = True
callbacks = myCallBack()
embedded_vector_length = 32
model = Sequential([
       Embedding(vocab_size,embedding_vector_length,input_length=50),
       SpatialDropout1D(0.25),
       LSTM(50,dropout=0.5,recurrent_dropout=0.5),
       Dropout(0.2);
       Dense(1,activation = 'sigmoid')

])
 model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
 history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2,epoch =10,batch_size =16,callbacks={callbacks})
