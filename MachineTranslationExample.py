import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import europarl

language_code='da'
mark_start='ssss'
mark_end='eeee'
europarl.maybe_download_and_extract(language_code=language_code)
data_src=europarl.load_data(english=False,language_code=language_code)
data_dest=europarl.load_data(english=True,language_code=language_code,start=mark_start,end=mark_end)
idx=2
print(data_src[idx])
print(data_dest[idx])
num_words=25000

class TokenizerWrap(Tokenizer):
    def __init__(self,texts,padding,reverse=False,num_words=None):
        Tokenizer.__init__(self,num_words=num_words)
        self.fit_on_texts(texts)
        self.index_to_word=dict(zip(self.word_index.values(),self.word_index.keys()))
        self.tokens=self.texts_to_sequences(texts)
        if(reverse):
            self.tokens=[list(reversed(x)) for x in self.tokens]
            truncating='pre'
        else:
            truncating='post'

        self.num_tokens=[len(x) for x in self.tokens]
        self.max_tokens=np.mean(self.num_tokens)+(2*np.std(self.num_tokens))
        self.max_tokens=int(self.max_tokens)
        self.tokens_padded=pad_sequences(self.tokens,maxlen=self.max_tokens,padding=padding,truncating=truncating)

    def token_to_word(self,token):
        word=" " if (token==0) else self.index_to_word[token]
        return word
    def tokens_to_string(self,tokens):
        words=[self.index_to_word[token] for token in tokens if(token!=0)]
        text = " ".join(words)
        return text
    def text_to_tokens(self,text,reverse=False,padding=False):
        tokens=self.texts_to_sequences([text])
        tokens=np.array(tokens)
        if (reverse):
            tokens=np.flip(tokens,axis=1)
            truncating='pre'
        else:
            truncating='post'

        if(padding):
            tokens=pad_sequences(tokens,maxlen=self.max_tokens,padding='pre',truncating=truncating)
        return tokens
    
tokenizer_src=TokenizerWrap(texts=data_src,padding='pre',reverse=True,num_words=num_words)
tokenizer_dest=TokenizerWrap(texts=data_dest,padding='post',reverse=False,num_words=num_words)
tokens_src=tokenizer_src.tokens_padded
tokens_dest=tokenizer_dest.tokens_padded
#np.save('tokenizer_src.npy',tokenizer_src)
#np.save('tokenizer_dest.npy',tokenizer_dest)
#print('Finished loading tokenizer')
print(tokens_src.shape)
print(tokens_dest.shape)
token_start = tokenizer_dest.word_index[mark_start.strip()]
print(token_start)
token_end = tokenizer_dest.word_index[mark_end.strip()]
print(token_end)
#Example of token sequences
idx=2
print(tokenizer_src.tokens_to_string(tokens_src[idx]))
print()
print(data_src[idx])
print()
print(tokenizer_dest.tokens_to_string(tokens_dest[idx]))
print()
print(data_dest[idx])
print()

#Training Data
encoder_input_data=tokens_src
decoder_input_data=tokens_dest[:, :-1]
decoder_output_data=tokens_dest[:, 1:]
print(decoder_input_data[idx])
print()
print(decoder_output_data[idx])

#Creating The Neural Network
#Encoder
encoder_input=Input(shape=(None,),name='encoder_input')
embedding_size=128
encoder_embedding=Embedding(input_dim=num_words,output_dim=embedding_size,name='encoder_embedding')
state_size=512
encoder_gru1=GRU(state_size,name='encoder_gru1',return_sequences=True)
encoder_gru2=GRU(state_size,name='encoder_gru2',return_sequences=True)
encoder_gru3=GRU(state_size,name='encoder_gru3',return_sequences=False)

def connect_encoder():
    net=encoder_input
    net=encoder_embedding(net)
    net=encoder_gru1(net)
    net=encoder_gru2(net)
    net=encoder_gru3(net)
    encoder_output=net
    return encoder_output

encoder_output = connect_encoder()
#Decoder
decoder_initial_state=Input(shape=(state_size,),name='decoder_initial_state')
decoder_input=Input(shape=(None,),name='decoder_input')
decoder_embedding=Embedding(input_dim=num_words,output_dim=embedding_size,name='decoder_embedding')
decoder_gru1=GRU(state_size,name='decoder_gru1',return_sequences=True)
decoder_gru2=GRU(state_size,name='decoder_gru2',return_sequences=True)
decoder_gru3=GRU(state_size,name='decoder_gru3',return_sequences=True)
decoder_dense=Dense(num_words,activation='linear',name='decoder_output')
def connect_decoder(initial_state):
    net=decoder_input
    net=decoder_embedding(net)
    net=decoder_gru1(net,initial_state=initial_state)
    net=decoder_gru2(net,initial_state=initial_state)
    net=decoder_gru3(net,initial_state=initial_state)
    decoder_output=decoder_dense(net)
    return decoder_output

decoder_output=connect_decoder(initial_state=encoder_output)
model_train=Model(inputs=[encoder_input,decoder_input],outputs=[decoder_output])
model_encoder=Model(inputs=[encoder_input],outputs=[encoder_output])
decoder_output=connect_decoder(initial_state=decoder_initial_state)
model_decoder=Model(inputs=[decoder_input,decoder_initial_state],outputs=[decoder_output])

def sparse_cross_entropy(y_true,y_pred):
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss_mean=tf.reduce_mean(loss)
    return loss_mean

optimizer=RMSprop(lr=1e-3)
decoder_target=tf.placeholder(dtype='int32',shape=(None,None))
model_train.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])

#Callbacks
path_checkpoint = '21_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,monitor='val_loss',verbose=1,save_weights_only=True,save_best_only=True)
callback_early_stopping=EarlyStopping(monitor='val_loss',patience=3,verbose=1)
callback_tensorboard=TensorBoard(log_dir='./21_logs/',histogram_freq=0,write_graph=False)
callbacks=[callback_early_stopping,callback_checkpoint,callback_tensorboard]

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

x_data={'encoder_input':encoder_input_data,
        'decoder_input':decoder_input_data}
y_data={'decoder_output':decoder_output_data}
model_train.fit(x=x_data,y=y_data,batch_size=512,validation_split=0.005,callbacks=callbacks)
modelname1='MachineTranslationTrain'
modelname2='MachineTranslationEncoder'
modelname3='MachineTranslationDecoder'
model_train.save('{}.keras'.format(modelname1))
model_encoder.save('{}.keras'.format(modelname2))
model_decoder.save('{}.keras'.format(modelname3))
with open('model_encoder.json', 'w', encoding='utf8') as f:
    f.write(model_encoder.to_json())
model_encoder.save_weights('model_encoder_weights.h5')
with open('model_decoder.json', 'w', encoding='utf8') as f:
    f.write(model_decoder.to_json())
model_decoder.save_weights('model_decoder_weights.h5')
with open('model_train.json', 'w', encoding='utf8') as f:
    f.write(model_train.to_json())
model_train.save_weights('model_train_weights.h5')
#Translate Texts

def translate(input_text,true_output_text=None):
    input_tokens=tokenizer_src.text_to_tokens(text=input_text,reverse=True,padding=True)
    initial_state=model_encoder.predict(input_tokens)
    max_tokens=tokenizer_dest.max_tokens
    shape=(1,max_tokens)
    decoder_input_data=np.zeros(shape=shape,dtype=np.int)
    token_int=token_start
    output_text=''
    count_tokens=0
    while(token_int!=token_end and count_tokens<max_tokens):
        decoder_input_data[0,count_tokens]=token_int
        x_data={'decoder_initial_state':initial_state,
                'decoder_input':decoder_input_data}
        decoder_output=model_decoder.predict(x_data)
        token_onehot=decoder_output[0,count_tokens,:]
        token_int=np.argmax(token_onehot)
        sampled_word=tokenizer_dest.token_to_word(token_int)
        output_text+=' '+sampled_word
        count_tokens+=1
    output_tokens = decoder_input_data[0]
    print("Input text:")
    print(input_text)
    print()
    print("Translated text:")
    print(output_text)
    print()
    if (true_output_text is not None):
        print("True output text:")
        print(true_output_text)
        print()
idx=3
translate(input_text=data_src[idx],true_output_text=data_dest[idx])
idx=5
translate(input_text=data_src[idx],true_output_text=data_dest[idx])
idx=7
translate(input_text=data_src[idx],true_output_text=data_dest[idx])
