###load data
print('load data...')
import numpy as np
x_train1=np.load('dataset/data/x_train1.npy') # sentence s1
x_train2=np.load('dataset/data/x_train2.npy') # sentence s2
x_train_w1=np.load('dataset/data/x_train_w1.npy')  # Exact word matching s1
x_train_w2=np.load('dataset/data/x_train_w2.npy')  # Exact word matching s2
y_train=np.load('dataset/data/y_train.npy') # label
x_val1=np.load('dataset/data/x_val1.npy')
x_val2=np.load('dataset/data/x_val2.npy')
x_val_w1=np.load('dataset/data/x_val_w1.npy')
x_val_w2=np.load('dataset/data/x_val_w2.npy')
y_val=np.load('dataset/data/y_val.npy')
x_test1=np.load('dataset/data/x_test1.npy')
x_test2=np.load('dataset/data/x_test2.npy')
x_test_w1=np.load('dataset/data/x_test_w1.npy')
x_test_w2=np.load('dataset/data/x_test_w2.npy')
y_test=np.load('dataset/data/y_test.npy')



embedding_matrix=np.load('dataset/data/embedding_matrix.npy') # Pre-trained word vector


x_char_train1=np.load('dataset/char/x_train1.npy') # Character s1
x_char_train2=np.load('dataset/char/x_train2.npy') # Character s2

x_char_val1=np.load('dataset/char/x_val1.npy')
x_char_val2=np.load('dataset/char/x_val2.npy')

x_char_test1=np.load('dataset/char/x_test1.npy')
x_char_test2=np.load('dataset/char/x_test2.npy')



###Build Model
print('build model...')
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape,BatchNormalization
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, LSTM, TimeDistributed, Bidirectional,Masking
from keras.layers.merge import concatenate
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import RMSprop
import keras.backend as K
from keras.layers import Dot,Softmax,Concatenate,Multiply,Subtract,Add
from keras.layers.merge import concatenate
import tensorflow as tf

np.random.seed(5689)
tf.set_random_seed(5689)


maxlen = 15  # We will cut reviews after 15 words 
max_words = 5257 
embedding_dim = 300

char_len=5  #char length
max_chars=1625 #char num
char_dim=50 
char_embdim=50
batchsize=32
beta=0.5
K=5

##input
input1 = Input(shape=(maxlen,), dtype='float32')
input2 = Input(shape=(maxlen,), dtype='float32')

input3 = Input(shape=(maxlen,5), dtype='float32')
input4 = Input(shape=(maxlen,5), dtype='float32')

input5 = Input(shape=(maxlen,), dtype='float32')
input6 = Input(shape=(maxlen,), dtype='float32')

gama_input=Input(shape=(1,), dtype='float32')

##1 Word Representation Layer
#1.1 Embedding Layer
#1.1.1 word embedding
word_embedder = Embedding(max_words, embedding_dim, input_length=maxlen,mask_zero=True,weights = [embedding_matrix], trainable = False)
embedder1=word_embedder(input1)
embedder2=word_embedder(input2)

#1.1.2 char embedding
charEmbedder=Embedding(max_chars, char_dim, input_length=(maxlen,char_len),mask_zero=True, trainable = True)
ci3=charEmbedder(input3)
ci4=charEmbedder(input4)

def rsp(x):
    ss=K.reshape(x,(-1,x.shape[2],x.shape[3]))
    return ss

c3=Lambda(rsp,output_shape=(char_len,char_dim))(ci3)
c4=Lambda(rsp,output_shape=(char_len,char_dim))(ci4)

char_gru=LSTM(char_embdim, dropout=0.2, recurrent_dropout=0.1)
cr3=char_gru(c3)
cr4=char_gru(c4)

def char_emb(x,dim=char_embdim):
    ss=K.reshape(x,(-1,maxlen,dim))
    return ss

embedder3=Lambda(char_emb,output_shape=(maxlen,char_embdim))(cr3)
embedder4=Lambda(char_emb,output_shape=(maxlen,char_embdim))(cr4)

#1.1.3 Exact word matching

def same_word(x):
    x=K.reshape(x,(-1,maxlen,1)) 
    return x


embedder5=Lambda(same_word,output_shape=(maxlen,1))(input5)
embedder6=Lambda(same_word,output_shape=(maxlen,1))(input6)

#1.2 Context layer
embed1=concatenate([embedder1,embedder3,embedder5], axis=-1)
embed2=concatenate([embedder2,embedder4,embedder6], axis=-1)

from WPlayer2 import MaskLayer as MSL #Mask zero
embed1=MSL()(embed1)
embed2=MSL()(embed2)

share_bLSTM1 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1,return_sequences=True))

l1 = share_bLSTM1(embed1)
r1 = share_bLSTM1(embed2)

l1=MSL()(l1)
r1=MSL()(r1)
share_bLSTM2 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1,return_sequences=True))

l2=share_bLSTM2(l1)
r2=share_bLSTM2(r1)

l2=MSL()(l2)
r2=MSL()(r2)

ct_l=concatenate([l1,l2],axis=-1)
ct_r=concatenate([r1,r2],axis=-1)


##2 KWPE Layer
from Layers import KWPE as WPE
al1,ar1,wp_l1,wp_r1,word_l1,word_r1=WPE(unit=16)([ct_l,ct_r,gama_input])
al2,ar2,wp_l2,wp_r2,word_l2,word_r2=WPE(unit=16)([ct_l,ct_r,gama_input])
al3,ar3,wp_l3,wp_r3,word_l3,word_r3=WPE(unit=16)([ct_l,ct_r,gama_input])
al4,ar4,wp_l4,wp_r4,word_l4,word_r4=WPE(unit=16)([ct_l,ct_r,gama_input])
al5,ar5,wp_l5,wp_r5,word_l5,word_r5=WPE(unit=16)([ct_l,ct_r,gama_input])


##3 Matching Layer
def match(vests):
    x1,x2=vests
    sub=x1-x2
    mult=x1*x2
    ks=K.abs(sub)
    norm =K.l2_normalize(sub,axis=-1)
    out=K.concatenate([x1,x2,mult,sub,ks,norm],axis=-1)
    return out

def match_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],shape1[1]*6)


# WP_Task
def wordpair(x):
    s1=Lambda(match, output_shape=match_output_shape)(x)
    s1=BatchNormalization()(s1)
    s1=Dense(64, activation='relu')(s1)
    s1=BatchNormalization()(s1)
    s1=Dense(16, activation='relu')(s1)
    s1=BatchNormalization()(s1)
    s1=Dense(1, activation='sigmoid')(s1)
    return s1


wy1=wordpair([word_l1,word_r1])
wy2=wordpair([word_l2,word_r2])
wy3=wordpair([word_l3,word_r3])
wy4=wordpair([word_l4,word_r4])
wy5=wordpair([word_l5,word_r5])

weight = Lambda(lambda x:x*0.2)


output1=Add()([weight(wy1),weight(wy2),weight(wy3),weight(wy4),weight(wy5)])

# SP_Task
m_l =Add()([wp_l1, wp_l2,wp_l3,wp_l4,wp_l5])
m_r =Add()([wp_r1, wp_r2,wp_r3,wp_r4,wp_r5])


##5 Denoising Layer
slstm=Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.1))

q1=slstm(m_l)
q2=slstm(m_r)


##6 Predict layer
senfeature = Lambda(match, output_shape=match_output_shape)([q1, q2])

senfeature = BatchNormalization()(senfeature)
senfeature = Dense(64, activation='relu')(senfeature)

merged = BatchNormalization()(senfeature)
merged = Dense(16, activation='relu')(merged)

merged = BatchNormalization()(merged)
output2 = Dense(1, activation='sigmoid')(merged)

weight_sp = Lambda(lambda x:x*beta)
weight_wp = Lambda(lambda x:x*(1-beta))

output = Add()([weight_sp(output2),weight_wp(output1)])

model = Model(inputs = [input1,input2,input3,input4,input5,input6,gama_input], outputs = output)
model.summary()

###Train model
print('Train model...')
train_gama=np.ones_like(y_train)
test_gama=np.ones_like(y_test)
val_gama=np.ones_like(y_val)


from keras.optimizers import RMSprop
opt=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)


model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['acc'])

epochs_num=30
sr=10
## annealing schedule
for i in range(epochs_num):
    print('epoch '+str(i+1))
    history = model.fit([x_train1,x_train2,x_train_w1,x_train_w2,x_char_train1,x_char_train2,train_gama*(i+1)/3],y_train,
                        epochs=1,
                        batch_size=batchsize,
                        validation_data=([x_val1,x_val2,x_val_w1,x_val_w2,x_char_val1,x_char_val2,val_gama*(i+1)/3],y_val))

