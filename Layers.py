from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

import tensorflow as tf
from keras.regularizers import l2


##KWPE Layer
class KWPE(Layer):
    def __init__(self,bias=True,sr=10,unit=16,**kwargs):
        self.supports_masking = True
        super(KWPE, self).__init__(**kwargs)
        self.bias=bias
        self.sr=sr
        self.unit=unit
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.w1 =self.add_weight(name='w1',
                                      shape=(input_shape[0][-1],self.unit),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)

        self.w2 =self.add_weight(name='w2',
                                      shape=(input_shape[0][1],self.unit),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)

        self.w3 =self.add_weight(name='w3',
                                      shape=(input_shape[0][1],self.unit),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)
        self.we =self.add_weight(name='we',
                                      shape=(self.unit,1),
                                      initializer='glorot_normal',
                                      regularizer=l2(0.000001), 
                                       trainable=True)
        super(KWPE, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        return None
    def call(self,x,mask=None):
        assert isinstance(x, list)
        x1,x2=x
        input_shape = K.shape(x1)
        pq=K.batch_dot(x1,x2,axes=2)
        pp=K.batch_dot(x1,x1,axes=2)
        qp=K.batch_dot(x2,x1,axes=2)
        qq=K.batch_dot(x2,x2,axes=2)
        eij1 = K.dot(K.tanh(K.dot(pq,self.w2)+K.dot(pp,self.w3)+K.dot(x1,self.w1)),self.we)*self.sr
        ai1 = K.exp(eij1)
        if mask!=None:
            ms1=tf.cast(mask[0],'float32')
            ms1=K.reshape(ms1,(-1,input_shape[1],1))
            ai1 = ai1*ms1
        
        ww1 = ai1/(K.sum(ai1, axis=1,keepdims=True)+ K.epsilon())
        ot1 = x1*ww1
        eij2 = K.dot(K.tanh(K.dot(qp,self.w2)+K.dot(qq,self.w3)+K.dot(x2,self.w1)),self.we)*self.sr
        ai2 = K.exp(eij2)
        if mask!=None:
            ms2=tf.cast(mask[1],'float32')
            ms2=K.reshape(ms2,(-1,input_shape[1],1))
            ai2 = ai2*ms2
        ww2 = ai2/(K.sum(ai2, axis=1,keepdims=True)+ K.epsilon())
        ot2 = x2*ww2
        oot1=K.sum(ot1,axis=1)
        oot2=K.sum(ot2,axis=1)
        return [ww1,ww2,ot1,ot2,oot1,oot2]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0],shape_a[1],1), (shape_b[0],shape_b[1],1),(shape_a[0],shape_a[1],shape_a[2]), (shape_b[0],shape_b[1],shape_b[2]),(shape_a[0],shape_a[2]), (shape_b[0],shape_b[2])]


class MaskLayer(Layer):
    def __init__(self,**kwargs):
        self.supports_masking = True
        super(MaskLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MaskLayer, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        return input_mask
    def call(self,x,mask=None):
        input_shape = K.shape(x)
        if mask!=None:
            ms=tf.cast(mask,'float32')
            ms=K.reshape(ms,(-1,input_shape[1],1))
            x=x*ms
        # return [ww1,ww2]
        return x
    def compute_output_shape(self, input_shape):
        return input_shape

