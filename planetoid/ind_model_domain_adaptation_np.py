import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import theano
from theano import sparse
import theano.tensor as T
import lasagne
import layers
import theano
import numpy as np
import random
from collections import defaultdict as dd
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from base_model import base_model

class ind_model(base_model):
    """Planetoid-I.
    """

    def add_data(self, x, y, allx, graph,domain_data):
        """add data to the model.
        x (scipy.sparse.csr_matrix): feature vectors for labeled training data.
        y (numpy.ndarray): one-hot label encoding for labeled training data.
        allx (scipy.sparse.csr_matrix): feature vectors for both labeled and unlabeled data.
        graph (dict): the format is {index: list_of_neighbor_index}. Only supports binary graph.
        Let n be the number of training (both labeled and unlabeled) training instances.
        These n instances should be indexed from 1 to n - 1 in the graph with the same order in allx.
        """
        self.x, self.y, self.allx, self.graph,self.domain_x = x, y, allx, graph,domain_data
        self.num_ver = self.allx.shape[0]
        self.nb_classes=self.x.shape[1]   
        self.base_lambda_val = 1e-4
        self.decay_lambda_val = 0.8

    def build(self):
        """build the model. This method should be called after self.add_data.
        """
#        x_sym = sparse.csr_matrix('x', dtype = 'float32')
        x_sym = T.matrix('x', dtype = 'float32')
        self.x_sym = x_sym
        y_sym = T.imatrix('y')
#        gx_sym = sparse.csr_matrix('gx', dtype = 'float32')
        gx_sym = T.matrix('gx', dtype = 'float32')
        gy_sym = T.ivector('gy')
        gz_sym = T.vector('gz')

        l_x_in = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = x_sym)
        l_gx_in = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = gx_sym)
        l_gy_in = lasagne.layers.InputLayer(shape = (None, ), input_var = gy_sym)

        l_x_in1=lasagne.layers.dropout(l_x_in, p=0.01)
        l_x_1 = layers.DenseLayer(l_x_in1, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        l_x_2 = layers.DenseLayer(l_x_in, self.embedding_size)

        W = l_x_2.W
                
        if self.use_feature:
            l_x = lasagne.layers.ConcatLayer([l_x_1, l_x_2], axis = 1)
            l_x = layers.DenseLayer(l_x, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        else:
            l_x = l_x_2
            
        l_x_2 = layers.DenseLayer(l_x_2, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)    
        l_gx = layers.DenseLayer(l_gx_in, self.embedding_size, W = W)
        HYPOTHETICALLY={l_gx:(200,self.embedding_size)}
        print("Layer Shape")
        LIN=get_output_shape(l_gx,HYPOTHETICALLY)
        print(HYPOTHETICALLY)
        print(LIN)        
        print("graph...") 
        
        if self.neg_samp > 0:
            l_gy = lasagne.layers.EmbeddingLayer(l_gy_in, input_size = self.num_ver, output_size = self.embedding_size)
            l_gx = lasagne.layers.ElemwiseMergeLayer([l_gx, l_gy], T.mul)
            pgy_sym = lasagne.layers.get_output(l_gx)
            g_loss = - T.log(T.nnet.sigmoid(T.sum(pgy_sym, axis = 1) * gz_sym)).sum()
        else:
            l_gx = lasagne.layers.DenseLayer(l_gx, self.num_ver, nonlinearity = lasagne.nonlinearities.softmax)
            pgy_sym = lasagne.layers.get_output(l_gx)
            g_loss = lasagne.objectives.categorical_crossentropy(pgy_sym, gy_sym).sum()
        
        self.l = [l_x, l_gx]

        py_sym = lasagne.layers.get_output(l_x)
        loss = lasagne.objectives.categorical_crossentropy(py_sym, y_sym).mean()
        if self.layer_loss and self.use_feature:
            hid_sym = lasagne.layers.get_output(l_x_1)
            loss += lasagne.objectives.categorical_crossentropy(hid_sym, y_sym).mean()
            emd_sym = lasagne.layers.get_output(l_x_2)
            loss += lasagne.objectives.categorical_crossentropy(emd_sym, y_sym).mean()

        params = [l_x_1.W, l_x_1.b, l_x_2.W, l_x_2.b, l_x.W, l_x.b] if self.use_feature else [l_x.W, l_x.b]
        if self.update_emb:
            params = lasagne.layers.get_all_params(l_x)
#        updates = lasagne.updates.adadelta(loss, params, learning_rate = self.learning_rate)
#        self.train_fn = theano.function([x_sym, y_sym], loss, updates = updates)

        g_params = lasagne.layers.get_all_params(l_gx)
        g_updates = lasagne.updates.adadelta(g_loss, g_params, learning_rate = self.g_learning_rate)
        self.g_fn = theano.function([gx_sym, gy_sym, gz_sym], g_loss, updates = g_updates, on_unused_input = 'ignore')

        self.test_fn = theano.function([x_sym], py_sym)
        
        #source_network = lasagne.layers.DenseLayer(l_x_in,num_units=nb_classes,nonlinearity=lasagne.nonlinearities.softmax)
        domain_network = lasagne.layers.DenseLayer(l_x_in,num_units=self.nb_classes,nonlinearity=lasagne.nonlinearities.softmax)
        
        #source_prediction = lasagne.layers.get_output(source_network)
        domain_prediction = lasagne.layers.get_output(domain_network)
        
        #source_loss = T.mean(lasagne.objectives.categorical_crossentropy(source_prediction, train_y_var))
        #source_params = lasagne.layers.get_all_params(source_network, trainable=True)
        #domain_prediction = lasagne.layers.get_output(domain_network)
        domain_y_var = T.imatrix('domain_label')
        domain_loss = T.mean(lasagne.objectives.categorical_crossentropy(domain_prediction, domain_y_var))
        domain_params = lasagne.layers.get_all_params(domain_network, trainable=True)
        
        common = set(params) & set(domain_params)
        
        #lambda_val =  1-1e-8
        val =  1e-8
        self.lambda_val = theano.shared(lasagne.utils.floatX(val))        
        updates = lasagne.updates.adagrad(loss - (val * domain_loss), params, learning_rate = 0.1)
        
        #updates1 = lasagne.updates.adadelta(source_loss - (lambda_val * domain_loss), source_params, learning_rate=1.0)  # update blue and green part
        updates2 = lasagne.updates.adagrad(domain_loss, list(set(domain_params) - common), learning_rate=1.0)
        updates3 = lasagne.updates.adagrad(-(val * domain_loss), list(common),learning_rate=1.0)
        updates.update(updates2)
        updates2.update(updates3)
            
        #domain_y_var = T.imatrix('domain_label')
        self.train_fn = theano.function([x_sym, y_sym,domain_y_var], loss, updates = updates)        
        #train1 = theano.function([l_x_in.input_var, y_sym, domain_y_var], loss, updates=updates1)
        self.train2 = theano.function([l_x_in.input_var, domain_y_var], domain_loss, updates=updates2)    
        

    def gen_train_inst(self):
        """generator for batches for classification loss.
        """
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype = np.int32)
            i = 0
            while i < self.x.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]]
                i = j

    def gen_graph(self):
        """generator for batches for graph context loss.
        """
        while True:
            ind = np.random.permutation(self.num_ver)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                for k in ind[i: j]:
                    if len(self.graph[k]) == 0: continue
                    path = [k]
                    for p in range(self.path_size):
                        #path.append(random.choice(self.graph[path[-1]]))
                        path.append(self.graph[path[-1]][p])
                    for l in range(len(path)):
                        if path[l] >= self.allx.shape[0]: continue
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path): continue
                            if path[m] >= self.allx.shape[0]: continue
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            for _ in range(self.neg_samp):
                                g.append([path[l], random.randint(0, self.num_ver - 1)])
                                gy.append(- 1.0)
                g = np.array(g, dtype = np.int32)
                yield self.allx[g[:, 0]], g[:, 1], gy
                i = j

    def gen_label_graph(self):
        """generator for batches for label context loss.
        """
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        while True:
            g, gy = [], []
            for _ in range(self.g_sample_size):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1: continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
                for _ in range(self.neg_samp):
                    g.append([x1, random.choice(not_label[label])])
                    gy.append(- 1.0)
            g = np.array(g, dtype = np.int32)
#            print(g.shape)
#            print(g[:, 0])
#            print(len(gy))
#            print(gy)        
            #print(g[:, 1])    
            yield self.allx[g[:, 0]], g[:, 1], gy



    def init_train(self, init_iter_label, init_iter_graph):
        """pre-training of graph embeddings.
        init_iter_label (int): # iterations for optimizing label context loss.
        init_iter_graph (int): # iterations for optimizing graph context loss.
        """
        for i in range(init_iter_label):
            gx, gy, gz = next(self.label_generator)
            loss = self.g_fn(gx, gy, gz)
            #print 'iter label', i, loss

        for i in range(init_iter_graph):
            gx, gy, gz = next(self.graph_generator)
            loss = self.g_fn(gx, gy, gz)
            #print 'iter graph', i, loss

    def step_train(self, max_iter, iter_graph, iter_inst, iter_label):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        iter_graph (int): # iterations for optimizing the graph context loss.
        iter_inst (int): # iterations for optimizing the classification loss.
        iter_label (int): # iterations for optimizing the label context loss.
        """
        for miter in range(max_iter):
            for _ in range(self.comp_iter(iter_graph)):
                gx, gy, gz = next(self.graph_generator)
                self.g_fn(gx, gy, gz)
            for iter_i in range(self.comp_iter(iter_inst)):
                x, y = next(self.inst_generator)
                loss = self.train_fn(x, y)
                #print 'maxiter, iter supervised loss', miter,iter_i, loss
            for _ in range(self.comp_iter(iter_label)):
                gx, gy, gz = next(self.label_generator)
                self.g_fn(gx, gy, gz)

    def predict(self, tx):
        """predict the dev or test instances.
        tx (scipy.sparse.csr_matrix): feature vectors for dev instances.

        returns (numpy.ndarray, #instacnes * #classes): classification probabilities for dev instances.
        """
        return self.test_fn(tx)

    def gen_graph_minibatch(self):
        """generator for batches for graph context loss.
        """
#        while True:
        ind = np.random.permutation(self.num_ver)
        i = 0
#        while i < ind.shape[0]:
        for i in range(0, self.x.shape[0] - self.g_batch_size + 1, self.g_batch_size):    
            g, gy = [], []
            j = i + self.g_batch_size-1 #min(ind.shape[0], i + self.g_batch_size)
            for k in ind[i: j]:
                if len(self.graph[k]) == 0: continue
                path = [k]
                for p in range(self.path_size):
                    #path.append(random.choice(self.graph[path[-1]]))
                    path.append(self.graph[path[-1]][p])
                for l in range(len(path)):
                    if path[l] >= self.allx.shape[0]: continue
                    for m in range(l - self.window_size, l + self.window_size + 1):
                        if m < 0 or m >= len(path): continue
                        if path[m] >= self.allx.shape[0]: continue
                        g.append([path[l], path[m]])
                        gy.append(1.0)
                        for _ in range(self.neg_samp):
                            g.append([path[l], random.randint(0, self.num_ver - 1)])
                            gy.append(- 1.0)
            g = np.array(g, dtype = np.int32)
            yield self.allx[g[:, 0]], g[:, 1], gy
            #i = j
                
    def gen_label_graph_minibatch(self,shuffle=False):
        """generator for batches for label context loss.
        """
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        while True:
#        for start_idx in range(0, self.x.shape[0] - self.g_sample_size + 1, self.g_sample_size):
            g, gy = [], []
            for _ in range(self.g_sample_size):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1: continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
                for _ in range(self.neg_samp):
                    g.append([x1, random.choice(not_label[label])])
                    gy.append(- 1.0)
                g = np.array(g, dtype = np.int32)
                yield self.allx[g[:, 0]], g[:, 1], gy  
            
    def iterate_minibatches(self, shuffle=False):
        if shuffle:
            indices = np.arange(self.x.shape[0])
            np.random.shuffle(indices)
        for start_idx in range(0, self.x.shape[0] - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield self.x[excerpt], self.y[excerpt]
            
    def iterate_minibatches_data(self, inputs, batchsize):
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]    
            
    def step_train_minibatch(self,iter_label,epoch):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        iter_graph (int): # iterations for optimizing the graph context loss.
        iter_inst (int): # iterations for optimizing the classification loss.
        iter_label (int): # iterations for optimizing the label context loss.
        """

        train_err=0;
        val = self.base_lambda_val * (self.decay_lambda_val ** epoch)
        #val=lambda_val.get_value() * 0.1
        self.lambda_val.set_value(lasagne.utils.floatX(val))        
        
        #for miter in range(max_iter):
        for batch in self.gen_graph_minibatch():
            gx, gy, gz = batch #next(self.graph_generator)
            loss=self.g_fn(gx, gy, gz)
#            print 'batch graph context loss', loss
            
        for batch in self.iterate_minibatches(shuffle=True):
            inputs, targets = batch
            source_tr_y=[0] * len(targets)
            y = np.zeros((len(targets), self.nb_classes))
            y[np.arange(len(targets)), source_tr_y] = 1
            source_tr_y=y.astype('int32')            
            loss = self.train_fn(inputs, targets,source_tr_y)
#            print 'batch supervised loss', loss
        for batch in self.iterate_minibatches_data(self.domain_x, self.batch_size):
            inputs = batch
            domain_tr_y=[1] * len(batch)
            #source_tr_y=[0] * len(targets)
            y = np.zeros((len(batch), self.nb_classes))
            y[np.arange(len(batch)), domain_tr_y] = 1
            domain_tr_y=y.astype('int32')
            train_err += self.train2(inputs, domain_tr_y)            
        #print("training error "+str(train_err))
        for _ in range(self.comp_iter(iter_label)):
            gx, gy, gz = next(self.label_generator)
            self.g_fn(gx, gy, gz)
#            print 'batch graph label loss', loss
            
#        for batch in self.gen_label_graph_minibatch(shuffle=True):                
#            gx, gy, gz = batch
#            self.g_fn(gx, gy, gz)
#            print 'batch graph label loss', loss            