import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
from numpy.random import RandomState
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQKGR(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)

        self._build_inputs()

        self.weights = self._build_weights()

        self._build_model_phase_II()

        self._build_loss_phase_II()

        self._build_model_phase_I()

        self._build_loss_phase_I()

        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'dqkgr'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100

        # initialize the attentive matrix A for phase I.
        self.A_in = data_config['A_in']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']
        self.all_v_list = data_config['all_v_list']

        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        # 64
        self.emb_dim = args.embed_size
        # 1024
        self.batch_size = args.batch_size

        # settings for KG part.
        # 64
        self.kge_dim = args.embed_size

        # default:2048
        self.batch_size_kg = args.batch_size_kg

        # [64,32,16]
        self.weight_size = eval(args.layer_size)

        self.n_layers = len(self.weight_size)
        # bi
        self.alg_type = args.alg_type

        self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type,
                                              args.alg_type, self.n_layers)
        # [1e-5, 1e-5]
        # self.regs = args.regs
        # self.regs = args.reg
        self.reg_1 = args.reg1
        self.reg_2 = args.reg2

        # 50
        self.verbose = args.verbose
        # score function type
        self.score_func = args.score_func
        self.normal_r = args.normal_r
        self.initial_c = args.initial
        # self.kge_loss2 = tf.constant(0.0, tf.float32, [1])
        self.att_type = args.att_type
        self.bi_type = args.bi_type

    def _build_inputs(self):
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None, ))
        self.pos_items = tf.placeholder(tf.int32, shape=(None, ))
        self.neg_items = tf.placeholder(tf.int32, shape=(None, ))

        self.A_values = tf.placeholder(tf.float32,
                                       shape=[len(self.all_v_list)],
                                       name='A_values')

        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')

        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def quaternion_init(self, in_features, out_features, criterion='he'):
    
        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(2019)

        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)


        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        return (weight_r, weight_i, weight_j, weight_k)

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            if self.initial_c == 1:
                u_a, u_b, u_c, u_d = self.quaternion_init(self.n_users, self.emb_dim)
                all_weights['user_embed_a'] = tf.Variable(
                    initial_value=u_a,
                    trainable=True,
                    name='user_embed_a',
                    dtype=tf.float32)
                all_weights['user_embed_b'] = tf.Variable(
                    initial_value=u_b,
                    trainable=True,
                    name='user_embed_b',
                    dtype=tf.float32)
                all_weights['user_embed_c'] = tf.Variable(
                    initial_value=u_c,
                    trainable=True,
                    name='user_embed_c',
                    dtype=tf.float32)
                all_weights['user_embed_d'] = tf.Variable(
                    initial_value=u_d,
                    trainable=True,
                    name='user_embed_d',
                    dtype=tf.float32)

                e_a, e_b, e_c, e_d = self.quaternion_init(self.n_entities, self.emb_dim)
                all_weights['entity_embed_a'] = tf.Variable(
                    initial_value=e_a,
                    trainable=True,
                    name='entity_embed_a',
                    dtype=tf.float32)
                all_weights['entity_embed_b'] = tf.Variable(
                    initial_value=e_b,
                    trainable=True,
                    name='entity_embed_b',
                    dtype=tf.float32)
                all_weights['entity_embed_c'] = tf.Variable(
                    initial_value=e_c,
                    trainable=True,
                    name='entity_embed_c',
                    dtype=tf.float32)
                all_weights['entity_embed_d'] = tf.Variable(
                    initial_value=e_d,
                    trainable=True,
                    name='entity_embed_d',
                    dtype=tf.float32)

                r_a, r_b, r_c, r_d = self.quaternion_init(self.n_relations, self.emb_dim)
                all_weights['relation_embed_a'] = tf.Variable(
                    initial_value=r_a,
                    trainable=True,
                    name='relation_embed_a',
                    dtype=tf.float32)
                all_weights['relation_embed_b'] = tf.Variable(
                    initial_value=r_b,
                    trainable=True,
                    name='relation_embed_b',
                    dtype=tf.float32)
                all_weights['relation_embed_c'] = tf.Variable(
                    initial_value=r_c,
                    trainable=True,
                    name='relation_embed_c',
                    dtype=tf.float32)
                all_weights['relation_embed_d'] = tf.Variable(
                    initial_value=r_d,
                    trainable=True,
                    name='relation_embed_d',
                    dtype=tf.float32)
                
                if self.bi_type == 2:
                    r_t_a, r_t_b, r_t_c, r_t_d = self.quaternion_init(self.n_relations, self.emb_dim)
                    all_weights['relation_t_embed_a'] = tf.Variable(
                        initial_value=r_t_a,
                        trainable=True,
                        name='relation_t_embed_a',
                        dtype=tf.float32)
                    all_weights['relation_t_embed_b'] = tf.Variable(
                        initial_value=r_t_b,
                        trainable=True,
                        name='relation_t_embed_b',
                        dtype=tf.float32)
                    all_weights['relation_t_embed_c'] = tf.Variable(
                        initial_value=r_t_c,
                        trainable=True,
                        name='relation_t_embed_c',
                        dtype=tf.float32)
                    all_weights['relation_t_embed_d'] = tf.Variable(
                        initial_value=r_t_d,
                        trainable=True,
                        name='relation_t_embed_d',
                        dtype=tf.float32)                    
                
                print('using initialization algorithm tailed for quaternion-valued networks')
            
            if self.initial_c == 0:
                all_weights['user_embed_a'] = tf.Variable(initializer(
                    [self.n_users, self.emb_dim]),
                                                        name='user_embed_a')
                all_weights['user_embed_b'] = tf.Variable(initializer(
                    [self.n_users, self.emb_dim]),
                                                        name='user_embed_b')
                all_weights['user_embed_c'] = tf.Variable(initializer(
                    [self.n_users, self.emb_dim]),
                                                        name='user_embed_c')
                all_weights['user_embed_d'] = tf.Variable(initializer(
                    [self.n_users, self.emb_dim]),
                                                        name='user_embed_d')
                all_weights['entity_embed_a'] = tf.Variable(initializer(
                    [self.n_entities, self.emb_dim]),
                                                            name='entity_embed_a')
                all_weights['entity_embed_b'] = tf.Variable(initializer(
                    [self.n_entities, self.emb_dim]),
                                                            name='entity_embed_b')
                all_weights['entity_embed_c'] = tf.Variable(initializer(
                    [self.n_entities, self.emb_dim]),
                                                            name='entity_embed_c')
                all_weights['entity_embed_d'] = tf.Variable(initializer(
                    [self.n_entities, self.emb_dim]),
                                                            name='entity_embed_d')
                all_weights['relation_embed_a'] = tf.Variable(initializer(
                    [self.n_relations, self.emb_dim]),
                                                            name='relation_embed_a')
                all_weights['relation_embed_b'] = tf.Variable(initializer(
                    [self.n_relations, self.emb_dim]),
                                                            name='relation_embed_b')
                all_weights['relation_embed_c'] = tf.Variable(initializer(
                    [self.n_relations, self.emb_dim]),
                                                            name='relation_embed_c')
                all_weights['relation_embed_d'] = tf.Variable(initializer(
                    [self.n_relations, self.emb_dim]),
                                                            name='relation_embed_d')   
                if self.bi_type == 2:
                    all_weights['relation_t_embed_a'] = tf.Variable(initializer(
                        [self.n_relations, self.emb_dim]),
                                                                name='relation_t_embed_a')
                    all_weights['relation_t_embed_b'] = tf.Variable(initializer(
                        [self.n_relations, self.emb_dim]),
                                                                name='relation_t_embed_b')
                    all_weights['relation_t_embed_c'] = tf.Variable(initializer(
                        [self.n_relations, self.emb_dim]),
                                                                name='relation_t_embed_c')
                    all_weights['relation_t_embed_d'] = tf.Variable(initializer(
                        [self.n_relations, self.emb_dim]),
                                                                name='relation_t_embed_d')

                print('using xavier initialization')
        else:
            all_weights['user_embed_a'] = tf.Variable(
                initial_value=self.pretrain_data['user_embed_a'],
                trainable=True,
                name='user_embed_a',
                dtype=tf.float32)
            all_weights['user_embed_b'] = tf.Variable(
                initial_value=self.pretrain_data['user_embed_b'],
                trainable=True,
                name='user_embed_b',
                dtype=tf.float32)
            all_weights['user_embed_c'] = tf.Variable(
                initial_value=self.pretrain_data['user_embed_c'],
                trainable=True,
                name='user_embed_c',
                dtype=tf.float32)
            all_weights['user_embed_d'] = tf.Variable(
                initial_value=self.pretrain_data['user_embed_d'],
                trainable=True,
                name='user_embed_d',
                dtype=tf.float32)

            all_weights['entity_embed_a'] = tf.Variable(
                initial_value=self.pretrain_data['entity_embed_a'],
                trainable=True,
                name='entity_embed_a',
                dtype=tf.float32)
            all_weights['entity_embed_b'] = tf.Variable(
                initial_value=self.pretrain_data['entity_embed_b'],
                trainable=True,
                name='entity_embed_b',
                dtype=tf.float32)
            all_weights['entity_embed_c'] = tf.Variable(
                initial_value=self.pretrain_data['entity_embed_c'],
                trainable=True,
                name='entity_embed_c',
                dtype=tf.float32)
            all_weights['entity_embed_d'] = tf.Variable(
                initial_value=self.pretrain_data['entity_embed_d'],
                trainable=True,
                name='entity_embed_d',
                dtype=tf.float32)

            all_weights['relation_embed_a'] = tf.Variable(
                initial_value=self.pretrain_data['relation_embed_a'],
                trainable=True,
                name='relation_embed_a',
                dtype=tf.float32)
            all_weights['relation_embed_b'] = tf.Variable(
                initial_value=self.pretrain_data['relation_embed_b'],
                trainable=True,
                name='relation_embed_b',
                dtype=tf.float32)
            all_weights['relation_embed_c'] = tf.Variable(
                initial_value=self.pretrain_data['relation_embed_c'],
                trainable=True,
                name='relation_embed_c',
                dtype=tf.float32)
            all_weights['relation_embed_d'] = tf.Variable(
                initial_value=self.pretrain_data['relation_embed_d'],
                trainable=True,
                name='relation_embed_d',
                dtype=tf.float32)
            
            if self.bi_type == 2:
                all_weights['relation_t_embed_a'] = tf.Variable(
                    initial_value=self.pretrain_data['relation_t_embed_a'],
                    trainable=True,
                    name='relation_t_embed_a',
                    dtype=tf.float32)
                all_weights['relation_t_embed_b'] = tf.Variable(
                    initial_value=self.pretrain_data['relation_t_embed_b'],
                    trainable=True,
                    name='relation_t_embed_b',
                    dtype=tf.float32)
                all_weights['relation_t_embed_c'] = tf.Variable(
                    initial_value=self.pretrain_data['relation_t_embed_c'],
                    trainable=True,
                    name='relation_t_embed_c',
                    dtype=tf.float32)
                all_weights['relation_t_embed_d'] = tf.Variable(
                    initial_value=self.pretrain_data['relation_t_embed_d'],
                    trainable=True,
                    name='relation_t_embed_d',
                    dtype=tf.float32)            
                    
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_a_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_gc_a_%d' % k)
            all_weights['W_gc_b_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_gc_b_%d' % k)
            all_weights['W_gc_c_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_gc_c_%d' % k)
            all_weights['W_gc_d_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_gc_d_%d' % k)

            all_weights['b_gc_a_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_gc_a_%d' % k)
            all_weights['b_gc_b_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_gc_b_%d' % k)
            all_weights['b_gc_c_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_gc_c_%d' % k)
            all_weights['b_gc_d_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_gc_d_%d' % k)

            all_weights['W_bi_a_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_bi_a_%d' % k)
            all_weights['W_bi_b_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_bi_b_%d' % k)
            all_weights['W_bi_c_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_bi_c_%d' % k)
            all_weights['W_bi_d_%d' % k] = tf.Variable(initializer(
                [self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                       name='W_bi_d_%d' % k)

            all_weights['b_bi_a_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_bi_a_%d' % k)
            all_weights['b_bi_b_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_bi_b_%d' % k)
            all_weights['b_bi_c_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_bi_c_%d' % k)
            all_weights['b_bi_d_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                       name='b_bi_d_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(initializer(
                [2 * self.weight_size_list[k], self.weight_size_list[k + 1]]),
                                                      name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(initializer(
                [1, self.weight_size_list[k + 1]]),
                                                      name='b_mlp_%d' % k)

        return all_weights

    def _build_model_phase_I(self):
        if self.alg_type in ['bi']:
            self.ua_embeddings_a, self.ea_embeddings_a, \
            self.ua_embeddings_b, self.ea_embeddings_b, \
            self.ua_embeddings_c, self.ea_embeddings_c, \
            self.ua_embeddings_d, self.ea_embeddings_d, = self._create_bi_interaction_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ea_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['graphsage']:
            self.ua_embeddings, self.ea_embeddings = self._create_graphsage_embed(
            )

        self.u_e_a = tf.nn.embedding_lookup(self.ua_embeddings_a, self.users)
        self.u_e_b = tf.nn.embedding_lookup(self.ua_embeddings_b, self.users)
        self.u_e_c = tf.nn.embedding_lookup(self.ua_embeddings_c, self.users)
        self.u_e_d = tf.nn.embedding_lookup(self.ua_embeddings_d, self.users)

        self.pos_i_e_a = tf.nn.embedding_lookup(self.ea_embeddings_a,
                                                self.pos_items)
        self.pos_i_e_b = tf.nn.embedding_lookup(self.ea_embeddings_b,
                                                self.pos_items)
        self.pos_i_e_c = tf.nn.embedding_lookup(self.ea_embeddings_c,
                                                self.pos_items)
        self.pos_i_e_d = tf.nn.embedding_lookup(self.ea_embeddings_d,
                                                self.pos_items)

        self.neg_i_e_a = tf.nn.embedding_lookup(self.ea_embeddings_a,
                                                self.neg_items)
        self.neg_i_e_b = tf.nn.embedding_lookup(self.ea_embeddings_b,
                                                self.neg_items)
        self.neg_i_e_c = tf.nn.embedding_lookup(self.ea_embeddings_c,
                                                self.neg_items)
        self.neg_i_e_d = tf.nn.embedding_lookup(self.ea_embeddings_d,
                                                self.neg_items)

        if self.score_func == 0:
            self.batch_predictions_a = tf.matmul(self.u_e_a,
                                                 self.pos_i_e_a,
                                                 transpose_a=False,
                                                 transpose_b=True)
            self.batch_predictions_b = tf.matmul(self.u_e_b,
                                                 self.pos_i_e_b,
                                                 transpose_a=False,
                                                 transpose_b=True)
            self.batch_predictions_c = tf.matmul(self.u_e_c,
                                                 self.pos_i_e_c,
                                                 transpose_a=False,
                                                 transpose_b=True)
            self.batch_predictions_d = tf.matmul(self.u_e_d,
                                                 self.pos_i_e_d,
                                                 transpose_a=False,
                                                 transpose_b=True)
        elif self.score_func == 1:
            self.batch_predictions_a = tf.matmul(self.u_e_a, self.pos_i_e_a, transpose_a=False, transpose_b=True) - \
                                    tf.matmul(self.u_e_b, self.pos_i_e_b, transpose_a=False, transpose_b=True) - \
                                    tf.matmul(self.u_e_c, self.pos_i_e_c, transpose_a=False, transpose_b=True) - \
                                    tf.matmul(self.u_e_d, self.pos_i_e_d, transpose_a=False, transpose_b=True) #+\
                                    #tf.matmul( self.pos_i_e_a, self.u_e_a, transpose_a=False, transpose_b=True) - \
                                    #tf.matmul( self.pos_i_e_b,self.u_e_b, transpose_a=False, transpose_b=True) - \
                                    #tf.matmul( self.pos_i_e_c,self.u_e_c, transpose_a=False, transpose_b=True) - \
                                    #tf.matmul( self.pos_i_e_d,self.u_e_d, transpose_a=False, transpose_b=True) 

            self.batch_predictions_b = tf.matmul(self.u_e_a, self.pos_i_e_b, transpose_a=False, transpose_b=True) + \
                                    tf.matmul(self.u_e_b, self.pos_i_e_a, transpose_a=False, transpose_b=True) + \
                                    tf.matmul(self.u_e_c, self.pos_i_e_d, transpose_a=False, transpose_b=True) - \
                                    tf.matmul(self.u_e_d, self.pos_i_e_c, transpose_a=False, transpose_b=True) #+\
                                    # tf.matmul( self.pos_i_e_b,self.u_e_a, transpose_a=False, transpose_b=True) + \
                                    #tf.matmul( self.pos_i_e_a,self.u_e_b, transpose_a=False, transpose_b=True) + \
                                    #tf.matmul(self.pos_i_e_d, self.u_e_c, transpose_a=False, transpose_b=True) - \
                                    #tf.matmul(self.pos_i_e_c, self.u_e_d, transpose_a=False, transpose_b=True) 

            self.batch_predictions_c = tf.matmul(self.u_e_a, self.pos_i_e_c, transpose_a=False, transpose_b=True) + \
                                    tf.matmul(self.u_e_c, self.pos_i_e_a, transpose_a=False, transpose_b=True) + \
                                    tf.matmul(self.u_e_d, self.pos_i_e_b, transpose_a=False, transpose_b=True) - \
                                    tf.matmul(self.u_e_b, self.pos_i_e_d, transpose_a=False, transpose_b=True) #+\
                                    # tf.matmul( self.pos_i_e_c,self.u_e_a, transpose_a=False, transpose_b=True) + \
                                    #tf.matmul(self.pos_i_e_a,self.u_e_c,  transpose_a=False, transpose_b=True) + \
                                    #tf.matmul( self.pos_i_e_b,self.u_e_d, transpose_a=False, transpose_b=True) - \
                                    #tf.matmul( self.pos_i_e_d,self.u_e_b, transpose_a=False, transpose_b=True)
                                    

            self.batch_predictions_d = tf.matmul(self.u_e_a, self.pos_i_e_d, transpose_a=False, transpose_b=True) + \
                                    tf.matmul(self.u_e_d, self.pos_i_e_a, transpose_a=False, transpose_b=True) + \
                                    tf.matmul(self.u_e_b, self.pos_i_e_c, transpose_a=False, transpose_b=True) - \
                                    tf.matmul(self.u_e_c, self.pos_i_e_b, transpose_a=False, transpose_b=True)#+\
                                    # tf.matmul(self.pos_i_e_d,self.u_e_a, transpose_a=False, transpose_b=True) + \
                                    #tf.matmul( self.pos_i_e_a,self.u_e_d, transpose_a=False, transpose_b=True) + \
                                    #tf.matmul(self.pos_i_e_c,self.u_e_b,  transpose_a=False, transpose_b=True) - \
                                    #tf.matmul( self.pos_i_e_b,self.u_e_c, transpose_a=False, transpose_b=True)


        # self.batch_predictions = tf.sigmoid(self.batch_predictions_a) + tf.sigmoid(self.batch_predictions_b) + \
        #                          tf.sigmoid(self.batch_predictions_c) + tf.sigmoid(self.batch_predictions_d)
        self.batch_predictions = self.batch_predictions_a + self.batch_predictions_b + \
                                 self.batch_predictions_c + self.batch_predictions_d                              

    def _build_model_phase_II(self):
        self.h_e_a, self.r_e_a, self.r_t_e_a, self.pos_t_e_a, self.neg_t_e_a, \
        self.h_e_b, self.r_e_b, self.r_t_e_b, self.pos_t_e_b, self.neg_t_e_b, \
        self.h_e_c, self.r_e_c, self.r_t_e_c, self.pos_t_e_c, self.neg_t_e_c, \
        self.h_e_d, self.r_e_d, self.r_t_e_d, self.pos_t_e_d, self.neg_t_e_d, \
        self.rr1,self.rr2,self.rr3,self.rr4,self.rt1,self.rt2,self.rt3,self.rt4   = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)


        self.A_kg_score = self._generate_quatE_score(h=self.h,
                                                     t=self.pos_t,
                                                     r=self.r)
        self.A_out = self._create_attentive_A_out()

    def _get_kg_inference(self, h, r, pos_t, neg_t):
        embeddings_a = tf.concat(
            [self.weights['user_embed_a'], self.weights['entity_embed_a']],
            axis=0)
        embeddings_b = tf.concat(
            [self.weights['user_embed_b'], self.weights['entity_embed_b']],
            axis=0)
        embeddings_c = tf.concat(
            [self.weights['user_embed_c'], self.weights['entity_embed_c']],
            axis=0)
        embeddings_d = tf.concat(
            [self.weights['user_embed_d'], self.weights['entity_embed_d']],
            axis=0)

        # embeddings_a = tf.expand_dims(embeddings_a, 1)
        # embeddings_b = tf.expand_dims(embeddings_b, 1)
        # embeddings_c = tf.expand_dims(embeddings_c, 1)
        # embeddings_d = tf.expand_dims(embeddings_d, 1)

        h_e_a = tf.nn.embedding_lookup(embeddings_a, h)
        h_e_b = tf.nn.embedding_lookup(embeddings_b, h)
        h_e_c = tf.nn.embedding_lookup(embeddings_c, h)
        h_e_d = tf.nn.embedding_lookup(embeddings_d, h)

        pos_t_e_a = tf.nn.embedding_lookup(embeddings_a, pos_t)
        pos_t_e_b = tf.nn.embedding_lookup(embeddings_b, pos_t)
        pos_t_e_c = tf.nn.embedding_lookup(embeddings_c, pos_t)
        pos_t_e_d = tf.nn.embedding_lookup(embeddings_d, pos_t)

        neg_t_e_a = tf.nn.embedding_lookup(embeddings_a, neg_t)
        neg_t_e_b = tf.nn.embedding_lookup(embeddings_b, neg_t)
        neg_t_e_c = tf.nn.embedding_lookup(embeddings_c, neg_t)
        neg_t_e_d = tf.nn.embedding_lookup(embeddings_d, neg_t)

        r_e_a = tf.nn.embedding_lookup(self.weights['relation_embed_a'], r)
        r_e_b = tf.nn.embedding_lookup(self.weights['relation_embed_b'], r)
        r_e_c = tf.nn.embedding_lookup(self.weights['relation_embed_c'], r)
        r_e_d = tf.nn.embedding_lookup(self.weights['relation_embed_d'], r)

        if self.bi_type == 2:  
            r_t_e_a = tf.nn.embedding_lookup(self.weights['relation_t_embed_a'], r)
            r_t_e_b = tf.nn.embedding_lookup(self.weights['relation_t_embed_b'], r)
            r_t_e_c = tf.nn.embedding_lookup(self.weights['relation_t_embed_c'], r)
            r_t_e_d = tf.nn.embedding_lookup(self.weights['relation_t_embed_d'], r)
            
            rr1 = tf.nn.embedding_lookup(self.weights['relation_t_embed_a'], r)
            rr2 = tf.nn.embedding_lookup(self.weights['relation_t_embed_b'], r)
            rr3 = tf.nn.embedding_lookup(self.weights['relation_t_embed_c'], r)
            rr4 = tf.nn.embedding_lookup(self.weights['relation_t_embed_d'], r)
            
            rt1 = tf.nn.embedding_lookup(self.weights['relation_t_embed_a'], r)
            rt2 = tf.nn.embedding_lookup(self.weights['relation_t_embed_b'], r)
            rt3 = tf.nn.embedding_lookup(self.weights['relation_t_embed_c'], r)
            rt4 = tf.nn.embedding_lookup(self.weights['relation_t_embed_d'], r)        
        
        
        if self.bi_type == 1:
            r_t_e_a = r_e_a
            r_t_e_b = - r_e_b
            r_t_e_c = - r_e_c
            r_t_e_d = - r_e_d
            rr1 = r_e_a
            rr2 = - r_e_b
            rr3 = - r_e_c
            rr4 = - r_e_d
            rt1 = r_e_a
            rt2 = - r_e_b
            rt3 = - r_e_c
            rt4 = - r_e_d
            
        return h_e_a, r_e_a, r_t_e_a, pos_t_e_a, neg_t_e_a, h_e_b, r_e_b, r_t_e_b, pos_t_e_b, neg_t_e_b, h_e_c, r_e_c, r_t_e_c, pos_t_e_c, neg_t_e_c, h_e_d, r_e_d, r_t_e_d, pos_t_e_d, neg_t_e_d,\
                rr1,rr2,rr3,rr4,rt1,rt2,rt3,rt4
        
        
        

    def _build_loss_phase_I(self):
        if self.score_func == 0:
            pos_scores_a = tf.reduce_sum(tf.multiply(self.u_e_a,
                                                     self.pos_i_e_a),
                                         axis=1)
            pos_scores_b = tf.reduce_sum(tf.multiply(self.u_e_b,
                                                     self.pos_i_e_b),
                                         axis=1)
            pos_scores_c = tf.reduce_sum(tf.multiply(self.u_e_c,
                                                     self.pos_i_e_c),
                                         axis=1)
            pos_scores_d = tf.reduce_sum(tf.multiply(self.u_e_d,
                                                     self.pos_i_e_d),
                                         axis=1)
            neg_scores_a = tf.reduce_sum(tf.multiply(self.u_e_a,
                                                     self.neg_i_e_a),
                                         axis=1)
            neg_scores_b = tf.reduce_sum(tf.multiply(self.u_e_b,
                                                     self.neg_i_e_b),
                                         axis=1)
            neg_scores_c = tf.reduce_sum(tf.multiply(self.u_e_c,
                                                     self.neg_i_e_c),
                                         axis=1)
            neg_scores_d = tf.reduce_sum(tf.multiply(self.u_e_d,
                                                     self.neg_i_e_d),
                                         axis=1)

        elif self.score_func == 1:
            pos_scores_a = tf.reduce_sum(tf.multiply(self.u_e_a, self.pos_i_e_a), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.pos_i_e_b), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.pos_i_e_c), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.pos_i_e_d), axis=1)

            pos_scores_b = tf.reduce_sum(tf.multiply(self.u_e_a, self.pos_i_e_b), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.pos_i_e_a), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.pos_i_e_d), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.pos_i_e_c), axis=1)

            pos_scores_c = tf.reduce_sum(tf.multiply(self.u_e_a, self.pos_i_e_c), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.pos_i_e_a), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.pos_i_e_b), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.pos_i_e_d), axis=1)

            pos_scores_d = tf.reduce_sum(tf.multiply(self.u_e_a, self.pos_i_e_d), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.pos_i_e_a), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.pos_i_e_c), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.pos_i_e_d), axis=1)

            neg_scores_a = tf.reduce_sum(tf.multiply(self.u_e_a, self.neg_i_e_a), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.neg_i_e_b), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.neg_i_e_c), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.neg_i_e_d), axis=1)

            neg_scores_b = tf.reduce_sum(tf.multiply(self.u_e_a, self.neg_i_e_b), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.neg_i_e_a), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.neg_i_e_d), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.neg_i_e_c), axis=1)

            neg_scores_c = tf.reduce_sum(tf.multiply(self.u_e_a, self.neg_i_e_c), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.neg_i_e_a), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.neg_i_e_b), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.neg_i_e_d), axis=1)

            neg_scores_d = tf.reduce_sum(tf.multiply(self.u_e_a, self.neg_i_e_d), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_d, self.neg_i_e_a), axis=1) + \
                        tf.reduce_sum(tf.multiply(self.u_e_b, self.neg_i_e_c), axis=1) - \
                        tf.reduce_sum(tf.multiply(self.u_e_c, self.neg_i_e_d), axis=1)

        # pos_scores = tf.sigmoid(pos_scores_a) + tf.sigmoid(pos_scores_b) + \
        #              tf.sigmoid(pos_scores_c) + tf.sigmoid(pos_scores_d)
        pos_scores = pos_scores_a + pos_scores_b + \
                     pos_scores_c + pos_scores_d                     

        neg_scores = neg_scores_a + neg_scores_b + \
                     neg_scores_c + neg_scores_d

        # pos_scores = pos_scores_a + pos_scores_b + pos_scores_c + pos_scores_d
        # neg_scores = neg_scores_a + neg_scores_b + neg_scores_c + neg_scores_d

        regularizer = tf.nn.l2_loss(self.u_e_a) + tf.nn.l2_loss(self.pos_i_e_a) + tf.nn.l2_loss(self.neg_i_e_a) + \
                      tf.nn.l2_loss(self.u_e_b) + tf.nn.l2_loss(self.pos_i_e_b) + tf.nn.l2_loss(self.neg_i_e_b) + \
                      tf.nn.l2_loss(self.u_e_c) + tf.nn.l2_loss(self.pos_i_e_c) + tf.nn.l2_loss(self.neg_i_e_c) + \
                      tf.nn.l2_loss(self.u_e_d) + tf.nn.l2_loss(self.pos_i_e_d) + tf.nn.l2_loss(self.neg_i_e_d)

        regularizer = regularizer / self.batch_size

        # Using the softplus as BPR loss to avoid the nan error.
        base_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # base_loss = tf.negative(tf.reduce_mean(maxi))

        self.base_loss = base_loss
        # First, set the kge_loss as zero
        
        # initial
        # self.kge_loss = tf.constant(0.0, tf.float32, [1])
        # add by lzp
        # self.kge_loss2 = tf.constant(0.0, tf.float32, [1])
        self.kge_loss1 = self.reg_1 * self.kge_loss2


        self.reg_loss = self.reg_2 * regularizer
        
        self.loss = self.base_loss + self.kge_loss1 + self.reg_loss + self.reg_loss2

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss)

    def _hamilton_product(self, h_e_a, h_e_b, h_e_c, h_e_d, r_e_a, r_e_b, r_e_c, r_e_d):
        a_0,a_1=tf.unstack(h_e_a,1); a_2,a_3=tf.unstack(h_e_b,1) 
        b_0,b_1=tf.unstack(h_e_c,1); b_2,b_3=tf.unstack(h_e_d,1)
        c_0,c_1=tf.unstack(r_e_a,1); c_2,c_3=tf.unstack(r_e_b,1) 
        d_0,d_1=tf.unstack(r_e_c,1); d_2,d_3=tf.unstack(r_e_d,1)        
        h_0=a_0*c_0-a_1*c_1-a_2*c_2-a_3*c_3
        h1_0=a_0*d_0+b_0*c_0-a_1*d_1-b_1*c_1-a_2*d_2-b_2*c_2-a_3*d_3-b_3*c_3
        h_1=a_0*c_1+a_1*c_0+a_2*c_3-a_3*c_2
        h1_1=a_0*d_1+b_0*c_1+a_1*d_0+b_1*c_0+a_2*d_3+b_2*c_3-a_3*d_2-b_3*c_2
        h_2=a_0*c_2-a_1*c_3+a_2*c_0+a_3*c_1
        h1_2=a_0*d_2+b_0*c_2-a_1*d_3-b_1*c_3+a_2*d_0+b_2*c_0+a_3*d_1+b_3*c_1
        h_3=a_0*c_3+a_1*c_2-a_2*c_1+a_3*c_0
        h1_3=a_0*d_3+b_0*c_3+a_1*d_2+b_1*c_2-a_2*d_1-b_2*c_1+a_3*d_0+b_3*c_0
        h_r_e_a=tf.concat([h_0,h1_0],1) 
        h_r_e_b =tf.concat([h_1,h1_1],1) 
        h_r_e_c=tf.concat([h_2,h1_2],1) 
        h_r_e_d =tf.concat([h_3,h1_3],1)        
        return h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d

    def _build_loss_phase_II(self):

        def _get_kg_score(h_e_a, r_e_a, r_t_e_a, t_e_a, h_e_b, r_e_b, r_t_e_b, t_e_b, h_e_c,
                          r_e_c, r_t_e_c, t_e_c, h_e_d, r_e_d, r_t_e_d, t_e_d,rr1,rr2,rr3,rr4,rt1,rt2,rt3,rt4):
            if self.normal_r == 1:
                denominator_r = tf.sqrt(
                    tf.reduce_sum(r_e_a**2, 1, keepdims=True) +
                    tf.reduce_sum(r_e_b**2, 1, keepdims=True) +
                    tf.reduce_sum(r_e_c**2, 1, keepdims=True) +
                    tf.reduce_sum(r_e_d**2, 1, keepdims=True))
                r_e_a = r_e_a / denominator_r
                r_e_b = r_e_b / denominator_r
                r_e_c = r_e_c / denominator_r
                r_e_d = r_e_d / denominator_r
                denominator_t_r = tf.sqrt(
                    tf.reduce_sum(r_t_e_a**2, 1, keepdims=True) +
                    tf.reduce_sum(r_t_e_b**2, 1, keepdims=True) +
                    tf.reduce_sum(r_t_e_c**2, 1, keepdims=True) +
                    tf.reduce_sum(r_t_e_d**2, 1, keepdims=True))
                r_t_e_a = r_t_e_a / denominator_t_r
                r_t_e_b = r_t_e_b / denominator_t_r
                r_t_e_c = r_t_e_c / denominator_t_r
                r_t_e_d = r_t_e_d / denominator_t_r                

            if self.normal_r == 2:
                denominator_r = tf.sqrt(r_e_a**2 + r_e_b**2 + r_e_c**2 + r_e_d**2)
                r_e_a = r_e_a / denominator_r
                r_e_b = r_e_b / denominator_r
                r_e_c = r_e_c / denominator_r
                r_e_d = r_e_d / denominator_r

                denominator_t_r = tf.sqrt(r_t_e_a**2 + r_t_e_b**2 + r_t_e_c**2 + r_t_e_d**2)
                r_t_e_a = r_t_e_a / denominator_t_r
                r_t_e_b = r_t_e_b / denominator_t_r
                r_t_e_c = r_t_e_c / denominator_t_r
                r_t_e_d = r_t_e_d / denominator_t_r                

                denominator_t_rr = tf.sqrt(rr1**2 + rr2**2 + rr3**2 + rr4**2)
                rr1 = rr1 / denominator_t_rr
                rr2 = rr2 / denominator_t_rr
                rr3 = rr3 / denominator_t_rr
                rr4 = rr4 / denominator_t_rr 

                denominator_t_rt = tf.sqrt(rt1**2 + rt2**2 + rt3**2 + rt4**2)
                rt1 = rt1 / denominator_t_rt
                rt2 = rt2 / denominator_t_rt
                rt3 = rt3 / denominator_t_rt
                rt4 = rt4 / denominator_t_rt 

            h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d =self._hamilton_product(h_e_a, h_e_b, h_e_c, h_e_d, r_e_a, r_e_b, r_e_c, r_e_d)
            #h_r_e_a1, h_r_e_b1, h_r_e_c1, h_r_e_d1  =self._hamilton_product(h_e_a, h_e_b, h_e_c, h_e_d, r_t_e_a, r_t_e_b, r_t_e_c, r_t_e_d)  #t_e_a, t_e_b, t_e_c, t_e_d
            t_r_e_a, t_r_e_b, t_r_e_c, t_r_e_d  =self._hamilton_product(t_e_a, t_e_b, t_e_c, t_e_d, r_t_e_a, r_t_e_b, r_t_e_c, r_t_e_d)  #t_e_a, t_e_b, t_e_c, t_e_d
            
            h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d=self._hamilton_product(h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d,rr1,rr2,rr3,rr4)
            #t_r_e_a1, t_r_e_b1, t_r_e_c1, t_r_e_d1=h_r_e_a1+rt1, h_r_e_b1+rt2, h_r_e_c1+rt3, h_r_e_d1+rt4
            #h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d=self._hamilton_product(h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d, t_r_e_a1, -t_r_e_b1, -t_r_e_c1, -t_r_e_d1)# /(t_r_e_a1**2+t_r_e_b1**2+t_r_e_c1**2+t_r_e_d1**2)
            #temp=(h_r_e_a**2+h_r_e_b**2+h_r_e_c**2+h_r_e_d**2)
            #temp=(t_r_e_a1**2+t_r_e_b**2+t_r_e_c**2+t_r_e_d**2)
            #h_r_e_a2, h_r_e_b2, h_r_e_c2, h_r_e_d2=self._hamilton_product(h_r_e_a,h_r_e_b,h_r_e_c,h_r_e_d,h_r_e_a1,h_r_e_b1,h_r_e_c1,h_r_e_d1)
            #h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d= h_r_e_a2/temp,h_r_e_b2/temp, h_r_e_c2/temp,h_r_e_d2/temp
            #h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d= h_r_e_a*t_r_e_a1/temp, h_r_e_b*t_r_e_b1/temp, h_r_e_c*t_r_e_c1/temp, h_r_e_d*t_r_e_d1/temp
            #h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d  =self._hamilton_product(h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d, h_r_e_a1, h_r_e_b1, h_r_e_c1, h_r_e_d1)  #chuli
            #h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d= h_r_e_a/temp, h_r_e_b/temp, h_r_e_c/temp, h_r_e_d/temp
            #t_r_e_a, t_r_e_b, t_r_e_c, t_r_e_d  =self._hamilton_product(t_e_a, t_e_b, t_e_c, t_e_d, r_t_e_a, r_t_e_b, r_t_e_c, r_t_e_d) 
            #t_r_e_a, t_r_e_b, t_r_e_c, t_r_e_d  =t_e_a, t_e_b, t_e_c, t_e_d
            #h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d =self._hamilton_product(h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d, rr1*0, rr2, rr3, rr4)


            kg_scores_a = tf.reduce_sum(tf.multiply(h_r_e_a, t_r_e_a),
                                        axis=1,
                                        keepdims=True)
            kg_scores_b = tf.reduce_sum(tf.multiply(h_r_e_b, t_r_e_b),
                                        axis=1,
                                        keepdims=True)
            kg_scores_c = tf.reduce_sum(tf.multiply(h_r_e_c, t_r_e_c),
                                        axis=1,
                                        keepdims=True)
            kg_scores_d = tf.reduce_sum(tf.multiply(h_r_e_d, t_r_e_d),
                                        axis=1,
                                        keepdims=True)

            kg_score = kg_scores_a + kg_scores_b + kg_scores_c + kg_scores_d
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e_a, self.r_e_a, self.r_t_e_a, self.pos_t_e_a,
                                     self.h_e_b, self.r_e_b, self.r_t_e_b, self.pos_t_e_b,
                                     self.h_e_c, self.r_e_c, self.r_t_e_c, self.pos_t_e_c,
                                     self.h_e_d, self.r_e_d, self.r_t_e_d, self.pos_t_e_d,self.rr1,self.rr2,self.rr3,self.rr4,self.rt1,self.rt2,self.rt3,self.rt4)

        neg_kg_score = _get_kg_score(self.h_e_a, self.r_e_a, self.r_t_e_a, self.neg_t_e_a,
                                     self.h_e_b, self.r_e_b, self.r_t_e_b, self.neg_t_e_b,
                                     self.h_e_c, self.r_e_c, self.r_t_e_c, self.neg_t_e_c,
                                     self.h_e_d, self.r_e_d, self.r_t_e_d, self.neg_t_e_d,self.rr1,self.rr2,self.rr3,self.rr4,self.rt1,self.rt2,self.rt3,self.rt4)

        kg_loss = tf.reduce_mean(
            tf.nn.softplus(-(pos_kg_score - neg_kg_score)))
        # maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))

        kg_reg_loss = tf.nn.l2_loss(self.h_e_a) + tf.nn.l2_loss(self.r_e_a) + tf.nn.l2_loss(self.r_t_e_a) + \
                      tf.nn.l2_loss(self.h_e_b) + tf.nn.l2_loss(self.r_e_b) + tf.nn.l2_loss(self.r_t_e_b) + \
                      tf.nn.l2_loss(self.h_e_c) + tf.nn.l2_loss(self.r_e_c) + tf.nn.l2_loss(self.r_t_e_c) + \
                      tf.nn.l2_loss(self.h_e_d) + tf.nn.l2_loss(self.r_e_d) + tf.nn.l2_loss(self.r_t_e_d) + \
                      tf.nn.l2_loss(self.pos_t_e_a) + tf.nn.l2_loss(self.neg_t_e_a) + \
                      tf.nn.l2_loss(self.pos_t_e_b) + tf.nn.l2_loss(self.neg_t_e_b) + \
                      tf.nn.l2_loss(self.pos_t_e_c) + tf.nn.l2_loss(self.neg_t_e_c) + \
                      tf.nn.l2_loss(self.pos_t_e_d) + tf.nn.l2_loss(self.neg_t_e_d)

        kg_reg_loss = kg_reg_loss / self.batch_size_kg

        self.kge_loss2 = kg_loss


        self.reg_loss2 = self.reg_2 * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss2)

    def _create_bi_interaction_embed(self):
        A = self.A_in

        A_fold_hat = self._split_A_hat(A)

        ego_embeddings_a = tf.concat(
            [self.weights['user_embed_a'], self.weights['entity_embed_a']],
            axis=0)
        ego_embeddings_b = tf.concat(
            [self.weights['user_embed_b'], self.weights['entity_embed_b']],
            axis=0)
        ego_embeddings_c = tf.concat(
            [self.weights['user_embed_c'], self.weights['entity_embed_c']],
            axis=0)
        ego_embeddings_d = tf.concat(
            [self.weights['user_embed_d'], self.weights['entity_embed_d']],
            axis=0)

        all_embeddings_a = [ego_embeddings_a]
        all_embeddings_b = [ego_embeddings_b]
        all_embeddings_c = [ego_embeddings_c]
        all_embeddings_d = [ego_embeddings_d]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed_a = []
            temp_embed_b = []
            temp_embed_c = []
            temp_embed_d = []

            for f in range(self.n_fold):
                temp_embed_a.append(
                    tf.sparse_tensor_dense_matmul(A_fold_hat[f],
                                                  ego_embeddings_a))
                temp_embed_b.append(
                    tf.sparse_tensor_dense_matmul(A_fold_hat[f],
                                                  ego_embeddings_b))
                temp_embed_c.append(
                    tf.sparse_tensor_dense_matmul(A_fold_hat[f],
                                                  ego_embeddings_c))
                temp_embed_d.append(
                    tf.sparse_tensor_dense_matmul(A_fold_hat[f],
                                                  ego_embeddings_d))


            # [n, 64]
            side_embeddings_a = tf.concat(temp_embed_a, 0)
            side_embeddings_b = tf.concat(temp_embed_b, 0)
            side_embeddings_c = tf.concat(temp_embed_c, 0)
            side_embeddings_d = tf.concat(temp_embed_d, 0)

            # add = e_h + e_{N_h}
            add_embeddings_a = ego_embeddings_a + side_embeddings_a
            add_embeddings_b = ego_embeddings_b + side_embeddings_b
            add_embeddings_c = ego_embeddings_c + side_embeddings_c
            add_embeddings_d = ego_embeddings_d + side_embeddings_d


            sum_embeddings_a = tf.nn.leaky_relu(
                tf.matmul(add_embeddings_a, self.weights['W_gc_a_%d' % k]) +
                self.weights['b_gc_a_%d' % k])
            sum_embeddings_b = tf.nn.leaky_relu(
                tf.matmul(add_embeddings_b, self.weights['W_gc_b_%d' % k]) +
                self.weights['b_gc_b_%d' % k])
            sum_embeddings_c = tf.nn.leaky_relu(
                tf.matmul(add_embeddings_c, self.weights['W_gc_c_%d' % k]) +
                self.weights['b_gc_c_%d' % k])
            sum_embeddings_d = tf.nn.leaky_relu(
                tf.matmul(add_embeddings_d, self.weights['W_gc_d_%d' % k]) +
                self.weights['b_gc_d_%d' % k])

            bi_embeddings_a = tf.multiply(ego_embeddings_a, side_embeddings_a)
            bi_embeddings_b = tf.multiply(ego_embeddings_b, side_embeddings_b)
            bi_embeddings_c = tf.multiply(ego_embeddings_c, side_embeddings_c)
            bi_embeddings_d = tf.multiply(ego_embeddings_d, side_embeddings_d)


            bi_embeddings_a = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_a, self.weights['W_bi_a_%d' % k]) +
                self.weights['b_bi_a_%d' % k])
            bi_embeddings_b = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_b, self.weights['W_bi_b_%d' % k]) +
                self.weights['b_bi_b_%d' % k])
            bi_embeddings_c = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_c, self.weights['W_bi_c_%d' % k]) +
                self.weights['b_bi_c_%d' % k])
            bi_embeddings_d = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings_d, self.weights['W_bi_d_%d' % k]) +
                self.weights['b_bi_d_%d' % k])

            ego_embeddings_a = bi_embeddings_a + sum_embeddings_a
            ego_embeddings_b = bi_embeddings_b + sum_embeddings_b
            ego_embeddings_c = bi_embeddings_c + sum_embeddings_c
            ego_embeddings_d = bi_embeddings_d + sum_embeddings_d

            ego_embeddings_a = tf.nn.dropout(ego_embeddings_a,
                                             1 - self.mess_dropout[k])
            ego_embeddings_b = tf.nn.dropout(ego_embeddings_b,
                                             1 - self.mess_dropout[k])
            ego_embeddings_c = tf.nn.dropout(ego_embeddings_c,
                                             1 - self.mess_dropout[k])
            ego_embeddings_d = tf.nn.dropout(ego_embeddings_d,
                                             1 - self.mess_dropout[k])

            norm_embeddings_a = tf.math.l2_normalize(ego_embeddings_a, axis=1)
            norm_embeddings_b = tf.math.l2_normalize(ego_embeddings_b, axis=1)
            norm_embeddings_c = tf.math.l2_normalize(ego_embeddings_c, axis=1)
            norm_embeddings_d = tf.math.l2_normalize(ego_embeddings_d, axis=1)

            all_embeddings_a += [norm_embeddings_a]
            all_embeddings_b += [norm_embeddings_b]
            all_embeddings_c += [norm_embeddings_c]
            all_embeddings_d += [norm_embeddings_d]


        all_embeddings_a = tf.concat(all_embeddings_a, 1)
        all_embeddings_b = tf.concat(all_embeddings_b, 1)
        all_embeddings_c = tf.concat(all_embeddings_c, 1)
        all_embeddings_d = tf.concat(all_embeddings_d, 1)

        ua_embeddings_a, ea_embeddings_a = tf.split(
            all_embeddings_a, [self.n_users, self.n_entities], 0)
        ua_embeddings_b, ea_embeddings_b = tf.split(
            all_embeddings_b, [self.n_users, self.n_entities], 0)
        ua_embeddings_c, ea_embeddings_c = tf.split(
            all_embeddings_c, [self.n_users, self.n_entities], 0)
        ua_embeddings_d, ea_embeddings_d = tf.split(
            all_embeddings_d, [self.n_users, self.n_entities], 0)

        return ua_embeddings_a, ea_embeddings_a, ua_embeddings_b, ea_embeddings_b, ua_embeddings_c, ea_embeddings_c, ua_embeddings_d, ea_embeddings_d

    def _create_gcn_embed(self):
        A = self.A_in

        A_fold_hat = self._split_A_hat(A)

        embeddings = tf.concat(
            [self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            # A_hat_drop = tf.nn.dropout(A_hat, 1 - self.node_dropout[k], [self.n_users + self.n_items, 1])
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(
                    tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) +
                self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_graphsage_embed(self):
        A = self.A_in
        A_fold_hat = self._split_A_hat(A)

        pre_embeddings = tf.concat(
            [self.weights['user_embed'], self.weights['entity_embed']], axis=0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(
                    tf.sparse_tensor_dense_matmul(A_fold_hat[f],
                                                  pre_embeddings))
            embeddings = tf.concat(temp_embed, 0)

            embeddings = tf.concat([pre_embeddings, embeddings], 1)
            pre_embeddings = tf.nn.relu(
                tf.matmul(embeddings, self.weights['W_mlp_%d' % k]) +
                self.weights['b_mlp_%d' % k])

            pre_embeddings = tf.nn.dropout(pre_embeddings,
                                           1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(
            all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_entities) // self.n_fold

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_attentive_A_out(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(
            tf.SparseTensor(indices, self.A_values, self.A_in.shape))
        return A

    def _generate_quatE_score(self, h, t, r):
        embeddings_a = tf.concat(
            [self.weights['user_embed_a'], self.weights['entity_embed_a']],
            axis=0)
        embeddings_b = tf.concat(
            [self.weights['user_embed_b'], self.weights['entity_embed_b']],
            axis=0)
        embeddings_c = tf.concat(
            [self.weights['user_embed_c'], self.weights['entity_embed_c']],
            axis=0)
        embeddings_d = tf.concat(
            [self.weights['user_embed_d'], self.weights['entity_embed_d']],
            axis=0)

        # embeddings_a = tf.expand_dims(embeddings_a, 1)
        # embeddings_b = tf.expand_dims(embeddings_b, 1)
        # embeddings_c = tf.expand_dims(embeddings_c, 1)
        # embeddings_d = tf.expand_dims(embeddings_d, 1)

        h_e_a = tf.nn.embedding_lookup(embeddings_a, h)
        h_e_b = tf.nn.embedding_lookup(embeddings_b, h)
        h_e_c = tf.nn.embedding_lookup(embeddings_c, h)
        h_e_d = tf.nn.embedding_lookup(embeddings_d, h)

        t_e_a = tf.nn.embedding_lookup(embeddings_a, t)
        t_e_b = tf.nn.embedding_lookup(embeddings_b, t)
        t_e_c = tf.nn.embedding_lookup(embeddings_c, t)
        t_e_d = tf.nn.embedding_lookup(embeddings_d, t)

        # relation embeddings: batch_size * kge_dim
        r_e_a = tf.nn.embedding_lookup(self.weights['relation_embed_a'], r)
        r_e_b = tf.nn.embedding_lookup(self.weights['relation_embed_b'], r)
        r_e_c = tf.nn.embedding_lookup(self.weights['relation_embed_c'], r)
        r_e_d = tf.nn.embedding_lookup(self.weights['relation_embed_d'], r)

        if self.bi_type == 2: 
            r_t_e_a = tf.nn.embedding_lookup(self.weights['relation_t_embed_a'], r)
            r_t_e_b = tf.nn.embedding_lookup(self.weights['relation_t_embed_b'], r)
            r_t_e_c = tf.nn.embedding_lookup(self.weights['relation_t_embed_c'], r)
            r_t_e_d = tf.nn.embedding_lookup(self.weights['relation_t_embed_d'], r)

        if self.bi_type == 1: 
            r_t_e_a = r_e_a
            r_t_e_b = -r_e_b
            r_t_e_c = -r_e_c
            r_t_e_d = -r_e_d

        # lzp: normalize the relation quaternion r to a unit quaternion
        # lzp: normalize the relation quaternion r to a unit quaternion
        # lzp: normalize the relation quaternion r to a unit quaternion
        if self.normal_r == 1:
            denominator_r = tf.sqrt(
                tf.reduce_sum(r_e_a**2, 1, keepdims=True) +
                tf.reduce_sum(r_e_b**2, 1, keepdims=True) +
                tf.reduce_sum(r_e_c**2, 1, keepdims=True) +
                tf.reduce_sum(r_e_d**2, 1, keepdims=True))
            r_e_a = r_e_a / denominator_r
            r_e_b = r_e_b / denominator_r
            r_e_c = r_e_c / denominator_r
            r_e_d = r_e_d / denominator_r
            
            denominator_t_r = tf.sqrt(
                tf.reduce_sum(r_t_e_a**2, 1, keepdims=True) +
                tf.reduce_sum(r_t_e_b**2, 1, keepdims=True) +
                tf.reduce_sum(r_t_e_c**2, 1, keepdims=True) +
                tf.reduce_sum(r_t_e_d**2, 1, keepdims=True))
            r_t_e_a = r_t_e_a / denominator_t_r
            r_t_e_b = r_t_e_b / denominator_t_r
            r_t_e_c = r_t_e_c / denominator_t_r
            r_t_e_d = r_t_e_d / denominator_t_r

        if self.normal_r == 2:
            denominator_r = tf.sqrt(r_e_a**2 + r_e_b**2 + r_e_c**2 + r_e_d**2)
            r_e_a = r_e_a / denominator_r
            r_e_b = r_e_b / denominator_r
            r_e_c = r_e_c / denominator_r
            r_e_d = r_e_d / denominator_r

            denominator_t_r = tf.sqrt(r_t_e_a**2 + r_t_e_b**2 + r_t_e_c**2 + r_t_e_d**2)
            r_t_e_a = r_t_e_a / denominator_t_r
            r_t_e_b = r_t_e_b / denominator_t_r
            r_t_e_c = r_t_e_c / denominator_t_r
            r_t_e_d = r_t_e_d / denominator_t_r            


        h_r_e_a, h_r_e_b, h_r_e_c, h_r_e_d =self._hamilton_product(h_e_a, h_e_b, h_e_c, h_e_d, r_e_a, r_e_b, r_e_c, r_e_d)
        t_r_e_a, t_r_e_b, t_r_e_c, t_r_e_d  =self._hamilton_product(t_e_a, t_e_b, t_e_c, t_e_d, r_t_e_a, r_t_e_b, r_t_e_c, r_t_e_d)

        if self.att_type == 1:
            kg_scores_a = tf.reduce_sum(tf.multiply(tf.tanh(h_r_e_a), tf.tanh(t_r_e_a)),
                                        axis=1)
            kg_scores_b = tf.reduce_sum(tf.multiply(tf.tanh(h_r_e_b), tf.tanh(t_r_e_b)),
                                        axis=1)
            kg_scores_c = tf.reduce_sum(tf.multiply(tf.tanh(h_r_e_c), tf.tanh(t_r_e_c)),
                                        axis=1)
            kg_scores_d = tf.reduce_sum(tf.multiply(tf.tanh(h_r_e_d), tf.tanh(t_r_e_d)),
                                        axis=1)
        elif self.att_type == 0:
            kg_scores_a = tf.reduce_sum(tf.multiply(h_r_e_a, t_r_e_a),
                                        axis=1)
            kg_scores_b = tf.reduce_sum(tf.multiply(h_r_e_b, t_r_e_b),
                                        axis=1)
            kg_scores_c = tf.reduce_sum(tf.multiply(h_r_e_c, t_r_e_c),
                                        axis=1)
            kg_scores_d = tf.reduce_sum(tf.multiply(h_r_e_d, t_r_e_d),
                                        axis=1)
            
        kg_score = kg_scores_a + kg_scores_b + kg_scores_c + kg_scores_d

        return kg_score

    def _statistics_params(self):
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([
            self.opt, self.loss, self.base_loss, self.kge_loss1, self.reg_loss
        ], feed_dict)

    def train_A(self, sess, feed_dict):
        return sess.run(
            [self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return batch_predictions


    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        kg_score = np.array(kg_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)),
                                  shape=(self.n_users + self.n_entities,
                                         self.n_users + self.n_entities))
        if self.alg_type in ['org', 'gcn']:
            self.A_in.setdiag(1.)
