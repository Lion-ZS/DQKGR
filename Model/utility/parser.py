

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run QKGN.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='last-fm',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1:use pretrained weights npy.')
    parser.add_argument('--pretrain_ckpt', type=int, default=0,
                    help='0: No pretrain, 1:use pretrained ckpt.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')

    parser.add_argument('--show_step', type=int, default=1,
                        help='The criteria of show performance.')
 
    parser.add_argument('--early_stop', type=int, default=50,
                        help='The criteria of early stopping.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')

    parser.add_argument('--layer_size', nargs='?', default='[]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048,
                        help='KG batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')

    parser.add_argument('--reg', type=float, default=1e-5,
                        help='Regularization for user and item embeddings.')

    parser.add_argument('--reg1', type=float, default=1e-1,
                        help='Regularization for KGE.')

    parser.add_argument('--reg2', type=float, default=1e-5,
                        help='Regularization for user and item embeddings.')                        

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='qkgn',
                        help='Specify a loss type from {kgat, qkgn}.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='bi',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=1,
                        help='Specify a gpu id.')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[1,5,10,20,50,100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver, -1: Save embeddings of mf, -2: Save embeddings of kgat, -3: Save embeddings of qkgn.')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=int, default=0,
                        help='whether using attention mechanism')

    parser.add_argument('--use_kge', type=int, default=1,
                        help='whether using knowledge graph embedding')

    parser.add_argument('--score_func', type=int, default=0,
                        help='user-item score function type, 0: quaternion inner product, 1: quaternion hamilton product')

    parser.add_argument('--user_num', type=int, default=-1,
                        help='the number of user used in training and test, default(-1) represents using all the users')

    parser.add_argument('--normal_r', type=int, default=2,
                        help='the way to normalize W_r, 0: without normalization, 1: my normalization, 2: nips normalization')

    parser.add_argument('--initial', type=int, default=1,
                        help='the way to initialize quaternions, 0: xavier, 1: quaternion-valued networks')

    parser.add_argument('--att_type', type=int, default=1,
                        help='attention type, 1: with tanh, 0: without tanh')

    parser.add_argument('--bi_type', type=int, default=0,
                        help='bi_rotation type, 1: rotate tail by conjugate of r, 2: rotate tail by annother representation of r')
    
    return parser.parse_args()