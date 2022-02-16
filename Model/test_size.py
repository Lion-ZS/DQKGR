
import tensorflow as tf
from utility.helper import *
from utility.batch_test import *
from time import time
from QKGN import QKGN

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_pretrained_data(args):
    pre_model = 'qkgn'
    pretrain_path = '%spretrain/%s_d%d/%s.npz' % (args.proj_path, args.dataset, args.embed_size,
                                              pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained qkgn model parameters.')
    except Exception:
        print(
            "---In fact no pretrained .npz file can be used as in %spretrain/%s_d%d/%s---"
            % (args.proj_path, args.dataset, args.embed_size, pre_model))
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    # get argument settings.
    tf.set_random_seed(2019)
    np.random.seed(2019)
    args = parse_args()
    # for lr in [64]:
    #     args.embed_size = size
    for size in [128,8,16,32]:
        args.embed_size = size
        # args.layer_size = []
        for reg1 in [1.0]:
            args.reg1 = reg1
            for reg2 in [1e-4]:
                args.reg2 = reg2
                print("-------Parse_args: ------------")
                print(args)
                print("-------------------------------")
                print("lambda1: ", args.reg1)
                print("lambda2: ", args.reg2)
                print("Embedding Size: ", args.embed_size)
                if args.gpu_id != -1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
                tf.reset_default_graph()
                # else:        
                #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                #     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                """
                *********************************************************
                Load Data from data_generator function.
                """
                config = dict()
                config['n_users'] = data_generator.n_users
                config['n_items'] = data_generator.n_items
                config['n_relations'] = data_generator.n_relations
                config['n_entities'] = data_generator.n_entities

                if args.model_type in ['qkgn']:
                    "Load the laplacian matrix."
                    config['A_in'] = sum(data_generator.lap_list)

                    "Load the KG triplets."
                    config['all_h_list'] = data_generator.all_h_list
                    config['all_r_list'] = data_generator.all_r_list
                    config['all_t_list'] = data_generator.all_t_list
                    config['all_v_list'] = data_generator.all_v_list

                print("------Success: Load Data from data_generator function--------.")
                t0 = time()
                """
                *********************************************************
                Use the pretrained data to initialize the embeddings.
                """
                if args.pretrain in [-1]:
                    pretrain_data = load_pretrained_data(args)
                    print("------Try to use the pretrained .npz file to initialize the embeddings-------.")
                else:
                    pretrain_data = None

                """
                *********************************************************
                Select one of the models.
                """
                if args.model_type in ['qkgn']:
                    model = QKGN(data_config=config,
                                pretrain_data=pretrain_data,
                                args=args)
                saver = tf.train.Saver()
                """
                *********************************************************
                Save the model parameters.
                """
                if args.save_flag == 1:
                    if args.model_type in ['qkgn']:
                        layer = '-'.join([str(l) for l in eval(args.layer_size)])
                        weights_save_path = '%sweights/%s/%s_d%d/%s/l%s_r%s' % (
                            args.weights_path, args.dataset, model.model_type, args.embed_size, layer,
                            str(args.lr), '-'.join([str(r) for r in eval(args.reg)]))

                    ensureDir(weights_save_path)
                    save_saver = tf.train.Saver(max_to_keep=1)
                    print("------Success: Create save_saver for saving model-------.")

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                """
                *********************************************************
                Reload the model parameters to fine tune.
                """
                if args.pretrain_ckpt == 1:
                    print("-------Trying to find pretrained ckpt file of model-------")
                    if args.model_type in ['qkgn']:
                        layer = '-'.join([str(l) for l in eval(args.layer_size)])
                        pretrain_path = '%sweights/%s/%s_d%d/%s/l%s_r%s' % (
                            args.weights_path, args.dataset, model.model_type, args.embed_size, layer,
                            str(args.lr), '-'.join([str(r) for r in eval(args.reg)]))

                    ckpt = tf.train.get_checkpoint_state(
                        os.path.dirname(pretrain_path + '/checkpoint'))

                    if ckpt and ckpt.model_checkpoint_path:
                        sess.run(tf.global_variables_initializer())
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('-------Sucess: load the pretrained ckpt parameters from: ',
                            pretrain_path)

                        # *********************************************************
                        # get the performance from the model to fine tune.
                        if args.report != 1:
                            users_to_test = list(data_generator.test_user_dict.keys())

                            ret = test(sess,
                                    model,
                                    users_to_test,
                                    drop_flag=False,
                                    batch_test_flag=batch_test_flag)
                            cur_best_pre_0 = ret['recall'][3]

                            pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                                        'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                                        (ret['recall'][0], ret['recall'][-1],
                                            ret['precision'][0], ret['precision'][-1],
                                            ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                            ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
                            print(pretrain_ret)
                    else:
                        sess.run(tf.global_variables_initializer())
                        cur_best_pre_0 = 0.
                        print('without pretraining (No useable ckpt).')
                else:
                    sess.run(tf.global_variables_initializer())
                    cur_best_pre_0 = 0.
                    print('without pretraining (Agrs.pretrain != -1).')
                """
                *********************************************************
                Get the final performance w.r.t. different sparsity levels.
                """
                # in kgat and qkgn, args.report = default(0)
                if args.report == 1:
                    assert args.test_flag == 'full'
                    users_to_test_list, split_state = data_generator.get_sparsity_split()
                    # lzp: users_to_test_list contains users of different sparsity level, and the last one contains all the test users
                    users_to_test_list.append(list(data_generator.test_user_dict.keys()))
                    split_state.append('all')

                    save_path = '%sreport/%s/%s_d%d.result' % (args.proj_path, args.dataset,
                                                        model.model_type, args.embed_size)
                    ensureDir(save_path)
                    f = open(save_path, 'w')
                    f.write('embed_size=%d, lr=%.4f, reg1=%s, reg1=%s, loss_type=%s, \n' %
                            (args.embed_size, args.lr, args.reg1, args.reg2, args.loss_type))

                    for i, users_to_test in enumerate(users_to_test_list):
                        ret = test(sess,
                                model,
                                users_to_test,
                                drop_flag=False,
                                batch_test_flag=batch_test_flag)

                        final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s], auc=[%.5f]" % \
                                    ('\t'.join(['%.5f' % r for r in ret['recall']]),
                                    '\t'.join(['%.5f' % r for r in ret['precision']]),
                                    '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                                    '\t'.join(['%.5f' % r for r in ret['ndcg']]),
                                    ret['auc'])
                        print(final_perf)

                        f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
                    f.close()
                    exit()
                """
                *********************************************************
                Train.
                """
                loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, auc_loger = [], [], [], [], [], []
                stopping_step = 0
                should_stop = False

                for epoch in range(args.epoch):
                    if epoch == 0:
                        print("------Success: Begin training-------.")
                    t1 = time()
                    loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
                    n_batch = data_generator.n_train // args.batch_size + 1
                    if epoch == 0:
                        print("Total batch number of QKGN recommender: ", n_batch)
                    """
                    *********************************************************
                    Alternative Training for QKGN:
                    ... phase 1: to train the recommender.
                    """
                    for idx in range(n_batch):
                        #if idx == 0:
                        #print("------Success: Begin first batch of first epoch in phase 1:-------.")
                        btime = time()
                        batch_data = data_generator.generate_train_batch(n_batch)
                        feed_dict = data_generator.generate_train_feed_dict(
                            model, batch_data)

                        _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(
                            sess, feed_dict=feed_dict)

                        loss += batch_loss
                        base_loss += batch_base_loss
                        kge_loss += batch_kge_loss
                        reg_loss += batch_reg_loss

                    if np.isnan(loss) == True:
                        print('ERROR: loss@phase1 is nan.')
                        sys.exit()
                    """
                    *********************************************************
                    Alternative Training for QKGN:
                    ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
                    """
                    if args.model_type in ['qkgn']:

                        n_A_batch = len(
                            data_generator.all_h_list) // args.batch_size_kg + 1

                        if epoch == 0:
                            print("Total batch number of QKGE: ", n_A_batch)

                        if args.use_kge == 1:
                            if epoch == 0:
                                print("------Begin try to use quaternion kge-------.")
                            # using KGE method (knowledge graph embedding).
                            for idx in range(n_A_batch):
                                # if idx == 0:
                                # print("------Success: Begin first batch of first epoch in phase 2:-------.")
                                btime = time()

                                A_batch_data = data_generator.generate_train_A_batch(n_A_batch)
                                feed_dict = data_generator.generate_train_A_feed_dict(
                                    model, A_batch_data)

                                _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(
                                    sess, feed_dict=feed_dict)

                                loss += batch_loss
                                kge_loss += batch_kge_loss
                                reg_loss += batch_reg_loss

                        if args.use_att == 1:
                            # updating attentive laplacian matrix.
                            if epoch == 0:
                                print("------Begin try to update_attentive_A-------.")
                            model.update_attentive_A(sess)

                    if np.isnan(loss) == True:
                        print('ERROR: loss@phase2 is nan.')
                        sys.exit()

                    show_step = args.show_step
                    if (epoch + 1) % show_step != 0:
                        if args.verbose > 0 and epoch % args.verbose == 0:
                            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                                epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
                            print(perf_str)
                        continue
                    """
                    *********************************************************
                    Test.
                    """
                    print("--Epoch: %d----Success: Begin test! -------." % epoch)
                    t2 = time()
                    users_to_test = list(data_generator.test_user_dict.keys())

                    ret = test(sess,
                            model,
                            users_to_test,
                            drop_flag=False,
                            batch_test_flag=batch_test_flag)
                    """
                    *********************************************************
                    Performance logging.
                    """
                    t3 = time()

                    loss_loger.append(loss)
                    rec_loger.append(ret['recall'])
                    pre_loger.append(ret['precision'])
                    ndcg_loger.append(ret['ndcg'])
                    hit_loger.append(ret['hit_ratio'])
                    auc_loger.append(ret['auc'])

                    if args.verbose > 0:
                        perf_str_e = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % \
                                (epoch, t2 - t1, t3 - t2, loss, base_loss, kge_loss, reg_loss)
                        print(perf_str_e)                        
                        perf_str = "hit=[%s]\nprecision=[%s]\nrecall=[%s]\nndcg=[%s]\nauc=[%.5f]" % \
                                    (', '.join(['%.5f' % r for r in ret['hit_ratio']]),
                                    ', '.join(['%.5f' % r for r in ret['precision']]),
                                    ', '.join(['%.5f' % r for r in ret['recall']]),
                                    ', '.join(['%.5f' % r for r in ret['ndcg']]),
                                    ret['auc'])
                        print(perf_str)

                    cur_best_pre_0, stopping_step, should_stop = early_stopping(
                        ret['recall'][3],
                        cur_best_pre_0,
                        stopping_step,
                        expected_order='acc',
                        flag_step=args.early_stop)

                    # *********************************************************
                    # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
                    if should_stop == True:
                        break

                    # *********************************************************
                    # save the user & item embeddings for pretraining.
                    if ret['recall'][3] == cur_best_pre_0 and args.save_flag == 1:
                        save_saver.save(sess,
                                        weights_save_path + '/weights',
                                        global_step=epoch)
                        print('Saved the ckpt model in path: ', weights_save_path)

                        # save the pretrained model parameters of qkgn (i.e., user & item & kg embeddings) for pretraining other models.
                        user_embed_a, user_embed_b, user_embed_c, user_embed_d, \
                        entity_embed_a, entity_embed_b, entity_embed_c, entity_embed_d, \
                        relation_embed_a, relation_embed_b, relation_embed_c, relation_embed_d = sess.run(
                            [
                                model.weights['user_embed_a'],
                                model.weights['user_embed_b'],
                                model.weights['user_embed_c'],
                                model.weights['user_embed_d'],
                                model.weights['entity_embed_a'],
                                model.weights['entity_embed_b'],
                                model.weights['entity_embed_c'],
                                model.weights['entity_embed_d'],
                                model.weights['relation_embed_a'],
                                model.weights['relation_embed_b'],
                                model.weights['relation_embed_c'],
                                model.weights['relation_embed_d']
                            ],
                            feed_dict={})

                        temp_save_path = '%spretrain/%s_d%d/%s.npz' % (
                            args.proj_path, args.dataset, args.embed_size, args.model_type)
                        ensureDir(temp_save_path)
                        np.savez(temp_save_path,
                                user_embed_a=user_embed_a,
                                user_embed_b=user_embed_b,
                                user_embed_c=user_embed_c,
                                user_embed_d=user_embed_d,
                                entity_embed_a=entity_embed_a,
                                entity_embed_b=entity_embed_b,
                                entity_embed_c=entity_embed_c,
                                entity_embed_d=entity_embed_d,
                                relation_embed_a=relation_embed_a,
                                relation_embed_b=relation_embed_b,
                                relation_embed_c=relation_embed_c,
                                relation_embed_d=relation_embed_d)
                        print('Saved the .npz weights of qkgn in path: ', temp_save_path)


                recs = np.array(rec_loger)

                pres = np.array(pre_loger)
                ndcgs = np.array(ndcg_loger)
                hit = np.array(hit_loger)

                aucs = np.array(auc_loger)

                best_rec_0 = max(recs[:, 3])
                idx = list(recs[:, 3]).index(best_rec_0)

                final_perf = "Best Iter=[%d]@[%.1f]\nrecall=[%s]\n precision=[%s]\n hit=[%s]\n ndcg=[%s]\n auc=[%.5f]" % \
                            (idx, time() - t0, ', '.join(['%.5f' % r for r in recs[idx]]),
                            ', '.join(['%.5f' % r for r in pres[idx]]),
                            ', '.join(['%.5f' % r for r in hit[idx]]),
                            ', '.join(['%.5f' % r for r in ndcgs[idx]]),
                            aucs[idx])

                print(final_perf)

                save_path = '%soutput/%s_d%d/%s.result' % (args.proj_path, args.dataset, args.embed_size,
                                                    model.model_type)
                ensureDir(save_path)
                f = open(save_path, 'a')
                
                f.write(str(args))
                f.write(final_perf)
                f.close()
                sess.close()
    exit()
