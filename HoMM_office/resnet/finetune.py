import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor
os.environ['CUDA_VISIBLE_DEVICES']='0'
from Utils import *
from center_loss import *
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.0001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01


tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs',1000, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes',31, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 70, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', "adapt,fc,scale5/block3", 'Finetuning layers, seperated by commas')  #scale5/block3
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/amazon.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/webcam.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))


    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    source = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    target = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    is_training = tf.placeholder('bool', [])
    dropout_rate=tf.placeholder(dtype=tf.float32,shape=None)

    domain_loss_param=tf.get_variable(name="domain_loss_param",dtype=tf.float32,initializer=tf.constant(1.0),trainable=False)
    target_loss_param=tf.get_variable(name='target_loss_param',dtype=tf.float32,initializer=tf.constant(0.0),trainable=False)
    logits_threshold=tf.get_variable(name='logits_threshold',dtype=tf.float32,initializer=tf.constant(0.0),trainable=False)
    ring_norm = tf.get_variable(name="fc/ring_norm", shape=None, dtype=tf.float32, initializer=tf.constant(100.0),trainable=False)
    clustering_param=tf.get_variable(name='Ortho_loss_param',dtype=tf.float32,initializer=tf.constant(0.0),trainable=False)

    # Model
    train_layers = FLAGS.train_layers.split(',')
    source_model = ResNetModel(source,is_training, depth=FLAGS.resnet_depth, dropout_rate=dropout_rate,num_classes=FLAGS.num_classes)
    target_model = ResNetModel(target,is_training,reuse=True, depth=FLAGS.resnet_depth, dropout_rate=dropout_rate,num_classes=FLAGS.num_classes)
    # fc_weights=tf.get_default_graph().get_tensor_by_name("fc/weights:0")
    # Orthogonal_regularizer=tf.reduce_mean(tf.norm(tf.matmul(tf.transpose(fc_weights),fc_weights)-tf.eye(FLAGS.num_classes),ord=2))
    # Grad_loss=GradRegularization(target_model.prob,target_model.avg_pool)

### Calculating the loss function

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=source_model.prob, labels=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    source_loss = cross_entropy_mean+1.0*regularization_losses
    domain_loss= HoMM(source_model.adapt,target_model.adapt,order=4,num=300000)
    target_loss=Target_loss(tf.nn.softmax(target_model.prob),logits_threshold)
    centers_update_op,discriminative_loss,source_centers=CenterBased(source_model.adapt,y)
    target_feature,target_logits=SelectTargetSamples(target_model.adapt,target_model.prob,logits_threshold=0.75)
    target_predict_label=tf.argmax(target_logits,axis=1)
    target_pseudo_label=tf.one_hot(target_predict_label,FLAGS.num_classes)
    with tf.variable_scope('target'):
        centers_update_op_1, discriminative_clustering,target_centers = CenterBased(target_feature, target_pseudo_label)
    # class_domain_loss=AlignCenter(centers_update_op,centers_update_op_1)
    ring_loss = Cal_RingLoss(ring_norm,source_model.avg_pool,target_model.avg_pool)


# office 1000 0.01  0.0003 ## office-Home 1000 0.01 0.001
    loss=source_loss+200*domain_loss_param*domain_loss+clustering_param*discriminative_clustering




    # train_op = model.optimize(FLAGS.learning_rate, train_layers)
    Varall=tf.trainable_variables()
    # print(Varall)
    trainable_var_names = ['weights', 'biases', 'beta', 'gamma','adapt']
    # var_list_1 = [v for v in tf.trainable_variables() if v.name.split(':')[0].split('/')[-1] in trainable_var_names and contains(v.name, train_layers)]
    var_list_1 = [var for var in tf.trainable_variables() if 'scale5/block3' in var.name]
    var_list_2=[var for var in tf.trainable_variables() if 'fc' in var.name or 'adapt' in var.name]
    var_list_3 = [var for var in tf.trainable_variables() if 'scale5/block2' in var.name]

    Varall = tf.trainable_variables()
    optimizer1 = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=0.0003)
    # optimizer3 = tf.train.AdamOptimizer(learning_rate=0.000005)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([centers_update_op,centers_update_op_1]):
            op1 = optimizer1.minimize(loss,var_list=var_list_1)
            op2 = optimizer2.minimize(loss,var_list=var_list_2)
            # op3 = optimizer3.minimize(loss,var_list=var_list_3)
            train_op=tf.group(op1,op2)



    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(source_model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    train_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.training_file, num_classes=FLAGS.num_classes,
                                           output_size=[224, 224], horizontal_flip=False, shuffle=True, multi_scale=multi_scale)

    target_preprocessor = BatchPreprocessor(dataset_file_path='../data/webcam.txt', num_classes=FLAGS.num_classes,output_size=[224, 224],shuffle=True)

    val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file, num_classes=FLAGS.num_classes, output_size=[224, 224])

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    target_batches_per_epoch = np.floor(len(target_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)

    # train_batches_per_epoch=np.minimum(train_batches_per_epoch,target_batches_per_epoch)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        varall=tf.trainable_variables()
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)
        # Load the pretrained weights
        source_model.load_original_weights(sess, skip_layers=train_layers)
        # target_model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print("{} Start training...".format(datetime.datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.datetime.now(), tensorboard_dir))


        Acc_convergency=[]
        for epoch in range(FLAGS.num_epochs):
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            step = 1
            # Start training
            while step < train_batches_per_epoch:
                if step%target_batches_per_epoch==0:
                    target_preprocessor.reset_pointer()
                batch_xs, batch_ys = train_preprocessor.next_batch(FLAGS.batch_size)
                batch_xt, batch_yt = target_preprocessor.next_batch(FLAGS.batch_size)
                TotalLoss, SourceLoss, DomainLoss, TargetLoss, RingLoss, _=sess.run(
                    fetches=[loss, source_loss, domain_loss, target_loss, ring_loss, train_op],
                    feed_dict={source: batch_xs,target:batch_xt, y: batch_ys, is_training: True,dropout_rate:1.0})

                ############################ print loss ##################################################
                print "Loss={} ### SourceLoss={} ### DomainLoss={} ### TargetLoss={} ### RingLoss={}".format(TotalLoss, SourceLoss, DomainLoss, TargetLoss, RingLoss)

                # Logging
                # if step % FLAGS.log_step == 0:
                #     s = sess.run(merged_summary, feed_dict={source: batch_xs, y: batch_ys, is_training: False})
                #     train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                step += 1

            if epoch % 3 == 0 :

                # Epoch completed, start validation
                print("{} Start validation".format(datetime.datetime.now()))
                test_acc = 0.
                test_count = 0

                for _ in range(val_batches_per_epoch):
                    batch_tx, batch_ty = val_preprocessor.next_batch(FLAGS.batch_size)
                    acc= sess.run(accuracy, feed_dict={source: batch_tx, y: batch_ty, is_training: False,dropout_rate:1.0})
                    test_acc += acc
                    test_count += 1

                test_acc /= test_count
                s = tf.Summary(value=[tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)])
                val_writer.add_summary(s, epoch+1)
                print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
                Acc_convergency.append(test_acc)
                print Acc_convergency


            if epoch==100:
                sess.run(tf.assign(clustering_param, 0.0))






            # Reset the dataset pointers
            val_preprocessor.reset_pointer()
            train_preprocessor.reset_pointer()
            target_preprocessor.reset_pointer()

####################### log the convergency data#######################
        savedata = np.array(Acc_convergency)
        np.save("AtoD_SDDA_Source.npy", savedata)
            #
            # print("{} Saving checkpoint of model...".format(datetime.datetime.now()))
            #
            # #save checkpoint of the model
            # checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            # save_path = saver.save(sess, checkpoint_path)
            #
            # print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))

def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv

def CORAL(h_src,h_trg,batchsize=128, gamma=1e-3):
    # regularized covariances (D-Coral is not regularized actually..)
    # First: subtract the mean from the data matrix
    batch_size = batchsize
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
    cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
    # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
    # The reduce_mean account for the factor 1/d^2
    return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))

def SqrtCORAL(h_src,h_trg):
    batch_size = 160.0
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src, transpose_a=True)
    cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg, transpose_a=True)
    sqrt_cov_source = CovSqrt(cov_source)
    sqrt_cov_target = CovSqrt(cov_target)
    return tf.reduce_mean(tf.square(tf.subtract(sqrt_cov_source, sqrt_cov_target)))



def HoMM3(h_src, h_trg):
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    xs=tf.expand_dims(h_src,axis=-1)
    xs = tf.expand_dims(xs, axis=-1)
    xt = tf.expand_dims(h_trg, axis=-1)
    xt = tf.expand_dims(xt, axis=-1)
    xs_1=tf.transpose(xs,[0,2,1,3])
    xs_2 = tf.transpose(xs, [0, 2, 3, 1])
    xt_1 = tf.transpose(xt, [0, 2, 1, 3])
    xt_2 = tf.transpose(xt, [0, 2, 3, 1])
    HR_Xs=xs*xs_1*xs_2
    HR_Xs=tf.reduce_mean(HR_Xs,axis=0)
    HR_Xt = xt * xt_1 * xt_2
    HR_Xt = tf.reduce_mean(HR_Xt, axis=0)
    return tf.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))


def HoMM4(xs,xt):
    ind=tf.range(tf.cast(xs.shape[1],tf.int32))
    ind=tf.random_shuffle(ind)
    xs=tf.transpose(xs,[1,0])
    xs=tf.gather(xs,ind)
    xs=tf.transpose(xs,[1,0])
    xt = tf.transpose(xt, [1, 0])
    xt=tf.gather(xt,ind)
    xt=tf.transpose(xt,[1,0])
    return HoMM4_loss(xs[:,0:30], xt[:,0:30])+HoMM4_loss(xs[:,30:60], xt[:,30:60])+HoMM4_loss(xs[:,60:90], xt[:,60:90])\
           +HoMM4_loss(xs[:,90:120], xt[:,90:120])+HoMM4_loss(xs[:,120:150], xt[:,120:150])+HoMM4_loss(xs[:,150:180], xt[:,150:180])




def HoMM4_loss(xs, xt):
    xs = xs - tf.reduce_mean(xs, axis=0)
    xt = xt - tf.reduce_mean(xt, axis=0)
    xs = tf.expand_dims(xs,axis=-1)
    xs = tf.expand_dims(xs, axis=-1)
    xs = tf.expand_dims(xs, axis=-1)
    xt = tf.expand_dims(xt, axis=-1)
    xt = tf.expand_dims(xt, axis=-1)
    xt = tf.expand_dims(xt, axis=-1)
    xs_1 = tf.transpose(xs,[0,2,1,3,4])
    xs_2 = tf.transpose(xs, [0, 2, 3, 1,4])
    xs_3 = tf.transpose(xs, [0, 2, 3, 4, 1])
    xt_1 = tf.transpose(xt, [0, 2, 1, 3,4])
    xt_2 = tf.transpose(xt, [0, 2, 3, 1,4])
    xt_3 = tf.transpose(xt, [0, 2, 3, 4, 1])
    HR_Xs=xs*xs_1*xs_2*xs_3
    HR_Xs=tf.reduce_mean(HR_Xs,axis=0)
    HR_Xt = xt * xt_1 * xt_2*xt_3
    HR_Xt = tf.reduce_mean(HR_Xt, axis=0)
    return tf.reduce_mean(tf.square(tf.subtract(HR_Xs, HR_Xt)))



## high-order moment matching
def HoMM(xs,xt,order=4,num=100000):
    xs = xs - tf.reduce_mean(xs, axis=0)
    xt = xt - tf.reduce_mean(xt, axis=0)
    dim=tf.cast(xs.shape[1], tf.int32)
    index=tf.random_uniform(shape=(num,dim),minval=0,maxval=dim-1,dtype=tf.int32,seed=0)
    index=index[:,:order]
    xs=tf.transpose(xs)
    xs=tf.gather(xs,index)  ##dim=[num,order,batchsize]
    xt = tf.transpose(xt)
    xt = tf.gather(xt, index)
    HO_Xs=tf.reduce_prod(xs,axis=1)
    HO_Xs=tf.reduce_mean(HO_Xs,axis=1)
    HO_Xt = tf.reduce_prod(xt, axis=1)
    HO_Xt = tf.reduce_mean(HO_Xt, axis=1)
    return tf.reduce_mean(tf.square(tf.subtract(HO_Xs, HO_Xt)))




####
def KHoMM(xs,xt,order=4,num=30000):
    xs = xs - tf.reduce_mean(xs, axis=0)
    xt = xt - tf.reduce_mean(xt, axis=0)
    dim = tf.cast(xs.shape[1], tf.int32)
    index = tf.random_uniform(shape=(num, dim), minval=0, maxval=dim - 1, dtype=tf.int32, seed=0)
    index = index[:, :order]
    xs = tf.transpose(xs)
    xs = tf.gather(xs, index)  ##dim=[num,order,batchsize]
    xt = tf.transpose(xt)
    xt = tf.gather(xt, index)
    Ho_Xs = tf.transpose(tf.reduce_prod(xs, axis=1))
    Ho_Xt = tf.transpose(tf.reduce_prod(xt, axis=1))
    KHoMM=KernelHoMM(Ho_Xs,Ho_Xt,sigma=0.00001)
    return KHoMM




def CovSqrt(Cov):
    Cov=Cov/tf.trace(Cov)
    Y0=Cov
    Z0=tf.eye(int(Cov.shape[0]))
    I=tf.eye(int(Cov.shape[0]))
    Y1=0.5*tf.matmul(Y0,3*I-tf.matmul(Z0,Y0))
    Z1=0.5*tf.matmul(3*I-tf.matmul(Z0,Y0),Z0)
    Y2 = 0.5 * tf.matmul(Y1, 3 * I - tf.matmul(Z1, Y1))
    Z2 = 0.5 * tf.matmul(3 * I - tf.matmul(Z1, Y1), Z1)
    Y3 = 0.5 * tf.matmul(Y2, 3 * I - tf.matmul(Z2, Y2))
    Z3 = 0.5 * tf.matmul(3 * I - tf.matmul(Z2, Y2), Z2)
    Y4 = 0.5 * tf.matmul(Y3, 3 * I - tf.matmul(Z3, Y3))
    Z4 = 0.5 * tf.matmul(3 * I - tf.matmul(Z3, Y3), Z3)
    Y5 = 0.5 * tf.matmul(Y4, 3 * I - tf.matmul(Z4, Y4))
    Z5 = 0.5 * tf.matmul(3 * I - tf.matmul(Z4, Y4), Z4)
    Y6 = 0.5 * tf.matmul(Y5, 3 * I - tf.matmul(Z5, Y5))
    Y6 = tf.multiply(tf.sign(Y6), tf.sqrt(tf.abs(Y6)+1e-12))
    Y6=Y6/tf.norm(Y6)
    return Y6


def Target_loss(target_softmax,logits_threshold):
    index=tf.where(tf.equal(tf.greater(target_softmax,logits_threshold),True))[:,0]
    target_softmax=tf.gather(target_softmax,index)
    return -tf.reduce_mean(tf.reduce_sum(target_softmax * tf.log(tf.clip_by_value(target_softmax,0.0001,1)), axis=1))


def GradRegularization(target_y,target_x):
    gradient=tf.gradients(target_y,target_x)[0]
    loss=tf.reduce_sum(tf.square(gradient))
    return loss


def AlignCenter(source_center,target_center):
    num=10000
    order=4
    dim = tf.cast(source_center.shape[1], tf.int32)
    index = tf.random_uniform(shape=(num, dim), minval=0, maxval=dim - 1, dtype=tf.int32, seed=0)
    index = index[:, :order]
    source_center = tf.transpose(source_center)
    source_center = tf.gather(source_center, index)
    target_center = tf.transpose(target_center)
    target_center = tf.gather(target_center, index)
    HO_source_center = tf.reduce_prod(source_center, axis=1)
    HO_target_center = tf.reduce_prod(target_center, axis=1)
    class_domain_loss = tf.reduce_mean(tf.norm(HO_source_center - HO_target_center, axis=0))
    # class_domain_loss = tf.reduce_mean(tf.norm(source_center - target_center, axis=1))
    return class_domain_loss


def SelectTargetSamples(features,logits,logits_threshold):
    target_softmax=tf.nn.softmax(logits)
    index = tf.where(tf.equal(tf.greater(target_softmax, logits_threshold), True))[:, 0]
    target_softmax = tf.gather(target_softmax, index)
    target_feature = tf.gather(features, index)
    return target_feature,target_softmax


def MMD(xs,xt):
    diff = tf.reduce_mean(xs, 0, keep_dims=False) - tf.reduce_mean(xt, 0, keep_dims=False)
    test=tf.multiply(diff, diff)
    loss=tf.reduce_sum(tf.multiply(diff, diff))
    return tf.reduce_sum(tf.multiply(diff, diff))


def MMD1(xs,xt):
    diff=xs-xt
    test=tf.matmul(diff,tf.transpose(diff))
    loss=tf.reduce_mean(tf.matmul(diff,tf.transpose(diff)))
    return loss



def log_coral_loss(h_src, h_trg,batch_size=128,gamma=1e-3):
    # regularized covariances result in inf or nan
    # First: subtract the mean from the data matrix
    batch_size = float(batch_size)
    h_src = h_src - tf.reduce_mean(h_src, axis=0)
    h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
    cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,transpose_a=True)  + gamma * tf.eye(128)
    cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg, transpose_a=True)  + gamma * tf.eye(128)
    # eigen decomposition
    eig_source = tf.self_adjoint_eig(cov_source)
    eig_target = tf.self_adjoint_eig(cov_target)
    log_cov_source = tf.matmul(eig_source[1],tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
    log_cov_target = tf.matmul(eig_target[1],tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))
    return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))



def CenterBased(fc4,y):
    source_label=y
    Xs = fc4
    labels = tf.argmax(source_label, 1)
    inter_loss, intra_loss, centers_update_op,centers = get_center_loss(Xs, labels, 0.2, FLAGS.num_classes)
    discriminative_loss = intra_loss
    discriminative_loss = discriminative_loss / (FLAGS.num_classes  * FLAGS.batch_size)
    return centers_update_op, discriminative_loss,centers


def CorNorm_loss(h_src,h_trg):
    gamma = 0.0001   # 0.001
    Xs = h_src
    Xt = h_trg
    Xs = tf.transpose(Xs - tf.reduce_mean(Xs, axis=0))
    Xt = tf.transpose(Xt - tf.reduce_mean(Xt, axis=0))
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    Xs_Cor_norm = tf.transpose(norm(tf.expand_dims(Xs, 2) - tf.transpose(Xs)))
    Xt_Cor_norm = tf.transpose(norm(tf.expand_dims(Xt, 2) - tf.transpose(Xt)))
    W1 = tf.exp(-gamma * Xs_Cor_norm)
    W2 = tf.exp(-gamma * Xt_Cor_norm)
    W1 = W1 / (tf.reduce_mean(W1))
    W2 = W2 / (tf.reduce_mean(W2))
    domain_loss = tf.reduce_mean(tf.square(tf.subtract(W1, W2)))
    return domain_loss


def Cal_RingLoss(ring_norm,Xs,Xt):
    Xs_norm=tf.norm(Xs,ord="euclidean",axis=1)
    Xt_norm = tf.norm(Xt, ord="euclidean", axis=1)
    ring_loss=tf.reduce_mean(tf.norm(Xs_norm-ring_norm))+tf.reduce_mean(tf.norm(Xt_norm-ring_norm))
    return ring_loss



if __name__ == '__main__':
    tf.app.run()
