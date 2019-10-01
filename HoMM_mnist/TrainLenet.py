import os
import pickle
from Lenet import *
from Utils import *
import scipy.io
import numpy as np
from tensorflow.contrib import slim
from center_loss import *
os.environ['CUDA_VISIBLE_DEVICES']='0'



class Train():
    def __init__(self,class_num,batch_size,iters,learning_rate,keep_prob,param):
        self.ClassNum=class_num
        self.BatchSize=batch_size
        self.Iters=iters
        self.LearningRate=learning_rate
        self.KeepProb=keep_prob
        self.discriminative_loss_param=param[0]
        self.domain_loss_param=param[1]
        self.adver_loss_param=param[2]
        self.ring_loss_param=param[3]
        self.target_loss_param =tf.get_variable(name="target_loss_param",shape=None,dtype=tf.float32,initializer=tf.constant(0.0),trainable=False)
        self.ring_loss_param=tf.get_variable(name="ring_loss_param",shape=None,dtype=tf.float32,initializer=tf.constant(0.0),trainable=False)
        self.ring_norm = tf.get_variable(name="ring_norm", shape=None, dtype=tf.float32, initializer=tf.constant(10.0),trainable=False)
        self.logits_threshold=tf.get_variable(name="logits_threshold",shape=None,dtype=tf.float32,initializer=tf.constant(0.9),trainable=False)


        self.SourceData,self.SourceLabel=load_svhn('svhn')
        self.TargetData, self.TargetLabel=load_mnist('mnist')
        self.TestData, self.TestLabel = load_mnist('mnist',split='test')
        # self.EdgeWeights=Label2EdgeWeights(self.SourceLabel)
        # self.EdgeWeights=zeros((self.SourceLabel.shape[0],self.SourceLabel.shape[0]))
        ###################################################################################




        #######################################################################################
        self.source_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 32,32,3],name="source_image")
        self.source_label = tf.placeholder(tf.float32, shape=[self.BatchSize, self.ClassNum],name="source_label")
        self.target_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 32, 32,1],name="target_image")
        self.Training_flag = tf.placeholder(tf.bool, shape=None,name="Training_flag")
        self.W = tf.placeholder(tf.float32, shape=[self.BatchSize, self.BatchSize])



    def TrainNet(self):
        self.source_model=Lenet(inputs=self.source_image, training_flag=self.Training_flag, reuse=False)
        self.target_model=Lenet(inputs=self.target_image, training_flag=self.Training_flag, reuse=True)
        self.CalLoss()
        varall=tf.trainable_variables()
        with tf.control_dependencies([self.centers_update_op]):
            self.solver = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss)
        self.source_prediction = tf.argmax(self.source_model.softmax_output, 1)
        self.target_prediction = tf.argmax(self.target_model.softmax_output, 1)
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.SourceLabel=sess.run(tf.one_hot(self.SourceLabel,10))
            self.TestLabel=sess.run(tf.one_hot(self.TestLabel,10))
            true_num = 0.0
            for step in range(self.Iters):
                i= step % int(self.SourceData.shape[0]/self.BatchSize)
                j= step % int(self.TargetData.shape[0]/self.BatchSize)
                source_batch_x = self.SourceData[i * self.BatchSize: (i + 1) * self.BatchSize]
                source_batch_y = self.SourceLabel[i * self.BatchSize: (i + 1) * self.BatchSize]
                target_batch_x = self.TargetData[j * self.BatchSize: (j + 1) * self.BatchSize]
                # W = self.EdgeWeights[i * self.BatchSize: (i + 1) * self.BatchSize,i * self.BatchSize: (i + 1) * self.BatchSize]
                total_loss, source_loss,domain_loss, clustering_loss, source_prediction,_= sess.run(
                    fetches=[self.loss, self.source_loss, self.domain_loss, self.Clustering_loss, self.source_prediction, self.solver],
                    feed_dict={self.source_image: source_batch_x, self.source_label: source_batch_y,self.target_image: target_batch_x, self.Training_flag: True})
                true_label = argmax(source_batch_y, 1)
                true_num = true_num + sum(true_label == source_prediction)


                if step % 200 ==0:
                    print "Iters-{} ### TotalLoss={} ### SourceLoss={} ###DomainLoss={} ### ClusteringLoss={}".format(
                        step, total_loss, source_loss, domain_loss, clustering_loss)
                    train_accuracy = true_num / (200*self.BatchSize)
                    true_num = 0.0
                    print " ########## train_accuracy={} ###########".format(train_accuracy)
                    self.Test(sess)


                if step==30000:
                    sess.run(tf.assign(self.target_loss_param,0.1))
                    sess.run(tf.assign(self.logits_threshold,0.92))


                # if step!=0 and step % 1000 == 0:
                #     self.SourceData, self.SourceLabel= shuffle0(self.SourceData, self.SourceLabel)

                # if step!=0 and step % 5000 == 0:
                #    self.SourceData, self.SourceLabel, self.EdgeWeights= shuffle(self.SourceData, self.SourceLabel, self.EdgeWeights)




    def CalLoss(self):
        self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label, logits=self.source_model.fc5)
        self.source_loss = tf.reduce_mean(self.source_cross_entropy)
        self.CalDomainLoss(method="HoMM3")
        self.DisClustering()
        self.CalTargetLoss("Entropy")
        # self.CalAdver()
        self.loss=self.source_loss+self.domain_loss_param*self.domain_loss+self.target_loss_param*self.Clustering_loss



    def CalDomainLoss(self,method):
        if method=="MMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            diff=tf.reduce_mean(Xs, 0, keep_dims=False) - tf.reduce_mean(Xt, 0, keep_dims=False)
            self.domain_loss=tf.reduce_sum(tf.multiply(diff,diff))


        elif method=="KMMD":
            Xs=self.source_model.fc4
            Xt=self.target_model.fc4
            self.domain_loss=tf.maximum(0.0001,KMMD(Xs,Xt))


        elif method=="CORAL":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=self.coral_loss(Xs,Xt)


        elif method =='LCORAL':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=self.log_coral_loss(Xs,Xt)


        elif method=='SqrtCORAL':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss = self.sqrt_coral_loss(Xs, Xt)

        elif method=='HoMM3':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss = self.HoMM3_loss(Xs, Xt)


        elif method=='HoMM4':
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss = self.HoMM4(Xs, Xt)

        elif method=="HoMM":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss = self.HoMM(Xs, Xt,order=3,num=300000)


        elif method=="KHoMM":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss = self.KHoMM(Xs, Xt, order=3, num=10000)




        elif method =="mmatch":
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.domain_loss=mmatch(Xs,Xt,5)

        elif method == "CorNorm":
            gamma=0.001 # 0.001  #0.0003 mnist->mnist-m
            Xs = self.source_model.fc4
            Xt = self.target_model.fc4
            self.CorNorm(Xs,Xt,gamma)



    def CalTargetLoss(self,method):
        if method=="Entropy":
            source_softmax=self.source_model.softmax_output
            target_softmax=self.target_model.softmax_output
            target_softmax_mean=tf.reduce_mean(target_softmax,axis=0)
            index=tf.where(tf.equal(tf.greater(target_softmax,self.logits_threshold),True))[:,0]
            target_softmax=tf.gather(target_softmax,index)
            self.target_loss_1=tf.reduce_mean(target_softmax_mean*tf.log(tf.clip_by_value(target_softmax_mean,0.0001,1)))
            self.target_loss_2=-tf.reduce_mean(tf.reduce_sum(target_softmax * tf.log(tf.clip_by_value(target_softmax,0.0001,1)), axis=1))
            self.target_loss=0.0*self.target_loss_1+self.target_loss_2




    def CalDiscriminativeLoss(self,method):
        if method=="InstanceBased":
            Xs = self.source_model.fc4
            norm = lambda x: tf.reduce_sum(tf.square(x), 1)
            self.F0 = tf.transpose(norm(tf.expand_dims(Xs, 2) - tf.transpose(Xs)))
            margin0 = 0
            margin1 = 100
            F0=tf.pow(tf.maximum(0.0, self.F0-margin0),2)
            F1=tf.pow(tf.maximum(0.0, margin1-self.F0),2)
            self.intra_loss=tf.reduce_mean(tf.multiply(F0, self.W))
            self.inter_loss=tf.reduce_mean(tf.multiply(F1, 1.0-self.W))
            self.discriminative_loss = (self.intra_loss+self.inter_loss) / (self.BatchSize * self.BatchSize)



        elif method=="CenterBased":
            Xs=self.source_model.fc4
            labels=tf.argmax(self.source_label,1)
            self.inter_loss, self.intra_loss, self.centers_update_op = get_center_loss(Xs, labels, 0.5, 10)
            self.intra_loss=self.intra_loss/(self.ClassNum*self.BatchSize+self.ClassNum*self.ClassNum)
            self.inter_loss=self.inter_loss/(self.ClassNum*self.BatchSize+self.ClassNum*self.ClassNum)
            self.discriminative_loss = self.intra_loss+ self.inter_loss



    def DisClustering(self):
        target_feature, target_logits = SelectTargetSamples(self.target_model.fc4, self.target_model.fc5, logits_threshold=self.logits_threshold)
        target_pseudo_label = tf.argmax(target_logits, axis=1)
        with tf.variable_scope('target'):
            self.inter_loss, self.intra_loss, self.centers_update_op = get_center_loss(target_feature,target_pseudo_label,0.5,10)
            self.Clustering_loss=self.intra_loss/(self.ClassNum*self.BatchSize)



    def Cal_RingLoss(self):
        Xs=self.source_model.fc4
        Xs_norm=tf.norm(Xs,ord="euclidean",axis=1)
        Xt = self.target_model.fc4
        Xt_norm = tf.norm(Xt, ord="euclidean", axis=1)
        self.ring_loss=tf.reduce_mean(tf.norm(Xs_norm-self.ring_norm))+tf.reduce_mean(tf.norm(Xt_norm-self.ring_norm))





    def coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size = self.BatchSize
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,
                                                         transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # cov_source=tf.linalg.cholesky(cov_source)
        # cov_target=tf.linalg.cholesky(cov_target)
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))





    def log_coral_loss(self, h_src, h_trg, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = float(self.BatchSize)
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(h_src, h_src,transpose_a=True)  + gamma * tf.eye(64)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(h_trg, h_trg,transpose_a=True)  + gamma * tf.eye(64)
        # eigen decomposition
        eig_source = tf.self_adjoint_eig(cov_source)
        eig_target = tf.self_adjoint_eig(cov_target)
        log_cov_source = tf.matmul(eig_source[1],tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
        log_cov_target = tf.matmul(eig_target[1],tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))
        return tf.reduce_mean(tf.square(tf.subtract(log_cov_source, log_cov_target)))


    def sqrt_coral_loss(self, xs, xt, gamma=1e-3):
        # regularized covariances result in inf or nan
        # First: subtract the mean from the data matrix
        batch_size = float(self.BatchSize)
        xs = xs - tf.reduce_mean(xs, axis=0)
        xt = xt - tf.reduce_mean(xt, axis=0)
        cov_source = (1. / (batch_size - 1)) * tf.matmul(xs, xs,transpose_a=True)  #+ gamma * tf.eye(64)
        cov_target = (1. / (batch_size - 1)) * tf.matmul(xt, xt,transpose_a=True)  #+ gamma * tf.eye(64)
        sqrt_cov_source=self.CovSqrt(cov_source)
        sqrt_cov_target=self.CovSqrt(cov_target)
        return tf.reduce_mean(tf.square(tf.subtract(sqrt_cov_source, sqrt_cov_target)))



    def CovSqrt(self,Cov):
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


    def HoMM3_loss(self, xs, xt):
        xs = xs - tf.reduce_mean(xs, axis=0)
        # xs=self.decoupling(xs)
        xt = xt - tf.reduce_mean(xt, axis=0)
        # xt=self.decoupling(xt)
        xs=tf.expand_dims(xs,axis=-1)
        xs = tf.expand_dims(xs, axis=-1)
        xt = tf.expand_dims(xt, axis=-1)
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


    def decoupling(self,X):
        dim=X.shape[0]
        Ones=0.0001*tf.ones(shape=[dim,1])
        Y=tf.concat([Ones,X],axis=1)
        return Y



    def HoMM4(self,xs,xt):
        ind=tf.range(tf.cast(xs.shape[1],tf.int32))
        ind=tf.random_shuffle(ind)
        xs=tf.transpose(xs,[1,0])
        xs=tf.gather(xs,ind)
        xs = tf.transpose(xs, [1, 0])
        xt = tf.transpose(xt, [1, 0])
        xt = tf.gather(xt, ind)
        xt = tf.transpose(xt, [1, 0])
        return self.HoMM4_loss(xs[:,:30],xt[:,:30])+self.HoMM4_loss(xs[:,30:60],xt[:,30:60])+self.HoMM4_loss(xs[:,60:90],xt[:,60:90])



    def HoMM4_loss(self, xs, xt):
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
    def HoMM(self,xs, xt, order=3, num=300000):
        xs = xs - tf.reduce_mean(xs, axis=0)
        xt = xt - tf.reduce_mean(xt, axis=0)
        dim = tf.cast(xs.shape[1], tf.int32)
        index = tf.random_uniform(shape=(num, dim), minval=0, maxval=dim - 1, dtype=tf.int32)
        index = index[:, :order]
        xs = tf.transpose(xs)
        xs = tf.gather(xs, index)  ##dim=[num,order,batchsize]
        xt = tf.transpose(xt)
        xt = tf.gather(xt, index)
        HO_Xs = tf.reduce_prod(xs, axis=1)
        HO_Xs = tf.reduce_mean(HO_Xs, axis=1)
        HO_Xt = tf.reduce_prod(xt, axis=1)
        HO_Xt = tf.reduce_mean(HO_Xt, axis=1)
        return tf.reduce_mean(tf.square(tf.subtract(HO_Xs, HO_Xt)))




    ####
    def KHoMM(self, xs, xt, order=3, num=10000):
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
        KHoMM = KernelHoMM(Ho_Xs, Ho_Xt, sigma=0.00001)
        return KHoMM




    def CorNorm(self,Xs,Xt,gamma):
        Xs = tf.transpose(Xs - tf.reduce_mean(Xs, axis=0))
        Xt = tf.transpose(Xt - tf.reduce_mean(Xt, axis=0))
        norm = lambda x: tf.reduce_sum(tf.square(x), 1)
        self.Xs_Cor_norm = tf.transpose(norm(tf.expand_dims(Xs, 2) - tf.transpose(Xs)))
        self.Xt_Cor_norm = tf.transpose(norm(tf.expand_dims(Xt, 2) - tf.transpose(Xt)))
        self.W1 = tf.exp(-gamma * self.Xs_Cor_norm)
        self.W2 = tf.exp(-gamma * self.Xt_Cor_norm)
        # self.W1=self.CovSqrt(self.W1)
        # self.W2 = self.CovSqrt(self.W2)
        # self.W1=tf.multiply(tf.sign(self.W1),tf.sqrt(tf.abs(self.W1)+1e-12))
        # self.W2 = tf.multiply(tf.sign(self.W2), tf.sqrt(tf.abs(self.W2)+1e-12))
        # self.W1=self.W1/(tf.norm(self.W1))
        # self.W2=self.W2/(tf.norm(self.W2))
        self.domain_loss = tf.reduce_mean(tf.square(tf.subtract(self.W1, self.W2)))





    def Test(self,sess):
        true_num=0.0
        num = int(self.TestData.shape[0] / self.BatchSize)
        total_num=num*self.BatchSize
        for i in range (num):
            k = i % int(self.TestData.shape[0] / self.BatchSize)
            target_batch_x = self.TestData[k * self.BatchSize: (k + 1) * self.BatchSize]
            target_batch_y= self.TestLabel[k * self.BatchSize: (k + 1) * self.BatchSize]
            prediction=sess.run(fetches=self.target_prediction, feed_dict={self.target_image:target_batch_x, self.Training_flag: False})
            true_label = argmax(target_batch_y, 1)
            true_num+=sum(true_label==prediction)
        accuracy=true_num / total_num
        print "###########  Test Accuracy={} ##########".format(accuracy)





def main():
    discriminative_loss_param = 0.0 ##0.03 for InstanceBased method, 0.01 for CenterBased method
    domain_loss_param = 10000   # 8.0 for Coral; 1000 for CorNorm, SqrtCORAL, 10000(3000) for TensorCORAL
    adver_loss_param=0.0
    ring_loss_param=0.0  ## svhn-> mnist 0.0005 ## mnist-> mnistm 0.0003 ## USPS-> mnist 0.0005  ## syn -> mnist 0.0005  ## 0.00001
    param=[discriminative_loss_param, domain_loss_param,adver_loss_param,ring_loss_param]
    Runer=Train(class_num=10,batch_size=128,iters=200000,learning_rate=0.0001,keep_prob=1,param=param)
    Runer.TrainNet()




def load_mnist(image_dir, split='train'):
    print ('Loading MNIST dataset.')

    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f)
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']
    labels=np.squeeze(labels).astype(int)
    return images,labels


def load_svhn(image_dir, split='train'):
    print ('Loading SVHN dataset.')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    # ~ images= resize_images(images)
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels


def load_USPS(image_dir,split='train'):
    print('Loading USPS dataset.')
    image_file='USPS_train.pkl' if split=='train' else 'USPS_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f)
    images = usps['data']
    images=np.reshape(images,[-1,32,32,1])
    labels = usps['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels



def load_syn(image_dir,split='train'):
    print('load syn dataset')
    image_file='synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'
    image_dir=os.path.join(image_dir,image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = syn['y'].reshape(-1)
    return images,labels


def load_mnistm(image_dir,split='train'):
    print('Loading mnistm dataset.')
    image_file='mnistm_train.pkl' if split=='train' else 'mnistm_test.pkl'
    image_dir=os.path.join(image_dir,image_file)
    with open(image_dir, 'rb') as f:
        mnistm = pickle.load(f)
    images = mnistm['data']

    labels = mnistm['label']
    labels=np.squeeze(labels).astype(int)
    return images,labels


if __name__=="__main__":
    main()