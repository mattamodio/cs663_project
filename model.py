import tensorflow as tf
import os
from utils import lrelu, nameop, tbn, obn
tf.set_random_seed(0)

WASSERSTEIN = False

def conv(x, nfilt, name, reuse, padding='same', k=4, s=2):
    return tf.layers.conv2d(x, filters=nfilt, kernel_size=k, padding=padding, strides=[s,s], 
                            kernel_initializer=tf.truncated_normal_initializer(0,.02), activation=None,
                            name=name, reuse=reuse)

def conv_t(x, nfilt, name, reuse, k=4, s=2):
    return tf.layers.conv2d_transpose(x, filters=nfilt, kernel_size=k, padding='same', strides=[s,s], 
                            kernel_initializer=tf.truncated_normal_initializer(0,.02), activation=None,
                            name=name, reuse=reuse)

def unet_conv(x, nfilt, name, reuse, is_training, s=2, use_batch_norm=True, activation=lrelu):
    x = conv(x, nfilt, name, reuse, s=s)
    if use_batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), training=is_training, reuse=reuse)

    if activation:
        x = activation(x)
    return x

def unet_conv_t(x, encoderx, nfilt, name, reuse, is_training, s=2, use_dropout=True, use_batch_norm=True, activation=tf.nn.relu):
    x = conv_t(x, nfilt, name, reuse, s=s)
    if use_dropout:
        x = tf.layers.dropout(x, .5, training=is_training)

    if use_batch_norm:
        x = batch_norm(x, name='batch_norm_{}'.format(name), training=is_training, reuse=reuse)

    if not encoderx is None:
        x = tf.concat([x,encoderx], 3)

    if activation:
        x = activation(x)
    return x

def batch_norm(tensor, name, reuse, training):

    #return tensor
    #return instance_norm(tensor, name=name, reuse=reuse)
    return tf.layers.batch_normalization(tensor, training=training, momentum=.9, scale=True, name=name, reuse=reuse)

def adversarial_loss(logits, labels):

    #return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    return (logits - labels)**2

def similarity_loss(real, fake):

    return tf.zeros_like(real)

def instance_norm(input, name="instance_norm", reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

class DiscoGAN(object):
    """The DiscoGAN model."""

    def __init__(self,
        dim_b1,
        dim_b2,
        activation=lrelu,
        learning_rate=.0002,
        restore_folder='',
        channels=1,
        attn=False,
        limit_gpu_fraction=.4):
        """Initialize the model."""
        self.dim_b1 = dim_b1
        self.dim_b2 = dim_b2
        self.activation = activation
        self.learning_rate = learning_rate
        self.iteration = 0
        self.channels = channels
        self.attn = attn

        if restore_folder:
            self._restore(restore_folder)
            return

        self.xb1 = tf.placeholder(tf.float32, shape=[None, self.dim_b1*self.dim_b1*channels], name='xb1')
        self.xb2 = tf.placeholder(tf.float32, shape=[None, self.dim_b2*self.dim_b2*channels], name='xb2')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self.init_session(limit_gpu_fraction=limit_gpu_fraction)
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.4, no_gpu=False):
        """Initialize the session."""
        if limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        elif no_gpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def graph_init(self, sess=None):
        """Initialize graph variables."""
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        """Save the model."""
        if not iteration: iteration = self.iteration
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: folder = self.save_folder

        savefile = os.path.join(folder, 'DiscoGAN')
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def _restore(self, restore_folder):
        """Restore the model from a saved checkpoint."""
        tf.reset_default_graph()
        self.init_session()
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        """Construct the DiscoGAN operations."""
        if self.attn:
            self.Attn = AttentionNet(self.dim_b1, channels=self.channels, name='Attn')
            self.predsb1 = self.Attn(self.xb1, is_training=self.is_training)
            self.predsb2 = self.Attn(self.xb2, is_training=self.is_training, reuse=True)
            #TODO: generators want to make their synthetics look like b1/b2 to attn model
            
            self.loss_attn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predsb1, labels=tf.zeros_like(self.predsb1)))
            self.loss_attn += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predsb2, labels=tf.ones_like(self.predsb2)))

            self.attnb1 = tf.gradients(self.loss_attn, self.xb1)[0]
            self.attnb2 = tf.gradients(self.loss_attn, self.xb2)[0]
            
            self.attnb1 = tf.abs(self.attnb1)
            self.attnb1 = self.attnb1 / tf.reduce_sum(self.attnb1, axis=1, keep_dims=True)
            self.attnb1 = self.attnb1 / tf.reduce_max(self.attnb1, axis=1, keep_dims=True)

            self.attnb2 = tf.abs(self.attnb2)
            self.attnb2 = self.attnb2 / tf.reduce_sum(self.attnb2, axis=1, keep_dims=True)
            self.attnb2 = self.attnb2 / tf.reduce_max(self.attnb2, axis=1, keep_dims=True)

            self.attnb1 = nameop(self.attnb1, 'attnb1')
            self.attnb2 = nameop(self.attnb2, 'attnb2')

        self.G12 = GeneratorResnet(self.dim_b1, self.dim_b2, channels=self.channels, name='G12')
        self.Gb2 = self.G12(self.xb1, is_training=self.is_training)
        self.Gb2 = nameop(self.Gb2, 'Gb2')

        self.G21 = GeneratorResnet(self.dim_b2, self.dim_b1, channels=self.channels, name='G21')
        self.Gb1 = self.G21(self.xb2, is_training=self.is_training)
        self.Gb1 = nameop(self.Gb1, 'Gb1')


        self.Gb2_reconstructed = self.G12(self.Gb1, is_training=self.is_training, reuse=True)
        self.Gb1_reconstructed = self.G21(self.Gb2, is_training=self.is_training, reuse=True)

        self.Gb1_reconstructed = nameop(self.Gb1_reconstructed, 'xb1_reconstructed')
        self.Gb2_reconstructed = nameop(self.Gb2_reconstructed, 'xb2_reconstructed')

        self.D1 = Discriminator(self.dim_b1, 1, channels=self.channels, name='D1')
        self.D2 = Discriminator(self.dim_b2, 1, channels=self.channels, name='D2')

        self.D1_probs_z = self.D1(self.xb1, is_training=self.is_training)
        self.D1_probs_G = self.D1(self.Gb1, is_training=self.is_training, reuse=True)
        self.D1_probs_z = nameop(self.D1_probs_z, 'D1_probs_z')
        self.D1_probs_G = nameop(self.D1_probs_G, 'D1_probs_G')

        self.D2_probs_z = self.D2(self.xb2, is_training=self.is_training)
        self.D2_probs_G = self.D2(self.Gb2, is_training=self.is_training, reuse=True)
        self.D2_probs_z = nameop(self.D2_probs_z, 'D2_probs_z')
        self.D2_probs_G = nameop(self.D2_probs_G, 'D2_probs_G')

        self._build_loss()

        self._build_optimization()

    def _build_loss(self):
        """Collect both of the losses."""
        self._build_loss_D()
        self._build_loss_G()
        tf.add_to_collection('losses', self.loss_D)
        tf.add_to_collection('losses', self.loss_G)

    def _build_loss_D(self):
        """Discriminator loss."""
        self.loss_D = 0.

        if not WASSERSTEIN:
            self.loss_D += .5*tf.reduce_mean(adversarial_loss(self.D1_probs_z, tf.ones_like(self.D1_probs_z)))
            self.loss_D += .5*tf.reduce_mean(adversarial_loss(self.D2_probs_z, tf.ones_like(self.D2_probs_z)))

            # Fake example loss for discriminators
            self.loss_D += .5*tf.reduce_mean(adversarial_loss(self.D1_probs_G, tf.zeros_like(self.D1_probs_G)))
            self.loss_D += .5*tf.reduce_mean(adversarial_loss(self.D2_probs_G, tf.zeros_like(self.D2_probs_G)))
        else:
            self.loss_D += tf.reduce_mean(self.D1_probs_z - self.D1_probs_G)
            self.loss_D += tf.reduce_mean(self.D2_probs_z - self.D2_probs_G)

        self.loss_D = nameop(self.loss_D, 'loss_D')

    def _build_loss_G(self):
        """Generator loss."""
        self.loss_G1 = 0.
        self.loss_G2 = 0.

        # fool the discriminator loss
        if not WASSERSTEIN:
            self.loss_G1 += tf.reduce_mean(adversarial_loss(self.D1_probs_G, tf.ones_like(self.D1_probs_G)))
            self.loss_G2 += tf.reduce_mean(adversarial_loss(self.D2_probs_G, tf.ones_like(self.D2_probs_G)))
        else:
            self.loss_G1 += tf.reduce_mean(self.D1_probs_G)
            self.loss_G2 += tf.reduce_mean(self.D2_probs_G)

        # reconstruction loss
        if not self.attn:
            self.loss_G1 += 10*tf.reduce_mean(tf.abs(self.xb1 - self.Gb1_reconstructed))
            self.loss_G2 += 10*tf.reduce_mean(tf.abs(self.xb2 - self.Gb2_reconstructed))
        else:
            self.loss_G1 += 1*tf.reduce_mean(tf.abs(self.xb1*self.attnb1 - self.Gb1_reconstructed*self.attnb1))
            self.loss_G2 += 1*tf.reduce_mean(tf.abs(self.xb2*self.attnb2 - self.Gb2_reconstructed*self.attnb2))

        # identity mapping loss
        self.loss_G1 += 1*tf.reduce_mean(tf.abs(self.G12(self.xb2, is_training=self.is_training, reuse=True) - self.xb2))
        self.loss_G2 += 1*tf.reduce_mean(tf.abs(self.G21(self.xb1, is_training=self.is_training, reuse=True) - self.xb1))

        # similarity loss
        self.loss_G1 += tf.reduce_mean(similarity_loss(self.xb1, self.Gb2))
        self.loss_G2 += tf.reduce_mean(similarity_loss(self.xb2, self.Gb1))

        self.loss_G1 = nameop(self.loss_G1, 'loss_G1')
        self.loss_G2 = nameop(self.loss_G2, 'loss_G2')

        self.loss_G = self.loss_G1 + self.loss_G2
        self.loss_G = nameop(self.loss_G, 'loss_G')

    def _build_optimization(self):
        """Build optimization components."""
        Gvars = [tv for tv in tf.global_variables() if 'G12' in tv.name or 'G21' in tv.name]
        Dvars = [tv for tv in tf.global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        G_update_ops = [op for op in update_ops if 'G12' in op.name or 'G21' in op.name]
        D_update_ops = [op for op in update_ops if 'D1' in op.name or 'D2' in op.name]

        with tf.control_dependencies(G_update_ops):
            optG = tf.train.AdamOptimizer(self.lr, beta1=.5)
            self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        with tf.control_dependencies(D_update_ops):
            optD = tf.train.AdamOptimizer(self.lr, beta1=.5)
            self.train_op_D = optD.minimize(self.loss_D, var_list=Dvars, name='train_op_D')


        if self.attn:
            attnvars = [tv for tv in tf.global_variables() if 'Attn']
            attn_update_ops = [op for op in update_ops if 'Attn']
            optAttn = tf.train.AdamOptimizer(.001, beta1=.5)
            self.train_op_attn = optG.minimize(self.loss_attn, var_list=attnvars, name='train_op_attn')

    def train(self, xb1, xb2):
        """Take a training step with batches from each domain."""
        self.iteration += 1

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('lr:0'): self.learning_rate,
                tbn('is_training:0'): True}

        if self.iteration%2==0:
            _ = self.sess.run([obn('train_op_G')], feed_dict=feed)
        else:
            _ = self.sess.run([obn('train_op_D')], feed_dict=feed)
            if self.attn:
                _ = self.sess.run([obn('train_op_attn')], feed_dict=feed)
 
    def get_layer(self, xb1, xb2, name):
        """Get a layer of the network by name for the entire datasets given in xb1 and xb2."""
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        layer = self.sess.run(tensor, feed_dict=feed)

        return layer

    def get_loss_names(self):
        """Return a string for the names of the loss values."""
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def get_loss(self, xb1, xb2):
        """Return all of the loss values for the given input."""
        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('is_training:0'): False}

        losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)

        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])

        return lstring



class AttentionNet(object):
    def __init__(self,
        input_dim,
        name='',
        channels=1):
        """Initialize a new discriminator."""
        self.input_dim = input_dim
        self.name = name
        self.channels = channels

    def __call__(self, x, is_training, nfilt=32, reuse=False):
        """Return the output of the discriminator."""
        with tf.variable_scope(self.name):
            x = tf.reshape(x, [-1, self.input_dim, self.input_dim, self.channels])

            # attnh1 = unet_conv(x, nfilt*1, 'attnh1', reuse, is_training, use_batch_norm=False)
            # attn1 = unet_conv_t(attnh1, None, 1, 'attn1', reuse, is_training, activation=tf.nn.tanh)

            # attnh2 = unet_conv(attnh1, nfilt*2, 'attnh2', reuse, is_training)
            # attn2 = unet_conv_t(attnh2, None, nfilt*2, 'attn2_1', reuse, is_training)
            # attn2 = unet_conv_t(attn2, None, 1, 'attn2_2', reuse, is_training, activation=tf.nn.tanh)

            # attnh3 = unet_conv(attnh2, nfilt*4, 'attnh3', reuse, is_training)
            # attn3 = unet_conv_t(attnh3, None, nfilt*4, 'attn3_1', reuse, is_training)
            # attn3 = unet_conv_t(attn3, None, nfilt*2, 'attn3_2', reuse, is_training)
            # attn3 = unet_conv_t(attn3, None, 1, 'attn3_3', reuse, is_training, activation=tf.nn.tanh)

            # salience = tf.concat([attn1, attn2, attn3], 3)
            # salience = conv(salience, 1, 'salience', s=1, reuse=reuse)
            # salience = tf.reshape(salience, (-1, self.input_dim*self.input_dim*1))
            # salience = tf.nn.softmax(salience)
            # salience = tf.reshape(salience, (-1, self.input_dim,self.input_dim,1))

            h1 = unet_conv(x, nfilt*1, 'h1', reuse, is_training, use_batch_norm=False)
            h2 = unet_conv(h1, nfilt*2, 'h2', reuse, is_training)
            h3 = unet_conv(h2, nfilt*4, 'h3', reuse, is_training)
            out = unet_conv(h3, 1, 'out', reuse, is_training, use_batch_norm=False, activation=None)

        return out

class GeneratorResnet(object):
    def __init__(self,
        input_dim,
        output_dim,
        name='',
        channels=1):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels

    def __call__(self, x_1d, is_training, nfilt=32, reuse=False):
        image = tf.reshape(x_1d, [-1, self.input_dim, self.input_dim, self.channels])
        with tf.variable_scope(self.name):
            def residule_block(x, dim, k=3, s=1, name='res'):
                p = int((k - 1) / 2)

                y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = conv(y, dim, k=k, s=s, padding='VALID', name=name+'_c1', reuse=reuse)
                y = batch_norm(y, name+'_batch_norm1', reuse=reuse, training=is_training)
                y = tf.nn.relu(y)

                y = tf.pad(y, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
                y = conv(y, dim, k=k, s=s, padding='VALID', name=name+'_c2', reuse=reuse)
                y = batch_norm(y, name+'_batch_norm2', reuse=reuse, training=is_training)

                return y + x

            c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c1 = conv(c0, nfilt, k=7, s=1, padding='VALID', name='e1', reuse=reuse)
            c1 = batch_norm(c1, 'batch_norm_e1', reuse=reuse, training=is_training)
            c1 = tf.nn.relu(c1)

            c2 = conv(c1, nfilt*2, k=3, s=2, name='e2', reuse=reuse)
            c2 = batch_norm(c2, 'batch_norm_e2', reuse=reuse, training=is_training)
            c2 = tf.nn.relu(c2)

            c3 = conv(c2, nfilt*4, k=3, s=2, name='e3', reuse=reuse)
            c3 = batch_norm(c3, 'batch_norm_e3', reuse=reuse, training=is_training)
            c3 = tf.nn.relu(c3)

            r1 = residule_block(c3, nfilt*4, name='res1')
            r2 = residule_block(r1, nfilt*4, name='res2')
            r3 = residule_block(r2, nfilt*4, name='res3')
            r4 = residule_block(r3, nfilt*4, name='res4')
            r5 = residule_block(r4, nfilt*4, name='res5')
            r6 = residule_block(r5, nfilt*4, name='res6')
            r7 = residule_block(r6, nfilt*4, name='res7')
            r8 = residule_block(r7, nfilt*4, name='res8')
            r9 = residule_block(r8, nfilt*4, name='res9')

            d1 = conv_t(r9, nfilt*2, k=3, s=2, name='d1', reuse=reuse)
            d1 = batch_norm(d1, 'batch_norm_d1', reuse=reuse, training=is_training)
            d1 = tf.nn.relu(d1)

            d2 = conv_t(d1, nfilt, k=3, s=2, name='d2', reuse=reuse)
            d2 = batch_norm(d2, 'batch_norm_d2', reuse=reuse, training=is_training)
            d2 = tf.nn.relu(d2)

            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            out = conv(d2, self.channels, k=7, s=1, padding='VALID', name='out', reuse=reuse)
            out = tf.nn.tanh(out)

            out = tf.reshape(out, [-1, self.output_dim*self.output_dim*self.channels])

        return out

class Generator(object):
    """A generator class."""

    def __init__(self,
        input_dim,
        output_dim,
        name='',
        channels=1):
        """Initialize a new generator."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.channels = channels

    def __call__(self, x_1d, is_training, reuse=False, nfilt=32):
        """Return the output of the generator."""
        with tf.variable_scope(self.name):
            x = tf.reshape(x_1d, [-1, self.input_dim, self.input_dim, self.channels])            

            e1 = unet_conv(x, nfilt*1, 'e1', reuse, is_training)
            e2 = unet_conv(e1, nfilt*2, 'e2', reuse, is_training)
            e3 = unet_conv(e2, nfilt*4, 'e3', reuse, is_training)
            e4 = unet_conv(e3, nfilt*8, 'e4', reuse, is_training)
            e5 = unet_conv(e4, nfilt*8, 'e5', reuse, is_training)
            e6 = unet_conv(e5, nfilt*8, 'e6', reuse, is_training, s=1)
            e7 = unet_conv(e6, nfilt*8, 'e7', reuse, is_training, s=1)
            e8 = unet_conv(e7, nfilt*8, 'e8', reuse, is_training, s=1)

            d1 = unet_conv_t(e8, e7, nfilt*8, 'd1', reuse, is_training, s=1)
            d2 = unet_conv_t(d1, e6, nfilt*8, 'd2', reuse, is_training, s=1)
            d3 = unet_conv_t(d2, e5, nfilt*8, 'd3', reuse, is_training, s=1)
            d4 = unet_conv_t(d3, e4, nfilt*8, 'd4', reuse, is_training)
            d5 = unet_conv_t(d4, e3, nfilt*4, 'd5', reuse, is_training)
            d6 = unet_conv_t(d5, e2, nfilt*2, 'd6', reuse, is_training)
            d7 = unet_conv_t(d6, e1, nfilt*1, 'd7', reuse, is_training)
            out = unet_conv_t(d7, None, self.channels, 'out', reuse, is_training, activation=tf.nn.tanh, use_batch_norm=False, use_dropout=False)

            out_1d = tf.reshape(out, (-1, self.output_dim*self.output_dim*self.channels))

        if not reuse:
            print("Generator {} input/output:".format(self.name))
            print(x)
            print(e1)
            print(e2)
            print(e3)
            print(e4)
            print(e5)
            print(e6)
            print(e7)
            print(e8)
            print(d1)
            print(d2)
            print(d3)
            print(d4)
            print(d5)
            print(d6)
            print(d7)
            print(out)
            print(out_1d)
            print("")

        return out_1d

class Discriminator(object):
    """A discriminator class."""

    def __init__(self,
        input_dim,
        output_dim,
        name='',
        channels=1):
        """Initialize a new discriminator."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.channels = channels

    def __call__(self, x, is_training, nfilt=32, reuse=False):
        """Return the output of the discriminator."""
        with tf.variable_scope(self.name):
            x = tf.reshape(x, [-1, self.input_dim, self.input_dim, self.channels])

            h1 = unet_conv(x, nfilt*1, 'h1', reuse, is_training, use_batch_norm=False)
            h2 = unet_conv(h1, nfilt*2, 'h2', reuse, is_training)

            # if self.input_dim==32:
            #     imdim16 = int(self.input_dim/4)
            #     minibatch_features = minibatch(tf.layers.flatten(h2), nfilt*2*imdim16*imdim16, num_kernels=1*imdim16*imdim16, reuse=reuse)
            #     minibatch_features = tf.reshape(minibatch_features, [-1, imdim16, imdim16, 1])
            #     h2 = tf.concat([h2, minibatch_features], 3)

            h3 = unet_conv(h2, nfilt*4, 'h3', reuse, is_training)
            h4 = unet_conv(h3, nfilt*8, 'h4', reuse, is_training)

            out = unet_conv(h4, 1, 'out', reuse, is_training, s=1, use_batch_norm=False, activation=None)
        if not reuse:
            print("Discriminator {} input/output:".format(self.name))
            print(x)
            print(h1)
            print(h2)
            print(h3)
            print(h4)
            print(out)
            print("")

        return out


def minibatch(input_, input_dim, num_kernels=32, kernel_dim=15, name='', reuse=False):
    """Add minibatch features to input."""
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable('{}/Wmb'.format(name), [input_dim, num_kernels * kernel_dim])
        b = tf.get_variable('{}/bmb'.format(name), [num_kernels * kernel_dim])

    x = tf.matmul(input_, W) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_mean(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_mean(tf.exp(-abs_diffs), 2)

    return minibatch_features











































