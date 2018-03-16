import tensorflow as tf
from utils import lrelu, nameop, tbn, obn
tf.set_random_seed(0)


class DiscoGAN(object):
    """The DiscoGAN model."""

    def __init__(self,
        dim_b1,
        dim_b2,
        activation=lrelu,
        learning_rate=.0001,
        restore_folder=''):
        """Initialize the model."""
        self.dim_b1 = dim_b1
        self.dim_b2 = dim_b2
        self.activation = activation
        self.learning_rate = learning_rate
        self.iteration = 0

        if restore_folder:
            self._restore(restore_folder)
            return

        self.xb1 = tf.placeholder(tf.float32, shape=[None, dim_b1], name='xb1')
        self.xb2 = tf.placeholder(tf.float32, shape=[None, dim_b2], name='xb2')

        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        self._build()
        self.init_session()
        self.graph_init(self.sess)

    def init_session(self, limit_gpu_fraction=.3, no_gpu=False):
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
        self.G12 = Generator(self.dim_b1, self.dim_b2, name='G12')
        self.Gb2 = self.G12(self.xb1, is_training=self.is_training)
        nameop(self.Gb2, 'Gb2')

        self.G21 = Generator(self.dim_b2, self.dim_b1, name='G21')
        self.Gb1 = self.G21(self.xb2, is_training=self.is_training)
        nameop(self.Gb1, 'Gb1')


        self.Gb2_reconstructed = self.G12(self.Gb1, is_training=self.is_training, reuse=True)
        self.Gb1_reconstructed = self.G21(self.Gb2, is_training=self.is_training, reuse=True)

        self.Gb1_reconstructed = nameop(self.Gb1_reconstructed, 'xb1_reconstructed')
        self.Gb2_reconstructed = nameop(self.Gb2_reconstructed, 'xb2_reconstructed')

        self.D1 = Discriminator(self.dim_b1, 1, name='D1')
        self.D2 = Discriminator(self.dim_b2, 1, name='D2')

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

        self.loss_D += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_z, labels=tf.ones_like(self.D1_probs_z)))
        self.loss_D += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_z, labels=tf.ones_like(self.D2_probs_z)))

        # Fake example loss for discriminators
        self.loss_D += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.zeros_like(self.D1_probs_G)))
        self.loss_D += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.zeros_like(self.D2_probs_G)))

        self.loss_D = nameop(self.loss_D, 'loss_D')

    def _build_loss_G(self):
        """Generator loss."""
        self.loss_G1 = 0.
        self.loss_G2 = 0.

        # fool the discriminator loss
        self.loss_G1 += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_probs_G, labels=tf.ones_like(self.D1_probs_G)))
        self.loss_G2 += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_probs_G, labels=tf.ones_like(self.D2_probs_G)))

        # reconstruction loss
        self.loss_G1 += tf.reduce_mean((self.xb1 - self.Gb1_reconstructed)**2)
        self.loss_G2 += tf.reduce_mean((self.xb2 - self.Gb2_reconstructed)**2)

        self.loss_G1 = nameop(self.loss_G1, 'loss_G1')
        self.loss_G2 = nameop(self.loss_G2, 'loss_G2')

        self.loss_G = self.loss_G1 + self.loss_G2
        self.loss_G = nameop(self.loss_G, 'loss_G')

    def _build_optimization(self):
        """Build optimization components."""
        Gvars = [tv for tv in tf.global_variables() if 'G12' in tv.name or 'G21' in tv.name]
        Dvars = [tv for tv in tf.global_variables() if 'D1' in tv.name or 'D2' in tv.name]

        optG = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.9)
        self.train_op_G = optG.minimize(self.loss_G, var_list=Gvars, name='train_op_G')

        optD = tf.train.AdamOptimizer(self.lr, beta1=.5, beta2=.9)
        self.train_op_D = optD.minimize(self.loss_D, var_list=Dvars, name='train_op_D')

    def train(self, xb1, xb2):
        """Take a training step with batches from each domain."""
        self.iteration += 1

        feed = {tbn('xb1:0'): xb1,
                tbn('xb2:0'): xb2,
                tbn('lr:0'): self.learning_rate,
                tbn('is_training:0'): True}

        if self.iteration % 2 == 0:
            _ = self.sess.run([obn('train_op_D')], feed_dict=feed)
        else:
            _ = self.sess.run([obn('train_op_G')], feed_dict=feed)

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





class Generator(object):
    """A generator class."""

    def __init__(self,
        input_dim,
        output_dim,
        name='',
        activation=tf.nn.relu):
        """Initialize a new generator."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.name = name

    def __call__(self, x, is_training, reuse=False):
        """Return the output of the generator."""
        with tf.variable_scope(self.name):
            x = tf.reshape(x, [-1, 28, 28, 1])

            h1 = tf.layers.conv2d(x, filters=16, kernel_size=4, padding='same', strides=2, activation=None, name='h1', reuse=reuse)
            h1 = lrelu(h1)

            h2 = tf.layers.conv2d(h1, filters=32, kernel_size=4, padding='same', strides=1, activation=None, name='h2', reuse=reuse)
            # h2 = tf.contrib.layers.batch_norm(h2, is_training=is_training, scale=True, decay=0.9, scope=self.name + 'bn_h2' + str(int(reuse)))
            h2 = lrelu(h2)

            h3 = tf.layers.conv2d(h2, filters=64, kernel_size=4, padding='same', strides=1, activation=None, name='h3', reuse=reuse)
            # h3 = tf.contrib.layers.batch_norm(h3, is_training=is_training, scale=True, decay=0.9, scope=self.name + 'bn_h3' + str(int(reuse)))
            h3 = lrelu(h3)

            h1_t = tf.layers.conv2d_transpose(h3, filters=16, kernel_size=4, padding='same', strides=1, activation=None, name='h1_t', reuse=reuse)
            # h1_t = tf.contrib.layers.batch_norm(h1_t, is_training=is_training, scale=True, decay=0.9, scope=self.name + 'bn_h1_t' + str(int(reuse)))
            h1_t = lrelu(h1_t)

            h2_t = tf.layers.conv2d_transpose(h1_t, filters=8, kernel_size=4, padding='same', strides=1, activation=None, name='h2_t', reuse=reuse)
            # h2_t = tf.contrib.layers.batch_norm(h2_t, is_training=is_training, scale=True, decay=0.9, scope=self.name + 'bn_h2_t' + str(int(reuse)))
            h2_t = lrelu(h2_t)

            h3_t = tf.layers.conv2d_transpose(h2_t, filters=1, kernel_size=4, padding='same', strides=2, activation=None, name='h3_t', reuse=reuse)


            out = tf.reshape(h3_t, [-1, 28 * 28])

        if not reuse:
            print("Generator {} input/output:".format(self.name))
            print(x)
            print(h1)
            print(h2)
            print(h3)
            print(h1_t)
            print(h2_t)
            print(h3_t)
            print(out)
            print("")

        return out

class Discriminator(object):
    """A discriminator class."""

    def __init__(self,
        input_dim,
        output_dim,
        name=''):
        """Initialize a new discriminator."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

    def __call__(self, x, is_training, reuse=False):
        """Return the output of the discriminator."""
        with tf.variable_scope(self.name):
            fc1 = tf.layers.dense(x, 1024, activation=None, reuse=reuse, name='fc1')
            fc1 = lrelu(fc1)

            fc2 = tf.layers.dense(fc1, 512, activation=None, reuse=reuse, name='fc2')
            fc2 = minibatch(fc2, 512, name=self.name + '2', reuse=reuse)
            fc2 = lrelu(fc2)

            fc3 = tf.layers.dense(fc2, 256, activation=None, reuse=reuse, name='fc3')
            fc3 = lrelu(fc3)

            out = tf.layers.dense(fc3, 1, activation=None, reuse=reuse, name='out')

        if not reuse:
            print("Discriminator {} input/output:".format(self.name))
            print(x)
            print(fc1)
            print(fc2)
            print(fc3)
            print(out)
            print("")

        return out


def minibatch(input_, input_dim, num_kernels=15, kernel_dim=15, name='', reuse=False):
    """Add minibatch features to input."""
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable('{}/Wmb'.format(name), [input_dim, num_kernels * kernel_dim])
        b = tf.get_variable('{}/bmb'.format(name), [num_kernels * kernel_dim])

    x = tf.matmul(input_, W) + b

    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_mean(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_mean(tf.exp(-abs_diffs), 2)

    return tf.concat([input_, minibatch_features], 1)















