import tensorflow as tf


class Transformer:

    def __init__(self, hparams):
        self.hparams = hparams

    def soft_max_weighted_sum(self, align, value, key_masks, drop_out, is_training, future_binding=False):
        """
        :param align:           [batch_size, None, time]
        :param value:           [batch_size, time, units]
        :param key_masks:       [batch_size, None, time]
                                2nd dim size with align
        :param drop_out:
        :param is_training:
        :param future_binding:  TODO: only support 2D situation at present
        :return:                weighted sum vector
                                [batch_size, None, units]
        """
        # exp(-large) -> 0
        paddings = tf.fill(tf.shape(align), float('-inf'))
        # [batch_size, None, time]
        align = tf.where(key_masks, align, paddings)

        if future_binding:
            length = tf.reshape(tf.shape(value)[1], [-1])
            # [time, time]
            lower_tri = tf.ones(tf.concat([length, length], axis=0))
            # [time, time]
            lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
            # [batch_size, time, time]
            masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
            # [batch_size, time, time]
            align = tf.where(tf.equal(masks, 0), paddings, align)

        # soft_max and dropout
        # [batch_size, None, time]
        align = tf.nn.softmax(align)
        align = tf.layers.dropout(align, drop_out, training=is_training)
        # weighted sum
        # [batch_size, None, units]
        return tf.matmul(align, value)

    def learned_positional_encoding(self, inputs, max_length, num_units):
        outputs = tf.range(tf.shape(inputs)[1])  # (T_q)
        outputs = tf.expand_dims(outputs, 0)     # (1, T_q)
        outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])  # (N, T_q)
        with tf.variable_scope("embeddings") as scope:
            pos_embedding = tf.get_variable(name="pos_embedding", shape=[max_length, num_units],
                                            dtype=tf.float32)
            encoded = tf.nn.embedding_lookup(pos_embedding, outputs)
        return encoded

    def pointwise_feedforward(self, inputs, drop_out, is_training, num_units=None, activation=None):
        # Inner layer
        # outputs = tf.layers.conv1d(inputs, num_units[0], kernel_size=1, activation=activation)
        outputs = tf.layers.dense(inputs, num_units[0], activation=activation)
        outputs = tf.layers.dropout(outputs, drop_out, training=is_training)
        # Readout layer
        # outputs = tf.layers.conv1d(outputs, num_units[1], kernel_size=1, activation=None)
        outputs = tf.layers.dense(outputs, num_units[1], activation=None)

        # drop_out before add&norm
        outputs = tf.layers.dropout(outputs, drop_out, training=is_training)
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = self.layer_norm(outputs)
        return outputs

    def layer_norm(self, inputs, epsilon=1e-8):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

        params_shape = inputs.get_shape()[-1:]
        gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
        beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())

        outputs = gamma * normalized + beta
        return outputs

    def general_attention(self, query, key):
        """
        :param query: [batch_size, None, query_size]
        :param key:   [batch_size, time, key_size]
        :return:      [batch_size, None, time]
            query_size should keep the same dim with key_size
        """
        # [batch_size, None, time]
        align = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        # scale (optional)
        align = align / (key.get_shape().as_list()[-1] ** 0.5)
        return align

    def self_multi_head_attn(self, inputs, num_units, num_heads, key_masks, dropout_rate, is_training):
        """
        Args:
          inputs(query): A 3d tensor with shape of [N, T_q, C_q]
          inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
        """
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # tf.nn.relu
        Q, K, V = tf.split(Q_K_V, 3, -1)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # (h*N, T_q, T_k)
        align = self.general_attention(Q_, K_)

        # (h*N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])
        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(inputs)[1], 1])
        # (h*N, T_q, C/h)
        outputs = self.soft_max_weighted_sum(align, V_, key_masks, dropout_rate, is_training, future_binding=False)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        # output linear
        outputs = tf.layers.dense(outputs, num_units)

        # drop_out before residual and layernorm
        outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
        # Residual connection
        outputs += inputs  # (N, T_q, C)
        # Normalize
        outputs = self.layer_norm(outputs)  # (N, T_q, C)

        return outputs

    def encoder(self, inputs, seq_len):
        max_seq_len = tf.shape(inputs)[1]
        key_masks_1d = tf.sequence_mask(seq_len, max_seq_len)
        attn_outputs = inputs

        with tf.name_scope("add_pos_encoding"):
            pos_encoding = self.learned_positional_encoding(attn_outputs, self.hparams["max_seq_len"], self.hparams["num_units"])
            attn_outputs = attn_outputs + pos_encoding
            attn_outputs = tf.layers.dropout(attn_outputs, self.hparams["dropout"], training=self.hparams["is_training"])

        for layer in range(self.hparams["num_multi_head"]):
            with tf.variable_scope('self_attn_' + str(layer)):
                attn_outputs = self.self_multi_head_attn(attn_outputs, num_units=self.hparams["num_units"],
                                                         num_heads=self.hparams["num_heads"],
                                                         key_masks=key_masks_1d,
                                                         dropout_rate=self.hparams["dropout"],
                                                         is_training=self.hparams["is_training"])
            with tf.variable_scope('ffn_' + str(layer)):
                attn_outputs = self.pointwise_feedforward(attn_outputs, self.hparams["dropout"],
                                                          self.hparams["is_training"],
                                                          num_units=[4 * self.hparams["num_units"], self.hparams["num_units"]],
                                                          activation=tf.nn.relu)
        return attn_outputs
