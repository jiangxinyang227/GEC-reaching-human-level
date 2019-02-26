import tensorflow as tf
import numpy as np


class Conv2Conv(object):
    """
    定义卷积到卷积的seq2seq网络结构
    """
    def __init__(self, batch_size, embedding_size, hidden_size, vocab_size, num_layers, dropout_rate,
                 kernel_size, num_filters):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.num_filters = num_filters

    def padding_and_softmax(self, logits, query_len, key_len):
        """
        对attention权重归一化处理
        :param logits: 未归一化的attention权重 [batch_size, de_seq_len, en_seq_len]
        :param query_len: decoder的输入的真实长度 [batch_size]
        :param key_len: encoder的输入的真实长度 [batch_size]
        :return:
        """
        with tf.name_scope("padding_aware_softmax"):
            # Lengths to which batches are padded to
            TQ = tf.shape(logits)[1]
            TK = tf.shape(logits)[2]

            # Derive masks
            query_mask = tf.sequence_mask(lengths=query_len, maxlen=TQ, dtype=tf.int32)  # [B, TQ]
            key_mask = tf.sequence_mask(lengths=key_len, maxlen=TK, dtype=tf.int32)  # [B, TK]

            # Introduce new dimensions (we want to have a batch-wise outer product)
            query_mask = tf.expand_dims(query_mask, axis=2)  # [B, TQ, 1]
            key_mask = tf.expand_dims(key_mask, axis=1)  # [B, 1, TK]

            # Combine masks
            joint_mask = tf.cast(tf.matmul(query_mask, key_mask), tf.float32, name="joint_mask")  # [B, TQ, TK]

            # Padding should not influence maximum (replace with minimum)
            logits_min = tf.reduce_min(logits, axis=2, keepdims=True, name="logits_min")  # [B, TQ, 1]
            logits_min = tf.tile(logits_min, multiples=[1, 1, TK])  # [B, TQ, TK]
            logits = tf.where(condition=joint_mask > .5,
                              x=logits,
                              y=logits_min)

            # Determine maximum
            logits_max = tf.reduce_max(logits, axis=2, keepdims=True, name="logits_max")  # [B, TQ, 1]
            logits_shifted = tf.subtract(logits, logits_max, name="logits_shifted")  # [B, TQ, TK]

            # Derive unscaled weights
            weights_unscaled = tf.exp(logits_shifted, name="weights_unscaled")

            # Apply mask
            weights_unscaled = tf.multiply(joint_mask, weights_unscaled, name="weights_unscaled_masked")  # [B, TQ, TK]

            # Derive total mass
            weights_total_mass = tf.reduce_sum(weights_unscaled, axis=2,
                                               keepdims=True, name="weights_total_mass")  # [B, TQ, 1]

            # Avoid division by zero
            weights_total_mass = tf.where(condition=tf.equal(query_mask, 1),
                                          x=weights_total_mass,
                                          y=tf.ones_like(weights_total_mass))

            # Normalize weights
            weights = tf.divide(weights_unscaled, weights_total_mass, name="normalize_attention_weights")  # [B, TQ, TK]

            return weights

    def attention(self, query, encoder_input, key, value, query_len, key_len):
        """
        计算encoder decoder之间的attention
        :param query: decoder 的输入 [batch_size, de_seq_len, embedding_size]
        :param encoder_input:  encoder的原始输入 [batch_size, en_seq_len, embedding_size]
        :param key: encoder的输出 [batch_size, en_seq_len, embedding_size]
        :param value: encoder的输出 [batch_size, en_seq_len, embedding_size]
        :param query_len: decoder的输入的真实长度 [batch_size]
        :param key_len: encoder的输入的真实长度 [batch_size]
        :return:
        """
        with tf.name_scope("attention"):
            # 通过点积的方法计算权重, 得到[batch_size, de_seq_len, en_seq_len]
            attention_scores = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))

            # 对权重进行归一化处理
            attention_scores = self.padding_and_softmax(logits=attention_scores,
                                                        query_len=query_len,
                                                        key_len=key_len)
            # 对source output进行加权平均 [batch_size, de_seq_len, embedding_size]
            weighted_output = tf.matmul(attention_scores, tf.add(value, encoder_input))

            return weighted_output

    def glu(self, x):
        """
        glu门函数, 将后半段计算门系数，前半段作为输入值，element-wise的乘积
        :param x: 卷积操作后的Tensor [batch_size, seq_len, hidden_size * 2]
        :return: [batch_size, seq_len, hidden_size]
        """
        a, b = tf.split(x, num_or_size_splits=2, axis=2)
        return tf.multiply(tf.nn.sigmoid(b), a)

    def encoder_layer(self, inputs, input_len, kernel_size, dropout_rate, is_training):
        """
        单层encoder层的实现
        :param inputs: encoder的输入 [batch_size, seq_len, hidden_size]
        :param input_len: encoder输入的实际长度
        :param kernel_size: 卷积核大小
        :param dropout_rate:
        :param is_training:
        :return:
        """

        seq_len = tf.shape(inputs)[1]

        # 计算序列在卷积时要补的长度
        num_pad = kernel_size - 1

        # 在序列的左右各补长一半
        x_pad = tf.pad(inputs, paddings=[[0, 0], [num_pad / 2, num_pad / 2], [0, 0]])

        # 一维卷积操作，针对3维的输入的卷积 [batch_size, seq_len, 2 * hidden_size]
        x_conv = tf.nn.conv1d(x_pad, [kernel_size, self.hidden_size, 2 * self.hidden_size], stride=1, padding="VALID")
        # 一维卷积操作，针对3维的输入的卷积，[B, T, 2*E]
        # x_conv = tf.layers.Conv1D(filters=2 * self.hidden_size,
        #                           kernel_size=kernel_size,
        #                           strides=1,
        #                           padding="valid",
        #                           activation=None,
        #                           use_bias=True)(x_pad)

        # glu 门控函数 [batch_size, seq_len, hidden_size]
        x_glu = self.glu(x_conv)

        # mask padding [batch_size, seq_len]
        mask = tf.sequence_mask(lengths=input_len, maxlen=seq_len, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)  # [batch_size, seq_len, 1]

        # element-wise的乘积, [batch_size, seq_len, hidden_size]
        x_mask = tf.multiply(mask, x_glu)

        # drouput 正则化 [batch_size, seq_len, hidden_size]
        x_drop = tf.nn.dropout(x_mask, keep_prob=dropout_rate, noise_shape=[self.batch_size, 1, self.hidden_size])

        # 残差连接 [batch_size, seq_len, hidden_size]
        x_drop = x_drop + inputs

        # bn处理，只对最后一个维度做bn处理 [batch_size, seq_len, hidden_size]
        x_bn = tf.layers.BatchNormalization(axis=2)(x_drop, training=is_training)

        return x_bn

    def encoder(self, inputs, input_len, is_training):
        """
        多层encoder层
        :param inputs: 原始输入（word_embedding + position_embedding） [batch_size, seq_len, embedding_size]
        :param input_len: padding补全前的真实长度 [batch_size]
        :param is_training:
        :return:
        """
        with tf.name_scope("start_linear_map"):
            w_start = tf.get_variable("w_start", shape=[self.embedding_size, self.hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_start = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="b_start")

            inputs_start = tf.matmul(inputs, w_start) + b_start  # [batch_size, seq_len, hidden_size]

        with tf.name_scope("encoder"):
            # 维度不变
            for layer_id in range(self.num_layers):
                inputs_start = self.encoder_layer(inputs=inputs_start,
                                                  input_len=input_len,
                                                  kernel_size=self.kernel_size,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=is_training)

        with tf.name_scope("final_linear_map"):
            w_final = tf.get_variable("w_final", shape=[self.hidden_size, self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_final = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name="b_final")
            inputs_final = tf.matmul(inputs_start, w_final) + b_final  # [batch_size, seq_len, embedding_size]

        return inputs_final

    def decoder_layer(self, raw_inputs, new_inputs, input_len, encoder_input, encoder_output,
                      encoder_length, kernel_size, dropout_rate, is_training):

        seq_len = tf.shape(new_inputs)[1]

        num_pad = kernel_size - 1

        # 在序列的左边进行补长，
        x_pad = tf.pad(new_inputs, paddings=[[0, 0], [num_pad, 0], [0, 0]])

        # 一维卷积操作，针对3维的输入的卷积，[batch_size, seq_len, 2 * hidden_size]
        x_conv = tf.nn.conv1d(x_pad, [kernel_size, self.hidden_size, 2 * self.hidden_size],
                              stride=1, padding="VALID")

        # x_conv = tf.layers.Conv1D(filters=2 * E,
        #                           kernel_size=kernel_size,
        #                           strides=1,
        #                           padding="valid",
        #                           activation=None,
        #                           use_bias=True)(x_pad)

        # glu 门控函数 [batch_size, seq_len, hidden_size]
        x_glu = self.glu(x_conv)

        with tf.name_scope("middle_linear_map"):
            w_middle = tf.get_variable("w_middle", shape=[self.hidden_size, self.embedding_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_middle = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name="b_middle")

            # [batch_size, seq_len, embedding_size]
            x_middle = tf.add(tf.matmul(x_glu, w_middle) + b_middle, raw_inputs)

        # attention [batch_size, seq_len, embedding_size]
        x_atten = self.attention(query=x_middle,
                                 encoder_input=encoder_input,
                                 key=encoder_output,
                                 value=encoder_output,
                                 query_len=input_len,
                                 key_len=encoder_length)

        with tf.name_scope("middle_linear_map_1"):
            w_middle_1 = tf.get_variable("w_middle_1",
                                         shape=[self.embedding_size, self.hidden_size],
                                         initializer=tf.contrib.layers.xavier_initializer())
            b_middle_1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="b_middle_1")

            # [batch_size, seq_len, hidden_size]
            x_middle_1 = tf.matmul(x_atten, w_middle_1) + b_middle_1

            # [batch_size, seq_len, hidden_size]
            x_final = x_glu + x_middle_1 + new_inputs

        # mask padding [batch_size, seq_len]
        mask = tf.sequence_mask(lengths=input_len, maxlen=seq_len, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)  # [batch_size, seq_len, 1]

        # element-wise的乘积, [batch_size, seq_len, hidden_size]
        x_mask = tf.multiply(mask, x_final)

        # drouput 正则化
        x_drop = tf.nn.dropout(x_mask, keep_prob=dropout_rate,
                               noise_shape=[self.batch_size, 1, self.hidden_size])

        # 残差连接
        x_drop = x_drop + new_inputs

        x_bn = tf.layers.BatchNormalization(axis=2)(x_drop, training=is_training)

        return x_bn

    def decoder(self, inputs, input_len, encoder_input, encoder_output, encoder_length,
                vocab_size, is_training):

        with tf.name_scope("start_linear_map"):
            w_start = tf.get_variable("w_start", shape=[self.embedding_size, self.hidden_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_start = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name="b_start")

            inputs_start = tf.matmul(inputs, w_start) + b_start  # [batch_size, seq_len, hidden_size]

        with tf.name_scope("decoder"):
            for layer_id in range(self.num_layers):
                inputs_start = self.decoder_layer(raw_inputs=inputs,
                                                  new_inputs=inputs_start,
                                                  input_len=input_len,
                                                  encoder_input=encoder_input,
                                                  encoder_output=encoder_output,
                                                  encoder_length=encoder_length,
                                                  kernel_size=self.kernel_size,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=is_training)

        with tf.name_scope("final_linear_map"):
            w_final = tf.get_variable("w_final", shape=[self.hidden_size, self.embedding_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_final = tf.Variable(tf.constant(0.1, shape=[self.embedding_size]), name="b_final")

            # [batch_size, seq_len, embedding_size]
            inputs_final = tf.matmul(inputs_start, w_final) + b_final

        with tf.name_scope("output"):
            w_output = tf.get_variable("w_output", shape=[self.embedding_size, self.vocab_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_output = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]), name="b_output")

            output = tf.matmul(inputs_final, w_output) + b_output

        return output