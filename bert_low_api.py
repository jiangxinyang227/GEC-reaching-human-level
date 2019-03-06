import os
import collections
import random

import tensorflow as tf
from bert import tokenization
from bert import modeling
from bert import optimization
from seq2seq import Conv2Conv
import metrics

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

flags = tf.flags

FLAGS = flags.FLAGS

# 定义必须要传的参数
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# 定义可选的参数

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3,
                     "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_list(
    "hidden_sizes", [128],
    "support multi lstm layers, only add hidden_size into hidden_sizes list."
)

flags.DEFINE_list(
    "layers", [128], "full connection layers"
)

flags.DEFINE_float("dropout_rate", 0.5, "dropout keep rate")


# 定义seq2seq的各项参数
flags.DEFINE_integer("seq2seq_hidden_size", 1024, "conv to conv hidden size")

flags.DEFINE_integer("kernel_size", 3, "conv kernel size")

flags.DEFINE_integer("num_layers", 7, "conv layer numbers")

flags.DEFINE_integer("num_filters", 2048, "conv kernel numbers")

flags.DEFINE_float("keep_prob", 0.8, "conv 2 conv keep_prob")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, source, target):
        """
        创建一个InputExample类
        :param guid: id号
        :param source: 源句子
        :param target: 目标句子
        """
        self.guid = guid
        self.source = source
        self.target = target


class InputFeatures(object):
    """
    创建一个InputFeatures对象，用来保存处理好的数据
    """

    def __init__(self, source_ids, source_mask, source_segment_ids, target_ids, target_mask, target_segment_ids):
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_segment_ids = source_segment_ids
        self.target_ids = target_ids
        self.target_mask = target_mask
        self.target_segment_ids = target_segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_vocab_size(self):
        """Gets the vocab size"""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, source_file, target_file):
        """读取数据."""
        with open(source_file, "r") as f:
            sources = [line.strip() for line in f.readlines()]

        with open(target_file, "r") as f_t:
            targets = [line.strip() for line in f_t.readlines()]

        return [(sources[i], targets[i]) for i in range(len(sources))]


class Seq2SeqProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "lang8-valid.en"),
                            os.path.join(data_dir, "lang8-valid.gec")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "lang8-valid.en"),
                            os.path.join(data_dir, "lang8-valid.gec")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "lang8-test.en"),
                            os.path.join(data_dir, "lang8-test.gec")), "test")

    def get_vocab_size(self):
        with open(FLAGS.vocab_file, "r", encoding="utf8") as f:
            lines = [line for line in f.readlines()]

            return len(lines)

    def get_labels(self):
        pass

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            source = tokenization.convert_to_unicode(line[0])
            target = tokenization.convert_to_unicode(line[1])

            examples.append(InputExample(guid=guid, source=source, target=target))
        return examples


class DataSet(object):
    def __init__(self):
        pass

    def convert_single_example(self, ex_index, example, max_seq_length, tokenizer):
        """
        将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
        :param ex_index: 样本在examples列表中的index
        :param example: 一个InputExample样本对象，包含了source和target句子
        :param max_seq_length:  序列的最大长度
        :param tokenizer:  tokenizer对象
        :param mode:  模式，训练，验证，预测
        :return:
        """

        # 在这里采用wordpiece算法对source和target进行分词
        tokens_source = tokenizer.tokenize(example.source)
        tokens_target = tokenizer.tokenize(example.target)

        # 序列截断，在这里 -2 的原因是因为序列需要加一个句首[CLS]和句尾[SEP]标志
        if len(tokens_source) > max_seq_length - 2:
            tokens_source = tokens_source[0:(max_seq_length - 2)]

        if len(tokens_target) > max_seq_length - 2:
            tokens_target = tokens_target[0:(max_seq_length - 2)]

        # 对输入的token首尾分别加上[CLS]和[SEP]
        ntokens_source = ["[CLS]"] + tokens_source + ["[SEP]"]
        ntokens_target = ["[CLS]"] + tokens_target + ["[SEP]"]

        # 将ntokens进行index映射，转化为index的形式, 映射的vocab是传入的vocab.txt
        source_ids = tokenizer.convert_tokens_to_ids(ntokens_source)
        target_ids = tokenizer.convert_tokens_to_ids(ntokens_target)

        # 初始化句子向量
        source_segment_ids = [0] * len(ntokens_source)
        target_segment_ids = [0] * len(ntokens_target)

        # mask，真实的token用1表示，pad用0表示，只有真实的token会被attention
        source_mask = [1] * len(source_ids)
        target_mask = [1] * len(target_ids)

        # 按照最大长度补全
        while len(source_ids) < max_seq_length:
            source_ids.append(0)
            source_mask.append(0)
            source_segment_ids.append(0)

        while len(target_ids) < max_seq_length:
            target_ids.append(0)
            target_mask.append(0)
            target_segment_ids.append(0)

        assert len(source_ids) == max_seq_length
        assert len(source_mask) == max_seq_length
        assert len(source_segment_ids) == max_seq_length
        assert len(target_ids) == max_seq_length
        assert len(target_mask) == max_seq_length
        assert len(target_segment_ids) == max_seq_length

        # 打印部分样本数据信息
        if ex_index < 1:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens_source]))
            tf.logging.info("source_ids: %s" % " ".join([str(x) for x in source_ids]))
            tf.logging.info("source_mask: %s" % " ".join([str(x) for x in source_mask]))
            tf.logging.info("source_segment_ids: %s" % " ".join([str(x) for x in source_segment_ids]))
            tf.logging.info("target_ids: %s" % " ".join([str(x) for x in target_ids]))
            tf.logging.info("target_mask: %s" % " ".join([str(x) for x in target_mask]))
            tf.logging.info("target_segment_ids: %s" % " ".join([str(x) for x in target_segment_ids]))

        # 实例化成一个InputFeatures对象
        feature = InputFeatures(
            source_ids=source_ids,
            source_mask=source_mask,
            source_segment_ids=source_segment_ids,
            target_ids=target_ids,
            target_mask=target_mask,
            target_segment_ids=target_segment_ids
        )

        return feature

    def get_data(self, examples, max_seq_length, tokenizer):
        # 遍历训练数据
        data = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            # 将InputExample对象的数据转换成InputFeature对象
            feature = self.convert_single_example(ex_index, example, max_seq_length, tokenizer)

            data.append(feature)

        return data

    def next_batch(self, examples, max_seq_length, tokenizer, batch_size, is_training=True):
        data = self.get_data(examples, max_seq_length, tokenizer)

        if is_training:
            random.shuffle(data)
        batch_num = len(data) // batch_size

        for i in range(batch_num):
            batch = data[batch_size * i: batch_size * (i + 1)]
            source_ids = [feature.source_ids for feature in batch]
            source_mask = [feature.source_mask for feature in batch]
            source_segment_ids = [feature.source_segment_ids for feature in batch]
            target_ids = [feature.target_ids for feature in batch]
            target_mask = [feature.target_mask for feature in batch]
            target_segment_ids = [feature.target_segment_ids for feature in batch]

            yield dict(source_ids=source_ids, source_mask=source_mask, source_segment_ids=source_segment_ids,
                       target_ids=target_ids, target_mask=target_mask, target_segment_ids=target_segment_ids)


class Seq2SeqBert(object):
    def __init__(self, init_checkpoint, bert_config, vocab_size, num_train_steps=None, num_warmup_steps=None,
                 use_one_hot_embeddings=False, is_training=False):

        self.init_checkpoint = init_checkpoint
        self.bert_config = bert_config
        self.vocab_size = vocab_size
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.is_training = is_training

        self.batch_size = tf.placeholder(tf.int32, None, name="batch_size")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        self.target_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="target_ids")
        self.target_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="target_mask")
        self.target_segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="target_segment_ids")

        self.source_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="source_ids")
        self.source_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="source_mask")
        self.source_segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="source_segment_ids")

        # 构建网络
        self.build_model()

    def create_model(self):
        # 初始化bert模型, 源句子输入到bert模型中
        source_model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.source_ids,
            input_mask=self.source_mask,
            token_type_ids=self.source_segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings
        )

        # 初始化bert模型，目标句子输入到bert模型中
        target_model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.target_ids,
            input_mask=self.target_mask,
            token_type_ids=self.target_segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings
        )

        # 根据source_mask来计算出每条序列的长度，因为input_mask中真实token是1，补全的pad是0
        source_used = tf.sign(tf.abs(self.source_mask))
        source_seq_len = tf.reduce_sum(source_used, 1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        # 根据target_mask来计算出每条序列的长度，因为input_mask中真实token是1，补全的pad是0
        target_used = tf.sign(tf.abs(self.target_mask))
        target_seq_len = tf.reduce_sum(target_used, 1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        # 获得bert模型最后的输出，维度为[batch_size, seq_length, embedding_size]
        # 将bert的输出作为我们的输入，相当于做word embedding，获得source和target的word embedding
        source_embedding = source_model.get_sequence_output()
        target_embedding = target_model.get_sequence_output()

        # stop_grad_source_embedding = tf.stop_gradient(source_embedding)
        # stop_grad_target_embedding = tf.stop_gradient(target_embedding)

        tf.logging.info("bert embedding size: {}".format(source_embedding.get_shape()))

        conv2conv = Conv2Conv(source_embedding=source_embedding, target_ids=self.target_ids,
                              target_embedding=target_embedding,
                              source_seq_len=source_seq_len, target_seq_len=target_seq_len,
                              hidden_size=FLAGS.seq2seq_hidden_size, batch_size=self.batch_size,
                              vocab_size=self.vocab_size,
                              num_layers=FLAGS.num_layers, kernel_size=FLAGS.kernel_size, num_filters=FLAGS.num_filters,
                              keep_prob=self.keep_prob, is_training=self.is_training)

        result = conv2conv.build_network()
        return result

    def build_model(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        # 加载bert模型, 初始化变量名，assignment_map和initialized_variable_names都是有序的字典，
        # assignment_map取出了tvars中所有的变量名，并且键和值都是变量名
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)

            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        (loss, logits, pred_y) = self.create_model()

        self.loss = loss
        self.pred_y = pred_y
        print(loss)
        print(FLAGS.learning_rate)
        print(self.num_train_steps)
        print(self.num_warmup_steps)
        self.train_op = optimization.create_optimizer(
            loss, FLAGS.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_prob):
        # 对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据

        feed_dict = {self.source_ids: batch["source_ids"],
                     self.source_mask: batch["source_mask"],
                     self.source_segment_ids: batch["source_segment_ids"],
                     self.target_ids: batch["target_ids"],
                     self.target_mask: batch["target_mask"],
                     self.target_segment_ids: batch["target_segment_ids"],
                     self.keep_prob: dropout_prob,
                     self.batch_size: len(batch["source_ids"])}

        # 训练模型
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.source_ids: batch["source_ids"],
                     self.source_mask: batch["source_mask"],
                     self.source_segment_ids: batch["source_segment_ids"],
                     self.target_ids: batch["target_ids"],
                     self.target_mask: batch["target_mask"],
                     self.target_segment_ids: batch["target_segment_ids"],
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch["source_ids"])}

        loss = sess.run(self.loss, feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.source_ids: batch["source_ids"],
                     self.source_mask: batch["source_mask"],
                     self.source_segment_ids: batch["source_segment_ids"],
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch["source_ids"])}
        predict = sess.run([self.pred_y], feed_dict=feed_dict)

        return predict


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # 定义数据处理的类
    processors = {"seq2seq": Seq2SeqProcessor}

    # 检查checkPoint名称是否和do_lower_case匹配，因为有的bert case是保留大写的
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    # 解析bert的配置参数
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # max_position_embeddings=512，因此序列长度最大512
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    # 创建一个目录
    tf.gfile.MakeDirs(FLAGS.output_dir)
    # 任务数据预处理对象的名称
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # 实例化数据处理类对象
    processor = processors[task_name]()

    # 创建一个端到端的tokenizer对象
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # 创建一个生成模型数据的对象
    data_set = DataSet()

    with tf.Session() as sess:

        if FLAGS.do_train:
            # 获得train data 的InputExample对象列表
            train_examples = processor.get_train_examples(FLAGS.data_dir)
            # 训练时迭代的总步数
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            # 训练时的预热步数，主要是用于学习速率的衰减选择
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

            # 获得eval data 的InputExample对象列表
            eval_examples = processor.get_dev_examples(FLAGS.data_dir)

            with tf.name_scope("train"):
                with tf.variable_scope("seq2seq", reuse=None):
                    train_model = Seq2SeqBert(init_checkpoint=FLAGS.init_checkpoint,
                                              bert_config=bert_config,
                                              vocab_size=processor.get_vocab_size(),
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              use_one_hot_embeddings=FLAGS.use_tpu,
                                              is_training=True)

            with tf.name_scope("eval"):
                with tf.variable_scope("seq2seq", reuse=True):
                    eval_model = Seq2SeqBert(init_checkpoint=FLAGS.init_checkpoint,
                                             bert_config=bert_config,
                                             vocab_size=processor.get_vocab_size(),
                                             )

            sess.run(tf.global_variables_initializer())

            current_step = 0
            for epoch in range(FLAGS.num_train_epochs):
                print("----- Epoch {}/{} -----".format(epoch + 1, FLAGS.num_train_epochs))

                for batch in data_set.next_batch(train_examples, FLAGS.max_seq_length,
                                                 tokenizer, FLAGS.train_batch_size):
                    loss = train_model.train(sess, batch, FLAGS.keep_prob)
                    current_step += 1
                    print("train: step: {}, loss: {}".format(current_step, loss))
                    if current_step % FLAGS.steps_per_checkpoint == 0 and FLAGS.do_eval:

                        eval_losses = []
                        for eval_batch in data_set.next_batch(eval_examples, FLAGS.max_seq_length,
                                                              tokenizer, FLAGS.eval_batch_size):
                            eval_loss = eval_model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)

                        print("\n")
                        print("eval: step: {}, loss: {}".format(current_step, sum(eval_losses) / len(eval_losses)))
                        print("\n")
                        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                        train_model.saver.save(sess, checkpoint_path, global_step=current_step)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
