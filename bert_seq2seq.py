import os
import pickle
import collections
import codecs

import tensorflow as tf
from bert import tokenization
from bert import modeling
from bert import optimization
from seq2seq import Conv2Conv
import metrics

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

flags.DEFINE_float("num_train_epochs", 3.0,
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
            self._read_data(os.path.join(data_dir, "lang8-train.en"),
                            os.path.join(data_dir, "lang8-train.gec")), "train"
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


def convert_single_example(ex_index, example, max_seq_length, tokenizer, mode):
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


def filed_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file, mode=None
):
    """
    将所有的数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  InputExample格式的数据列表, 包含source和target句子
    :param max_seq_length: 预先设定的最大序列长度，不要超过512
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    # 定义写入tfrecord的对象
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 将InputExample对象的数据转换成InputFeature对象
        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            # tf.train.Feature是一个protobuf消息，共有三个Feature类型：int，float，bytes
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # 创建一个有序的词典，用词典来保存InputFeature中的数据
        features = collections.OrderedDict()
        features["source_ids"] = create_int_feature(feature.source_ids)
        features["source_mask"] = create_int_feature(feature.source_mask)
        features["source_segment_ids"] = create_int_feature(feature.source_segment_ids)
        features["target_ids"] = create_int_feature(feature.target_ids)
        features["target_mask"] = create_int_feature(feature.target_mask)
        features["target_segment_ids"] = create_int_feature(feature.target_segment_ids)

        # tf.train.Example是一个protobuf消息
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # 保存成tfrecord文件
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """
    返回一个input_fn引用
    :param input_file:  tfrecord文件的路径
    :param seq_length:  转换后的序列长度, 也就是最大长度
    :param is_training:  是否是训练模式
    :param drop_remainder:  布尔值，若为True，则最后的batch若小于batch_size,则被去掉
    :return:
    """
    # tf.FixedLenFeature返回一个定长的tensor，并且会将稀疏转为为稠密对待
    name_to_features = {
        "source_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "source_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "source_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "target_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "target_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "target_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """
        解码一个record到tensorflow中的example
        :param record:  一条tfrecord文件中的record数据
        :param name_to_features:  输入的features
        :return:
        """
        # 解析生成一个feature key 到对应的tensor的字典
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            # tf.example对象中的数据类型只支持int64，但是在tpu中只支持int32，因此在这里做数据转换
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """
        input function方法，会以batch的形式返回数据
        :param params:  tf.contrib.tup.TPUEatimator 会将train，eval，
                        predict对应的batch_size放置在params["batch_size"]中
        :return:
        """
        batch_size = params["batch_size"]
        # 创建tfrecord dataset对象去读取tfrecord文件
        d = tf.data.TFRecordDataset(input_file)
        # 训练的时候需要对数据进行重复使用和shuffle
        if is_training:
            # repeat方法可以让训练集被重复使用，不传值支持无限重复
            d = d.repeat()
            # 打乱训练集，返回一个子数据集，大小为100
            d = d.shuffle(buffer_size=100)

        # apply支持应用一个函数到数据集上，该函数就是tf.contrib.data.map_and_batch
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def create_model(bert_config, is_training, source_ids, source_mask, source_segment_ids, source_seq_len, vocab_size,
                 target_ids, target_mask, target_segment_ids ,target_seq_len, use_one_hot_embeddings):
    """
    创建模型
    :param bert_config:  bert 模型的配置参数
    :param is_training:  判断是否是训练模式
    :param input_ids:  输入的数据的index表示
    :param input_mask:  mask列表
    :param segment_ids:  句子的index
    :param label_ids:  标签序列
    :param num_labels:  标签的数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 初始化bert模型, 源句子输入到bert模型中
    source_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=source_ids,
        input_mask=source_mask,
        token_type_ids=source_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # 初始化bert模型，目标句子输入到bert模型中
    target_model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=target_ids,
        input_mask=target_mask,
        token_type_ids=target_segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # 获得bert模型最后的输出，维度为[batch_size, seq_length, embedding_size]
    # 将bert的输出作为我们的输入，相当于做word embedding，获得source和target的word embedding
    source_embedding = source_model.get_sequence_output()
    target_embedding = target_model.get_sequence_output()

    stop_grad_source_embedding = tf.stop_gradient(source_embedding)
    stop_grad_target_embedding = tf.stop_gradient(target_embedding)

    batch_size = source_embedding.shape[0].value

    tf.logging.info("bert embedding size: {}".format(source_embedding.get_shape()))

    conv2conv = Conv2Conv(source_embedding=stop_grad_source_embedding, target_ids=target_ids,
                          target_embedding=stop_grad_target_embedding,
                          source_seq_len=source_seq_len, target_seq_len=target_seq_len,
                          hidden_size=FLAGS.seq2seq_hidden_size, batch_size=batch_size, vocab_size=vocab_size,
                          num_layers=FLAGS.num_layers, kernel_size=FLAGS.kernel_size, num_filters=FLAGS.num_filters,
                          keep_prob=FLAGS.keep_prob, is_training=is_training)

    result = conv2conv.build_network()
    return result


def model_fn_builder(bert_config, init_checkpoint, learning_rate, vocab_size, num_train_steps,
                     num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """
    构建bert模型
    :param bert_config:  bert模型的配置参数
    :param init_checkpoint:  # 初始化的check_point
    :param learning_rate:  # 初始化的学习速率
    :param vocab_size: # 词汇表空间大小
    :param num_train_steps:  # 训练的步数
    :param num_warmup_steps:  # 训练时预热的步数
    :param use_tpu:
    :param use_one_hot_embeddings:  # 是否使用one_hot_embedding
    :return:  返回一个model_fn的引用
    """
    def model_fn(features, mode, params):
        """
        内部函数
        :param features:  数据的features，一个字典，接收从input_fn中返回的features
        :param mode:
        :param params:
        :return:
        """
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        source_ids = features["source_ids"]
        source_mask = features["source_mask"]
        source_segment_ids = features["source_segment_ids"]
        target_ids = features["target_ids"]
        target_mask = features["target_mask"]
        target_segment_ids = features["target_segment_ids"]

        # 根据source_mask来计算出每条序列的长度，因为input_mask中真实token是1，补全的pad是0
        source_used = tf.sign(tf.abs(source_mask))
        source_seq_len = tf.reduce_sum(source_used, 1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        # 根据target_mask来计算出每条序列的长度，因为input_mask中真实token是1，补全的pad是0
        target_used = tf.sign(tf.abs(target_mask))
        target_seq_len = tf.reduce_sum(target_used, 1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

        # 如果是train，则is_training=True
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        (loss, logits, pred_y) = create_model(
            bert_config, is_training, source_ids, source_mask, source_segment_ids, source_seq_len, vocab_size,
            target_ids, target_mask, target_segment_ids, target_seq_len, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None

        # 加载bert模型, 初始化变量名，assignment_map和initialized_variable_names都是有序的字典，
        # assignment_map取出了tvars中所有的变量名，并且键和值都是变量名
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    # 按照assignment_map中的变量名从init_checkpoint中加载出初始化变量值
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 打印模型的参数
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 创建一个优化训练的op入口
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            # 将训练时的变量初始化参数，损失和优化器封装起来
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)  # scaffold_fn这里用来将BERT中的参数作为我们模型的初始值

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(logits, trans_params):
                # 获得验证集上的性能指标

                return {
                    "eval_pred": (pred_y, pred_y),
                    "eval_target": (target_ids, target_ids),
                    "eval_seq_len": (target_seq_len, target_seq_len)
                }

            # 这里eval_metrics必须是一个元祖
            eval_metrics = (metric_fn, [logits])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:
            # 预测时只返回预测的结果
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_y,
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


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
    # 任务名称
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # 实例化数据处理类对象
    processor = processors[task_name]()

    # 创建一个端到端的tokenizer对象
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # 主要是控制训练时的batch_size的，分布式计算时会用到，
    # 下面的模式是global_batch_size // cores的数量 = 新的batch_size，新的batch_size被用于每个分区
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    # 构建tpu config，在这里指定了model的存储位置，checkpoint保存的间隔步数
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        # 获得train data 的InputExample对象列表
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        # 训练时迭代的总步数
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # 训练时的预热步数，主要是用于学习速率的衰减选择
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # 自定义Estimator类时，需要定义一个model_fn的方法，该方法实现了模型的结构，并定义了训练，验证和测试等方法。
    # 在创建模型时加载了BERT模型的参数进行了自己模型的参数初始化过程。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        vocab_size=processor.get_vocab_size(),
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # 创建一个estimator对象，在这里会定义trian，eval，predict的batch_size,
    # 在调用model_fn和input_fn会将相应的模式下的batch_size放入到params["batch_size"]中
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:

        # 定义train data数据转化为tfrecord数据的存储路径
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # 将train data数据转化为tfrecord数据
        filed_based_convert_examples_to_features(
            train_examples, FLAGS.max_seq_length, tokenizer, train_file)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        # 返回一个input_fn引用，
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        # estimator对象有train，evaluate，predict三个方法，调用train训练模型
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        # 获得eval data 的InputExample 对象列表
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        # eval data 的tfrecord文件保存路径
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        # 将eval data保存成tfrecord数据文件
        filed_based_convert_examples_to_features(
            eval_examples, FLAGS.max_seq_length, tokenizer, eval_file)

        # 打印验证集数据信息
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False

        # 创建eval data 数据的input_fn引用
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        # 调用evaluate方法进行验证
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        f_beta = metrics.f_beta(result["eval_pred"], result["eval_target"], result["eval_seq_len"])
        tf.logging.info("eval_f0.5: {}".format(f_beta))

        # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        # with tf.gfile.GFile(output_eval_file, "w") as writer:
        #     tf.logging.info("***** Eval results *****")
        #     for key in sorted(result.keys()):
        #         tf.logging.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        # 在进行测试集token化时对解析后的token保存起来了，因此下面需要判断该文件是否存在
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        # 读取出label-index映射表
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        # 将test data 转换成InputExample对象列表
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        # test data 的tfrecord文件保存路径
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        # 将test data保存成tfrecord数据文件
        filed_based_convert_examples_to_features(predict_examples,
                                                 FLAGS.max_seq_length,
                                                 tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        # 预测时不让使用TPU
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False

        # 创建test data 数据的input_fn引用
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        # 调用predict方法对测试集进行预测
        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.csv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
