import math
import os
import tensorflow as tf

from conv2conv import Conv2Conv
from data_helper import DataSet
from metrics import f_beta


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("hidden_size", 1024, "Number of hidden units in each layer")
flags.DEFINE_integer("num_layers", 7, "Number of layers in each encoder and decoder")
flags.DEFINE_integer("embedding_size", 500, "Embedding dimensions of encoder and decoder inputs")
flags.DEFINE_integer("kernel_size", 3, "kernel size of conv")
flags.DEFINE_integer("num_filters", 2048, "filter numbers of conv")
flags.DEFINE_float("learning_rate", 2e-5, "Learning rate")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("keep_prob", 0.2, "keep dropout prob")
flags.DEFINE_integer("epochs", 30, "Maximum # of training epochs")

flags.DEFINE_integer("steps_per_checkpoint", 100, "Save model checkpoint every this iteration")
flags.DEFINE_string("model_dir", "model/", "Path to save model checkpoints")
flags.DEFINE_string("model_name", "gec", "File name used for model checkpoints")

flags.DEFINE_string("source_file", "data/token_lang-8/lang8-train.en", "source train file path")
flags.DEFINE_string("target_file", "data/token_lang-8/lang8-train.gec", "target train file path")
flags.DEFINE_string("source_valid", "data/token_lang-8/lang8-valid.en", "source valid file path")
flags.DEFINE_string("target_valid", "data/token_lang-8/lang8-valid.gec", "target valid file path")

dataSet = DataSet(FLAGS.embedding_size, FLAGS.source_file, FLAGS.target_file, FLAGS.source_valid,
                  FLAGS.target_valid, FLAGS.batch_size, is_first=False)

# 生成训练数据和测试数据
dataSet.gen_train_valid()
vocab_size = len(dataSet.idx_to_word)

print(vocab_size)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)

with tf.device("/device:GPU:0"):
    with tf.Session(config=config) as sess:
        model = Conv2Conv(FLAGS.embedding_size, FLAGS.hidden_size, vocab_size, FLAGS.num_layers,
                          FLAGS.kernel_size, FLAGS.num_filters, is_training=True)

        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        current_step = 0
        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        for epoch in range(FLAGS.epochs):
            print("----- Epoch {}/{} -----".format(epoch + 1, FLAGS.epochs))

            for batch in dataSet.next_batch(dataSet.train_data):
                loss, predictions = model.train(sess, batch, FLAGS.keep_prob)
                f = f_beta(predictions.tolist(), batch["targets"], batch["target_length"])
                current_step += 1
                print("train: step: {}, loss: {}, f_0.5: {}".format(current_step, loss, f))
                if current_step % FLAGS.steps_per_checkpoint == 0:

                    valid_losses = []
                    valid_fs = []
                    for valid_batch in dataSet.next_batch(dataSet.valid_data):
                        valid_loss, valid_predictions = model.valid(sess, valid_batch, 1.0)
                        valid_f = f_beta(valid_predictions.tolist(), valid_batch["targets"], valid_batch["target_length"])

                        valid_losses.append(valid_loss)
                        valid_fs.append(valid_f)
                    print("\n")
                    print("valid: step: {}, loss: {}, perplexity: {}".format(current_step,
                                                                             sum(valid_losses) / len(valid_losses),
                                                                             sum(valid_fs) / len(valid_fs)))
                    print("\n")
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    saver.save(sess, checkpoint_path, global_step=current_step)