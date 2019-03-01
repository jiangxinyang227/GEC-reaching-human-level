import logging
from collections import Counter
import pickle
import random

import numpy as np
import gensim


class DataSet(object):
    def __init__(self, embedding_size, source_file, target_file, source_valid,
                 target_valid, batch_size, is_first=False):
        self.embedding_size = embedding_size
        self.source_file = source_file
        self.target_file = target_file
        self.source_valid = source_valid
        self.target_valid = target_valid
        self.batch_size = batch_size

        self.is_first = is_first

        self.train_data = []
        self.valid_data = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_embedding = None

    def _read_data(self, file_path):
        with open(file_path, "r") as f:
            sentences = [line.strip().split() for line in f.readlines()]

        return sentences

    def _get_embedding(self, sentences):
        """
        训练词向量
        :param sentences:
        :return:
        """
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        model = gensim.models.Word2Vec(sentences, size=self.embedding_size, sg=0, min_count=0, iter=20)
        model.wv.save_word2vec_format("word2vec/word2Vec" + ".bin", binary=True)

    def _trans_index(self, sentences, word_to_idx):
        """
        将数据转换成索引表示
        :param sentences:
        :param word_to_idx: 词到索引的映射表
        :return:
        """

        data_idx = [[word_to_idx.get(word, word_to_idx["<unk>"]) for word in sentence]for sentence in sentences]

        return data_idx

    def _gen_vocabulary(self, sentences):
        """
        生成vocab
        :param sentences:
        :return:
        """

        all_words = [word for sentence in sentences for word in sentence]

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count if item[1] >= 0]

        vocab, word_embedding = self._get_word_embedding(words)
        self.word_embedding = word_embedding

        self.word_to_idx = dict(zip(vocab, list(range(len(vocab)))))
        self.idx_to_word = dict(zip(list(range(len(vocab))), vocab))

        vocab_dict = {"word_embedding": word_embedding, "word2idx": self.word_to_idx, "idx2word": self.idx_to_word}
        # 将vocab，word_to_idx，idx_to_word保存为pkl文件
        with open("data/vocab/vocab_dict_500dim.pkl", "wb") as f:
            pickle.dump(vocab_dict, f)

    def _get_word_embedding(self, words):
        """
        按照预训练好的词向量，去提取vocab
        :param words:
        :return:
        """
        word_vec = gensim.models.KeyedVectors.load_word2vec_format("word2vec/word2Vec.bin", binary=True)
        vocab = []
        word_embedding = []

        # 添加"<pad>", "<unk>", "<go>", "<eos>",
        vocab.append("<pad>")
        vocab.append("<unk>")
        vocab.append("<go>")
        vocab.append("<eos>")
        word_embedding.append(np.zeros(self.embedding_size))
        word_embedding.append(np.random.randn(self.embedding_size))
        word_embedding.append(np.random.randn(self.embedding_size))
        word_embedding.append(np.random.randn(self.embedding_size))

        count = 0
        for word in words:
            try:
                vector = word_vec.wv[word]
                vocab.append(word)
                word_embedding.append(vector)
            except:
                count += 1

        print("总共有{}个词不在词典中".format(count))

        return vocab, np.array(word_embedding)

    def gen_train_valid(self):
        """
        生成训练，验证数据集
        :return:
        """
        # 将数据读取出来
        source_data = self._read_data(self.source_file)
        target_date = self._read_data(self.target_file)

        source_valid = self._read_data(self.source_valid)
        target_valid = self._read_data(self.target_valid)

        # 训练词向量，并得到vocab_dict
        if self.is_first:
            self._get_embedding(target_date + source_data)
            self._gen_vocabulary(target_date + source_data)

        else:
            with open("data/vocab/vocab_dict_500dim.pkl", "rb") as f:
                vocab_dict = pickle.load(f)

                self.word_embedding = vocab_dict["word_embedding"]
                self.word_to_idx = vocab_dict["word2idx"]
                self.idx_to_word = vocab_dict["idx2word"]

        source_data_idx = self._trans_index(source_data, self.word_to_idx)
        target_date_idx = self._trans_index(target_date, self.word_to_idx)

        train_data = [[source_data_idx[i], target_date_idx[i]] for i in range(len(source_data_idx))]

        source_valid_idx = self._trans_index(source_valid, self.word_to_idx)
        target_valid_idx = self._trans_index(target_valid, self.word_to_idx)

        valid_data = [[source_valid_idx[i], target_valid_idx[i]] for i in range(len(source_valid_idx))]

        self.train_data = train_data
        self.valid_data = valid_data

    def _process_data(self, batch: list, is_train=True) -> dict:
        """
        对每个batch进行按最大长度补全处理
        :param batch:
        :return:
        """
        pad_token = self.word_to_idx["<pad>"]
        eos_token = self.word_to_idx["<eos>"]

        source_length = [len(sample[0]) for sample in batch]
        max_source_length = max(source_length)
        sources = [sample[0] + [pad_token] * (max_source_length - len(sample[0])) for sample in batch]

        if is_train:
            # 在这里先对response加上一个终止符<eos>
            targets = [sample[1] + [eos_token] for sample in batch]
            target_length = [len(target) for target in targets]
            max_target_length = max(target_length)

            # 对response按最大长度补齐
            pad_targets = [target + [pad_token] * (max_target_length - len(target)) for target in targets]

            return dict(sources=sources, targets=pad_targets,
                        source_length=source_length, target_length=target_length)

        return dict(sources=sources, source_length=source_length)

    def next_batch(self, data: list) -> dict:
        """
        用生成器的形式返回batch数据
        :param data:
        :return:
        """
        random.shuffle(data)
        batch_num = len(data) // self.batch_size

        for i in range(batch_num):
            batch_data = data[self.batch_size * i: self.batch_size * (i + 1)]
            new_batch = self._process_data(batch_data)
            yield new_batch