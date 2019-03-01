# 运行apply_bpe.py生成subword vocab，30000为指定的vocab size，中间的为bpe算法的数据，可以用英文维基百科
python learn_bpe.py -s 30000 <"data/lang-8/lang8-train.en"> "data/bpe/subword.txt"

# 运行apply_bpe.py将数据集进行分词处理
python apply_bpe.py -i "data/lang-8/lang8-test.en" -c "data/bpe/subword.txt" -o "data/lang-8/test"
