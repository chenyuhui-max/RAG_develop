from rank_bm25 import BM25Okapi
import jieba

# 示例数据集
documents = [
    "这是一篇关于BM25的文章",
    "BM25是一种优秀的检索算法",
    "信息检索中BM25很重要"
]

# 分词
tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
print("分词后的文档：", tokenized_docs)
print(BM25Okapi(tokenized_docs))

# 初始化BM25模型
bm25 = BM25Okapi(tokenized_docs)

# 查询
query = "BM25检索"
tokenized_query = list(jieba.cut(query))
print("分词后的查询：", tokenized_query)

# 计算相关性得分
scores = bm25.get_scores(tokenized_query)
print(scores)