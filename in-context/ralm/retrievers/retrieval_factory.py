"""

in-context所需参数：
    1、首先进行检索分类判断，   即稀疏 sparse 密集 dense
                            num_docs，检索片段限制
    2、进行数据集参数判断，    hf:huggingface路径 或者 file 自定义数据集路径
                        数据集路径、数据集名称、数据集划分
    3、进行检索模型参数判断，   tokenizer_name，编码模型名称
                            max_length，设定模型输入最大长度
                            stride，in-context进行的步长查询
    4、输出文件名称

稀疏检索器额外配置：
    forbidden_titles_path：针对wikitext103数据集的特定操作
    index_name：所用数据集的索引，例如 wikipedia-dpr
    num_tokens_for_query，检索时所使用的前缀token

密集检索器额外配置：
    model_type：模型类型
    model_name：模型名称
    index_name：所用数据集的索引，例如 wikipedia-dpr-100w.dpr_multi
    forbidden_titles_path：针对wikitext103数据集的特定操作
    num_tokens_for_query，检索时所使用的前缀token

获取DPR查询编码器  facebook/dpr-question_encoder-multiset-base


"""

def add_retriever_args(parser, retriever_type):
    if retriever_type == "sparse":
        parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")
        parser.add_argument("--num_tokens_for_query", type=int, default=32)

    elif retriever_type == "dense":
        parser.add_argument("--model_type", type=str, default='dpr', choices=["dpr", "contriever"])
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--index_name", type=str, default='wikipedia-dpr-100w.dpr_multi')
        parser.add_argument("--result_index", type=str, default='wikipedia-dpr')
        parser.add_argument("--num_tokens_for_query", type=int, default=64)
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")
        # parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"])
        # parser.add_argument("--encoded_files", type=str, required=True)
        # parser.add_argument("--corpus_path", type=str, required=True)
        # parser.add_argument("--batch_size", type=int, default=2048)
        # parser.add_argument("--fp16", action="store_true", default="False")

    else:
        raise ValueError


def get_retriever(retriever_type, args, tokenizer):
    if retriever_type == "sparse":
        from ralm.retrievers.sparse_retrieval import SparseRetriever
        return SparseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            forbidden_titles_path=args.forbidden_titles_path,
            num_tokens_for_query=args.num_tokens_for_query,
        )
        # from ralm.retrievers.learned_sparse_retrieval import SparseRetriever
        # return SparseRetriever(
        #     tokenizer=tokenizer,
        #     index_name=args.index_name,
        #     forbidden_titles_path=args.forbidden_titles_path,
        #     num_tokens_for_query=args.num_tokens_for_query,
        # )
    elif retriever_type == "dense":
        # raise ValueError("We currently don't support dense retrieval, we will soon add this option.")
        from ralm.retrievers.dense_retrieval import DenseRetriever
        return DenseRetriever(
            tokenizer=tokenizer,
            encoder_name=args.model_name,
            encoder_type=args.model_type,
            index_name=args.index_name,
            result_index=args.result_index,
            num_tokens_for_query=args.num_tokens_for_query,
            forbidden_titles_path=args.forbidden_titles_path,
        )
    raise ValueError
