import json
import multiprocessing

from ralm.retrievers.base_retrieval import BaseRetriever
from pyserini.encode import QueryEncoder, DprQueryEncoder, AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher


class DenseRetriever(BaseRetriever):
    # 初始化密集检索器
    def __init__(self, tokenizer, encoder_name, encoder_type, index_name, result_index, num_tokens_for_query, forbidden_titles_path):
        super(DenseRetriever, self).__init__(tokenizer=tokenizer)
        self.encoder = self._get_encoder(encoder_type, encoder_name)
        self.searcher = self._get_searcher(index_name)
        self.resultSearcher = self._get_lucenesearcher(result_index)
        self.num_tokens_for_query = num_tokens_for_query
        self.forbidden_titles = self._get_forbidden_titles(forbidden_titles_path)

    def _get_encoder(self, encoder_type, encoder_name):
        """
        获取DPR查询编码器  facebook/dpr-question_encoder-multiset-base
        """
        query_encoder_class_map = {
            "dpr": DprQueryEncoder,
            "contriever": AutoQueryEncoder,
        }
        if encoder_name:
            _encoder_type = encoder_type

            # determine encoder_class
            if encoder_type is not None:
                encoder_type = query_encoder_class_map[encoder_type]

            return encoder_type(encoder_name)

        # try:
        #     return DprQueryEncoder(encoder_name)
        # except Exception as e:
        #     print(f"Error loading DPR encoder: {e}")
        #     raise

    def _get_searcher(self, index_name):
        try:
            print(f"Attempting to download the index as if prebuilt by pyserini")
            return FaissSearcher.from_prebuilt_index(index_name, self.encoder)
        except ValueError:
            print(f"Index does not exist in pyserini.")
            print("Attempting to treat the index as a directory (not prebuilt by pyserini)")
            return FaissSearcher(index_name, self.encoder)


    def _get_lucenesearcher(self, result_index):
        try:
            print(f"Attempting to download the index as if prebuilt by pyserini")
            return LuceneSearcher.from_prebuilt_index(result_index)
        except ValueError:
            print(f"Index does not exist in pyserini.")
            print("Attempting to treat the index as a directory (not prebuilt by pyserini)")
            return LuceneSearcher(result_index)

    # 从给定路径中加载标题
    def _get_forbidden_titles(self, forbidden_titles_path):
        if forbidden_titles_path is None:
            return []
        with open(forbidden_titles_path, "r", encoding='utf-8') as f:
            forbidden_titles = [line.strip() for line in f]
        return set(forbidden_titles)

    # 从检索文档中提取标题
    def _get_title_from_retrieved_document(self, doc):
        title, _ = doc.split("\n")
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        return title

    # def _encode_query(self, query_str: str):
    #     """
    #     使用DPR编码器对查询进行编码
    #     """
    #     return self.encoder.encode(query_str)


    def _get_query_string(self, sequence_input_ids, target_begin_location, target_end_location, title=None):
        prefix_tokens = sequence_input_ids[0, :target_begin_location]
        query_tokens = prefix_tokens[-self.num_tokens_for_query:]
        query_str = self.tokenizer.decode(query_tokens)
        return query_str

    def retrieve(self, sequence_input_ids, dataset, k=1):
        queries = [
            self._get_query_string(
                sequence_input_ids,
                d["begin_location"],
                d["end_location"],
                d["title"] if "title" in d else None
            )
            for d in dataset
        ]
        assert len(queries) == len(dataset)
        all_res = self.searcher.batch_search(
            queries,
            q_ids=[str(i) for i in range(len(queries))],
            k=max(100, 4*k) if self.forbidden_titles else k,
            threads=multiprocessing.cpu_count()
        )

        # print(all_res)

        for qid, res in all_res.items():
            qid = int(qid)
            d = dataset[qid]
            d["query"] = queries[qid]
            allowed_docs = []

            for hit in res:
                doc_id = hit.docid
                doc = self.resultSearcher.doc(doc_id)
                res_dict = json.loads(doc.raw())
                context_str = res_dict["contents"]
                title = self._get_title_from_retrieved_document(context_str)
                if title not in self.forbidden_titles:
                    allowed_docs.append({"text": context_str, "score": hit.score})
                    if len(allowed_docs) >= k:
                        break
            d["retrieved_docs"] = allowed_docs
        return dataset