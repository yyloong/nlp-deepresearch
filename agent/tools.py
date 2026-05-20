from .browsecomp_searcher import BrowseCompBM25Searcher


def build_searcher(index_path: str) -> BrowseCompBM25Searcher:
    return BrowseCompBM25Searcher(index_path=index_path)
