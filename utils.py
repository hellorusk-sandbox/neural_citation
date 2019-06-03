import re
import pandas as pd
import spacy
import logging
import json
from pandas import DataFrame
from pathlib import Path
from ast import literal_eval
from typing import Union, Collection, List, Dict


logging.basicConfig(level=logging.INFO, style='$')
PathOrStr = Union[Path, str]
CITATION_PATTERNS = r"<DBLP:.*?>|<GC:.*?>"

"""
1. Step of preprocessing:
    For each context with all available data create
        Dictionary/JSON of the form:
        {   "context": Tensor w. shapes 300xlen(context),
            "title_citing": Tensor w. shapes 300xlen(title_citing),
            "authors_citing": Vector with length author_vocab,     
            "title_cited": "...",
            "authors_cited": "...",
            ]
        }

2. Step of preprocessing:
    For data JSON:
        1. Tokenize
        2. Lemmatize
        3. Remove formulas
        4. Prune vocabulary?

3. Step of preprocessing:
    Apply GloVe embeddings and store results as torch tensors of the form 1xembed_dimxseq_len

4. Get in consumption format for NCN
"""

def process_text(text: str, delimiter: str = "\n============\n") -> List[str]:
    text = re.sub("<formula>", '', text)
    sentences = text.split(delimiter)
    contexts = []
    for sentence in sentences:
        if re.search(CITATION_PATTERNS, sentence):
            contexts.append(sentence)
    return contexts


def process_refs(refs: str, delimiter: str = "\n") -> List[str]:
    return refs.split(delimiter)


# TODO: FIX ref splitting at \n to GC and DBLP
def generate_json_text(contexts: Collection[str], refs: Collection[str], 
                       meta: Dict[str, str], textpath: Path) -> DataFrame:
    samples = []
    for sentence in contexts:
        hits = re.findall(CITATION_PATTERNS, sentence)
        for hit in hits:
            test = hit[1:-1]
            for ref in refs:
                if re.search(hit[1:-1], ref):
                    author_idx = ref.find(';') + 1
                    data = ref[author_idx:]
                    try:
                        authors, title, *_ = data.split('.')
                        authors = re.sub(r"\band\b", ',', authors)
                        authors = authors.split(',')
                        authors = [author.strip() for author in authors if len(author) > 3]
                    except ValueError:
                        logging.info("Erroneous reference file found: " + textpath.stem)
                    sample = {"context": re.sub(CITATION_PATTERNS, '', sentence),
                            "title_citing": meta["title"],
                            "authors_citing": ','.join(meta["authors"]),
                            "title_cited": title,
                            "authors_cited": ','.join(authors)}
                    samples.append(pd.DataFrame(sample, index=[0]))
    return samples
                    


def prepare_data(path: PathOrStr) -> None:
    """
    Prepare the arxiv CS dataset and save in JSON format.
    INPUTS:
    * __path__(PathOrStr):         Path or string to files
    """
    path = Path(path)
    save_dir =Path("/home/timo/DataSets/KD_arxiv_CS")
    data = []

    for textpath in path.glob("*.txt"):
        metapath = textpath.with_suffix(".meta")
        refpath = textpath.with_suffix(".refs")

        with open(textpath, 'r') as f:
            text = f.read()
        with open(metapath, 'r') as f:
            meta = f.read()
        with open(refpath, 'r') as f:
            refs = f.read()
        
        # throw away incomplete data instances before further processing rest
        if len(text) == 0 or len(meta) == 0 or len(refs) == 0:
            logging.info("Incomplete Data at file " + textpath.stem)
        else:
            # preprocess string data
            meta = json.loads(meta)
            text = process_text(text)
            refs = process_refs(refs)
            data.append(generate_json_text(text, refs, meta, textpath))
    
    # prepare data for storage and save
    dataset = pd.concat(data, axis=0)
    dataset.reset_index(inplace=True)
    dataset.drop("index", axis=1, inplace=True)
    dataset.to_pickle(save_dir/"arxiv_data", compression=None)



def main():
    path_to_data = "/home/timo/DataSets/KD_arxiv_CS/arxiv"
    prepare_data(path_to_data)


if __name__ == '__main__':
    main()