from collections import defaultdict
from typing import Any, Dict, List

import sentencepiece as sp
import tempfile
import os
from subprocess import Popen, PIPE


class SentencePieceBPETokenizer:
    r"""
    A tokenizer based on `SentencePiece <https://github.com/google/sentencepiece>`_
    with BPE sub-routine. It encodes caption strings into list of tokens.

    Args:
        model_path: Path to the ``.model`` file trained by SentencePiece.
    """
    SP_SPACE = u"▁"

    def __init__(self, model_path: str):
        self.model_path = model_path

        # Load pretrained tokenizer model.
        self.model = sp.SentencePieceProcessor()
        self.model.Load(model_path)

    def __getstate__(self):
        r"""
        This magic method, along with ``__setstate__`` makes an object of this
        class picklable (and usable while data loading with multiple workers).
        """
        state_dict = self.__dict__.copy()
        state_dict["model"] = None
        return state_dict

    def __setstate__(self, state_dict: Dict[str, Any]):
        self.__dict__ = state_dict

        self.model = sp.SentencePieceProcessor()
        self.model.Load(self.model_path)

    def get_vocab_size(self) -> int:
        r"""Return number of tokens in vocabulary (including special tokens)."""
        return len(self.model)

    def token_to_id(self, token: str) -> int:
        r"""Get integer ID of a string token (``<unk>`` if does not exist)."""
        # Since tokenizer uses subword regularization, one token may break down to multiple IDs.
        # Keep trying till we get a single ID.
        return self.model.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        r"""Get string token of an integer ID (``<unk>`` if does not exist)."""
        return self.model.id_to_piece(token_id)

    def encode(self, text: str) -> List[int]:
        r"""Convert a text string to a list of integer token ids."""
        return self.model.EncodeAsIds(text)

    def decode(self, token_ids: List[int]) -> str:
        r"""Convert a sequence of token IDs to a text string."""
        return self.model.DecodeIds(token_ids)


class PennTreebankTokenizer():
    r"""
    Given a mapping of image id to a list of corresponding captions, tokenize
    captions in place according to Penn Treebank Tokenizer. This method assumes
    the presence of Stanford CoreNLP JAR file in directory of this modules.
    """
    def __init__(
            self, jar_path: str
    ):
        self.CORENLP_JAR = (jar_path)

    def encode(self, image_id_to_captions: Dict[int, List[str]]) -> Dict[int, List[str]]:

        # Prepare data for Tokenizer: write captions to a text file, one per line.
        image_ids = [k for k, v in image_id_to_captions.items() for _ in range(len(v))]
        sentences = "\n".join(
            [c.replace("\n", " ") for k, v in image_id_to_captions.items() for c in v]
        )
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        tmp_file.write(sentences.encode())
        tmp_file.close()

        # fmt: off
        # Tokenize sentences. We use the JAR file for tokenization.
        command = [
            "java", "-cp", self.CORENLP_JAR, "edu.stanford.nlp.process.PTBTokenizer",
            "-preserveLines", "-lowerCase", tmp_file.name
        ]
        tokenized_captions = (
            Popen(command, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=PIPE)
            .communicate(input=sentences.rstrip())[0]
            .decode()
            .split("\n")
        )
        # fmt: on
        os.remove(tmp_file.name)

        # Map tokenized captions back to their image IDs.
        # Punctuations to be removed from the sentences (PTB style)).
        # fmt: off
        PUNCTS = [
            "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?",
            "!", ",", ":", "-", "--", "...", ";",
        ]
        # fmt: on
        image_id_to_tokenized_captions: Dict[int, List[str]] = defaultdict(list)
        for image_id, caption in zip(image_ids, tokenized_captions):
            image_id_to_tokenized_captions[image_id].append(
                " ".join([w for w in caption.rstrip().split(" ") if w not in PUNCTS])
            )

        return image_id_to_tokenized_captions