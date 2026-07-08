"""Byte-level BPE tokenizer with UTF-8 fallback."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]
BYTE_OFFSET = 256  # byte tokens occupy ids 0..255 after special tokens


def _bytes_to_unicode() -> Dict[int, str]:
    """Map bytes to visible unicode chars (GPT-2 style)."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE = {v: k for k, v in _BYTE_TO_UNICODE.items()}


class BPETokenizer:
    """Hand-written byte-level BPE tokenizer."""

    def __init__(
        self,
        merges: List[Tuple[str, str]],
        vocab: Dict[int, bytes],
        special_tokens: List[str] | None = None,
    ):
        self.merges = merges
        self.vocab = vocab  # id -> bytes
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        for tid, raw in vocab.items():
            if isinstance(raw, bytes):
                token = "".join(_BYTE_TO_UNICODE[b] for b in raw)
            else:
                token = raw
            self.id_to_token[tid] = token
            self.token_to_id[token] = tid

        self.special_tokens = special_tokens or SPECIAL_TOKENS
        self._merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self._pattern = re.compile(
            "|".join(re.escape(t) for t in self.special_tokens)
            + r"|'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens[0]]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.special_tokens[1]]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.special_tokens[2]]

    def _byte_encode(self, text: str) -> str:
        return "".join(_BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))

    def _byte_decode(self, bpe_str: str) -> str:
        raw = bytes([_UNICODE_TO_BYTE[c] for c in bpe_str])
        return raw.decode("utf-8", errors="replace")

    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _apply_merges(self, word: Tuple[str, ...]) -> Tuple[str, ...]:
        if len(word) < 2:
            return word
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            bigram = min(pairs, key=lambda p: self._merge_ranks.get(p, float("inf")))
            if bigram not in self._merge_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return word

    def _encode_chunk(self, text: str) -> List[int]:
        bpe_str = self._byte_encode(text)
        chars = tuple(bpe_str)
        merged = self._apply_merges(chars)
        ids = []
        for token in merged:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # fallback: single-byte tokens
                for ch in token:
                    byte_val = _UNICODE_TO_BYTE[ch]
                    ids.append(self.token_to_id.get(ch, byte_val + len(self.special_tokens)))
        return ids

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        parts = []
        last = 0
        for m in self._pattern.finditer(text):
            if m.start() > last:
                parts.append(("text", text[last:m.start()]))
            parts.append(("special" if m.group() in self.token_to_id else "text", m.group()))
            last = m.end()
        if last < len(text):
            parts.append(("text", text[last:]))

        ids: List[int] = []
        for kind, piece in parts:
            if kind == "special":
                ids.append(self.token_to_id[piece])
            else:
                ids.extend(self._encode_chunk(piece))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i)
            if tok is None:
                continue
            if tok in self.special_tokens:
                tokens.append(tok)
            else:
                tokens.append(tok)
        text = "".join(tokens)
        # strip special tokens from output
        for sp in self.special_tokens:
            text = text.replace(sp, "")
        return self._byte_decode(text)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [list(m) for m in self.merges],
            "vocab": {str(k): v.hex() if isinstance(v, bytes) else v for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data["merges"]]
        vocab = {}
        for k, v in data["vocab"].items():
            if all(c in "0123456789abcdef" for c in v) and len(v) % 2 == 0:
                try:
                    vocab[int(k)] = bytes.fromhex(v)
                    continue
                except ValueError:
                    pass
            vocab[int(k)] = v
        return cls(merges=merges, vocab=vocab, special_tokens=data.get("special_tokens", SPECIAL_TOKENS))

    @classmethod
    def _text_to_words(cls, text: str, max_word_bytes: int = 256, max_words: int = 8000) -> List[Tuple[str, ...]]:
        """把语料切成短词表单元；中文无空格时必须截断，否则 BPE 极慢。"""
        words: List[Tuple[str, ...]] = []
        for doc in text.split("\n\n"):
            doc = doc.strip()
            if not doc:
                continue
            bpe = cls._byte_encode_static(doc)
            for i in range(0, len(bpe), max_word_bytes):
                chunk = bpe[i : i + max_word_bytes]
                if chunk:
                    words.append(tuple(chunk))
                    if len(words) >= max_words:
                        return words
        if not words and text.strip():
            bpe = cls._byte_encode_static(text.strip())
            words.append(tuple(bpe[:max_word_bytes]))
        return words

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        special_tokens: List[str] | None = None,
        max_word_bytes: int = 256,
        max_words: int = 8000,
    ) -> "BPETokenizer":
        special_tokens = special_tokens or SPECIAL_TOKENS
        vocab: Dict[int, bytes] = {}
        idx = 0
        for sp in special_tokens:
            vocab[idx] = sp  # type: ignore[assignment]
            idx += 1
        for b in range(256):
            vocab[idx + b] = bytes([b])

        words = cls._text_to_words(text, max_word_bytes=max_word_bytes, max_words=max_words)
        if not words:
            words = [tuple(cls._byte_encode_static("hello")[:max_word_bytes])]
        print(f"  BPE corpus: {len(words)} chunks (max_word_bytes={max_word_bytes})")
        merges: List[Tuple[str, str]] = []

        def get_stats(corpus_words):
            pairs: Counter = Counter()
            for word in corpus_words:
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += 1
            return pairs

        def merge_pair(pair, corpus_words):
            a, b = pair
            merged = a + b
            new_words = []
            for word in corpus_words:
                i = 0
                nw: List[str] = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        nw.append(merged)
                        i += 2
                    else:
                        nw.append(word[i])
                        i += 1
                new_words.append(tuple(nw))
            new_bytes = bytes([_UNICODE_TO_BYTE[c] for c in merged])
            return new_words, new_bytes

        target = vocab_size
        n_merges = target - len(vocab)
        while len(vocab) < target:
            stats = get_stats(words)
            if not stats:
                break
            best = stats.most_common(1)[0][0]
            words, new_bytes = merge_pair(best, words)
            merges.append(best)
            vocab[len(vocab)] = new_bytes
            if len(merges) % 50 == 0 or len(merges) == n_merges:
                print(f"  BPE merge {len(merges)}/{n_merges} ...")

        tok = cls(merges=merges, vocab=vocab, special_tokens=special_tokens)
        return tok

    @staticmethod
    def _byte_encode_static(text: str) -> str:
        return "".join(_BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))
