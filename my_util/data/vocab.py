from .seq import delete_noisy_char
import torchtext
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
from util.general import get_logger
import torch
from tqdm import tqdm

_pjoin = os.path.join
log = get_logger(__name__)

class VocabBuilder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tokenizer = lambda s: delete_noisy_char(s).strip().split()

    def build_from_examples(self, examples, unk="<UNK>", pad="<PAD>", min_freq=1, max_tokens=None):
        vocab = torchtext.vocab.build_vocab_from_iterator(
            [self.tokenizer(e) for e in examples],
            min_freq=min_freq,
            max_tokens=max_tokens,
        )

        vocab_list = vocab.get_itos()
        self.build_from_list(vocab_list, unk=unk, pad=pad)

    def build_from_list(self, vocab_list, unk="<UNK>", pad="<PAD>"):
        self.unk = unk
        self.pad = pad
        self.itos = [pad, unk] + vocab_list
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def load_glove_vectors(self, alias="glove.6B.300d"):
        """load vectors from torchtext
        e.g. "glove.6B.300d"
        """
        cache_dir = _pjoin(self.cfg.root_dir, ".data-bin", "cache")

        pretrained_vocab = torchtext.vocab.pretrained_aliases[alias](cache=cache_dir)
        dim = int(alias.split(".")[-1][:-1])
        embeddings = []

        for s in self.itos:
            if s not in pretrained_vocab.stoi:
                embeddings += [np.zeros(dim)]
            else:
                embeddings += [pretrained_vocab.vectors[pretrained_vocab.stoi[s]]]

        self.glove = np.stack(embeddings)

    def lookup_indices(self, xs):
        ret_indices = []
        for x in xs:
            if x in self.stoi:
                ret_indices += [self.stoi[x]]
            elif x.lower() in self.stoi:
                ret_indices += [self.stoi[x.lower()]]
            else:
                ret_indices += [self.stoi[self.unk]]

        return ret_indices

    def lookup_glove_embeddings(self, sentences):
        assert hasattr(self, "glove"), "no glove generated"
        sent_glove_embeddings = []
        for sent in sentences:
            xs = self.tokenizer(sent)
            inds = self.lookup_indices(xs)
            sent_glove_embeddings += [self.glove[inds]]

        return sent_glove_embeddings

    def lookup_bert_embeddings(self, sentences, use_cuda=True):
        cfg = self.cfg
        assert hasattr(cfg, "bert_name")

        ret_list = []
        cuda_avail = torch.cuda.is_available() and use_cuda

        if not hasattr(self, "bert"):
            self.bert = AutoModel.from_pretrained(cfg.bert_name)
            if cuda_avail:
                log.info("running bert on cuda")
                self.bert = self.bert.cuda()

        if not hasattr(self, "bert_tokenizer"):
            self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.bert_name)

        num_sent = len(sentences)
        for sent in tqdm(sentences, desc="converting bert embeddings..."):
            inputs = self.bert_tokenizer(sent, return_tensors="pt")
            if cuda_avail:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.bert(**inputs)
            last_hidden_states = outputs.last_hidden_state.detach().cpu().squeeze(0).numpy()
            ret_list += [last_hidden_states]

        del self.bert
        del self.bert_tokenizer

        return ret_list

    def lookup_tokens(self, ids):
        return [self.itos[_id] for _id in ids]