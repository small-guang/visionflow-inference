import numpy as np
import triton_python_backend_utils as pb_utils
import os
import gzip
import html
from functools import lru_cache
import ftfy
import regex as re


@lru_cache()
def default_bpe():
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")
    except NameError:
        print("Warning: __file__ not defined. Assuming 'bpe_simple_vocab_16e6.txt.gz' is in the current working directory.")
        return os.path.join(os.getcwd(), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        try:
            with gzip.open(bpe_path, 'rt', encoding="utf-8") as f:
                 merges = f.read().split('\n')
        except FileNotFoundError:
             print(f"Error: BPE vocabulary file not found at {bpe_path}")
             print("Please ensure 'bpe_simple_vocab_16e6.txt.gz' exists.")
             raise

        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))

        if len(vocab) > 49406:
            vocab = vocab[:49406]
        elif len(vocab) < 49406:
            vocab.extend([f'<extra_{i}>' for i in range(49406 - len(vocab))])

        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        if '<|mask|>' in self.encoder:
             self.cache['<|mask|>'] = '<|mask|>'

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = str(text)
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            try:
                 encoded_token = token.encode('utf-8')
            except UnicodeEncodeError:
                 print(f"Warning: Skipping token '{token}' due to encoding issues.")
                 continue
            token_bytes = (self.byte_encoder[b] for b in encoded_token)
            token = ''.join(token_bytes)
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        valid_tokens = [t for t in tokens if t in self.decoder]
        if len(valid_tokens) != len(tokens):
             print(f"Warning: {len(tokens) - len(valid_tokens)} invalid token IDs found during decode.")

        text = ''.join([self.decoder[token] for token in valid_tokens])
        try:
             decoded_bytes = bytearray([self.byte_decoder[c] for c in text])
             text = decoded_bytes.decode('utf-8', errors="replace")
        except KeyError as e:
             print(f"Warning: KeyError during byte decoding for character '{e}'. Replacing.")
             text = ''.join(
                 chr(self.byte_decoder.get(c, 0xFFFD)) for c in text
            ).decode('utf-8', errors="replace")

        text = text.replace('</w>', ' ').strip()
        return text

_tokenizer = SimpleTokenizer()

def tokenize(texts, context_length=77, return_length=False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens_list = []
    original_lengths = []

    for text in texts:
        encoded = _tokenizer.encode(text)
        tokens = [sot_token] + encoded + [eot_token]
        original_lengths.append(len(tokens))
        all_tokens_list.append(np.array(tokens, dtype=np.int64))

    result = np.zeros((len(all_tokens_list), context_length), dtype=np.int64)
    token_lengths = np.zeros(len(all_tokens_list), dtype=np.int64)

    for i, tokens in enumerate(all_tokens_list):
        if len(tokens) > context_length:
            tokens = np.concatenate((
                tokens[0:1],
                tokens[1:context_length - 1],
                tokens[-1:]
            ))
            tokens[-1] = eot_token
            length = context_length
        else:
            length = len(tokens)

        result[i, :length] = tokens
        token_lengths[i] = length

    if return_length:
        return result, token_lengths
    else:
        return result

class TritonPythonModel:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
        
    def _execute_request(self, request):
        in_text = pb_utils.get_input_tensor_by_name(request, "TEXT_IN")
        if in_text is None:
            raise ValueError("missing required input 'TEXT_IN'")
        values = in_text.as_numpy()
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(
                "input 'TEXT_IN' must have shape [batch, 1], "
                f"got {values.shape}"
            )
        if values.shape[0] == 0:
            raise ValueError("input 'TEXT_IN' batch is empty")

        token_rows = []
        for index in range(values.shape[0]):
            try:
                value = values[index, 0]
                if isinstance(value, (str, np.str_)):
                    text = str(value)
                elif isinstance(value, (bytes, bytearray, np.bytes_)):
                    text = bytes(value).decode("utf-8")
                else:
                    raise TypeError("text value is not encoded bytes")
                token_rows.append(tokenize([text])[0])
            except Exception as exc:
                raise ValueError(
                    f"unable to preprocess text at batch index {index}"
                ) from exc

        tokenized = np.stack(token_rows, axis=0)
        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("TEXT_TOKENS", tokenized)]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_request(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"TBPS CLIP text preprocessing failed: {exc}"
                        )
                    )
                )
        return responses

    def finalize(self):
        print('Cleaning up TBPS CLIP text preprocessing...')
