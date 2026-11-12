import numpy as np
import json
import triton_python_backend_utils as pb_utils
from clip_tokenizer import FullTokenizer, default_vocab



class TritonPythonModel:
    """clip 模型文本预处理模型"""
    def initialize(self, args):
        vocab_path = default_vocab()
        self.tokenizer = FullTokenizer(vocab_file=vocab_path)
        self.model_config = json.loads(args['model_config'])
        self.input_ids_config = pb_utils.get_output_config_by_name(
            self.model_config, "INPUT_IDS"
        )
        self.input_ids_dtype = pb_utils.triton_string_to_numpy(self.input_ids_config['data_type'])

    def _execute_request(self, request):
        in_text = pb_utils.get_input_tensor_by_name(request, "TEXT")
        if in_text is None:
            raise ValueError("missing required input 'TEXT'")
        values = in_text.as_numpy()
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(
                "input 'TEXT' must have shape [batch, 1], "
                f"got {values.shape}"
            )
        if values.shape[0] == 0:
            raise ValueError("input 'TEXT' batch is empty")

        input_ids_rows = []
        for index in range(values.shape[0]):
            try:
                value = values[index, 0]
                if isinstance(value, (str, np.str_)):
                    text = str(value)
                elif isinstance(value, (bytes, bytearray, np.bytes_)):
                    text = bytes(value).decode("utf-8")
                else:
                    raise TypeError("text value is not encoded bytes")
                input_ids = self.process_texts([text])
                input_ids_rows.append(input_ids[0])
            except Exception as exc:
                raise ValueError(
                    f"unable to preprocess text at batch index {index}"
                ) from exc

        input_ids = np.stack(input_ids_rows, axis=0)
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "INPUT_IDS", input_ids.astype(self.input_ids_dtype)
                )
            ]
        )

    def execute(self, requests):
        """
        处理推理请求
        """
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_request(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            f"CLIP text preprocessing failed: {exc}"
                        )
                    )
                )
        return responses
    
    def process_texts(self, texts, context_length=52):
        """文本预处理"""
        batch_size = len(texts)
        input_ids = np.zeros((batch_size, context_length), dtype=np.int64)
        for i, text in enumerate(texts):
            tokens = ['[CLS]'] + self.tokenizer.tokenize(text)[:context_length-2] + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_ids[i, :len(token_ids)] = token_ids
        
        return input_ids
    
    def finalize(self):
        """
        模型释放回调
        """
        print('Cleaning up CLIP text preprocessing...')
