"""有界、保序的图片解码执行器；不依赖 Triton、OpenCV 或 CUDA。"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar


Sample = TypeVar("Sample")
Value = TypeVar("Value")


@dataclass(frozen=True)
class DecodedSlot(Generic[Value]):
    """普通解码异常只影响当前槽位，调用者决定所属请求的错误响应。"""

    value: Value | None = None
    error: Exception | None = None


class BoundedImageDecoder(Generic[Sample, Value]):
    """复用线程池，并将每个实例的在途任务限制在一个 batch 内。

    单图和 workers=1 在调用线程直接执行。多图调用按输入顺序返回，
    多个 decode 调用串行进入窗口，避免 ThreadPoolExecutor 的无界队列
    积累跨 batch 工作。decode_one 不得递归调用本实例的 decode/close。
    """

    def __init__(
        self,
        decode_one: Callable[[Sample], Value],
        *,
        workers: int,
        max_batch_size: int = 4,
    ) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if (isinstance(max_batch_size, bool)
                or not isinstance(max_batch_size, int) or max_batch_size < 1):
            raise ValueError("max_batch_size must be a positive integer")
        if not callable(decode_one):
            raise TypeError("decode_one must be callable")
        self._decode_one = decode_one
        self._max_batch_size = max_batch_size
        self._gate = Lock()
        self._closed = False
        self._executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="triton-jpeg-decode")
            if workers > 1 else None
        )

    def _decode_slot(self, sample: Sample) -> DecodedSlot[Value]:
        try:
            return DecodedSlot(value=self._decode_one(sample))
        except Exception as error:
            return DecodedSlot(error=error)

    def decode(self, samples: Sequence[Sample]) -> list[DecodedSlot[Value]]:
        with self._gate:
            if self._closed:
                raise RuntimeError("image decoder is closed")
            if not isinstance(samples, Sequence):
                raise TypeError("samples must be a bounded sequence")
            size = len(samples)
            if not 1 <= size <= self._max_batch_size:
                raise ValueError(f"batch size must be between 1 and {self._max_batch_size}")
            # 只读取已经校验的数量，不消费可能无限的迭代器。
            items = tuple(samples[index] for index in range(size))
            if size == 1 or self._executor is None:
                return [self._decode_slot(sample) for sample in items]
            futures = []
            try:
                for sample in items:
                    futures.append(self._executor.submit(self._decode_slot, sample))
                return [future.result() for future in futures]
            finally:
                # 提交失败或 BaseException 也先排空当前窗口，再允许下次调用。
                # KeyboardInterrupt/SystemExit 仍向上传播，不转换为图片错误。
                wait_futures(futures)

    def close(self, wait: bool = True) -> None:
        """停止接收工作；等待在途 decode 排空后关闭线程池，可重复调用。

        wait=False 仅影响空闲线程的 shutdown 等待；仍需等待已进入窗口的
        decode 完成，以免其结果在模型 finalize 后继续访问模型资源。
        """
        with self._gate:
            if self._closed:
                return
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=wait)
