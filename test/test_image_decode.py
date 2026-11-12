"""CPU-only tests for bounded decode scheduling and shutdown ownership."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Barrier, Event, Lock, current_thread
import unittest
from unittest import mock

from scripts import image_decode as production


class BoundedImageDecoderTests(unittest.TestCase):
    def test_result_order_does_not_depend_on_completion_order(self):
        others_done = Event()
        mutex = Lock()
        completed = []

        def decode_one(value):
            if value == 0:
                if not others_done.wait(2):
                    raise AssertionError("other slots did not run concurrently")
            with mutex:
                completed.append(value)
                if len(completed) == 3:
                    others_done.set()
            return value * 10

        decoder = production.BoundedImageDecoder(decode_one, workers=4)
        try:
            slots = decoder.decode([0, 1, 2, 3])
            self.assertEqual([slot.value for slot in slots], [0, 10, 20, 30])
            self.assertTrue(all(slot.error is None for slot in slots))
            self.assertEqual(completed[-1], 0)
        finally:
            decoder.close()

    def test_one_bad_slot_keeps_successful_values_and_original_exception(self):
        failure = ValueError("invalid JPEG")

        def decode_one(value):
            if value == "bad":
                raise failure
            return value

        for workers in (1, 2):
            with self.subTest(workers=workers):
                decoder = production.BoundedImageDecoder(decode_one, workers=workers)
                try:
                    slots = decoder.decode(["first", "bad", None, "last"])
                    self.assertEqual([slot.value for slot in slots], ["first", None, None, "last"])
                    self.assertIs(slots[1].error, failure)
                    self.assertIsNone(slots[2].error)
                finally:
                    decoder.close()

    def test_single_image_and_single_worker_run_in_calling_thread(self):
        caller = current_thread()
        for workers, values in ((4, [1]), (1, [1, 2, 3, 4])):
            decoder = production.BoundedImageDecoder(lambda _: current_thread(), workers=workers)
            try:
                self.assertTrue(all(slot.value is caller for slot in decoder.decode(values)))
            finally:
                decoder.close()

    def test_pool_threads_are_reused_between_batches(self):
        rendezvous = Barrier(2)

        def decode_one(_):
            rendezvous.wait(timeout=2)
            return current_thread()

        decoder = production.BoundedImageDecoder(decode_one, workers=2)
        try:
            first = {slot.value for slot in decoder.decode([1, 2])}
            second = {slot.value for slot in decoder.decode([3, 4])}
            self.assertEqual(len(first), 2)
            self.assertEqual(first, second)
        finally:
            decoder.close()

    def test_concurrent_calls_never_queue_more_than_one_batch(self):
        release = Event()
        first_submitted = Event()
        second_started = Event()
        exceeded_window = Event()
        mutex = Lock()
        outstanding = 0
        maximum = 0

        class TrackingExecutor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                nonlocal outstanding, maximum
                with mutex:
                    outstanding += 1
                    maximum = max(maximum, outstanding)
                    if outstanding == 4:
                        first_submitted.set()
                    if outstanding > 4:
                        exceeded_window.set()
                future = super().submit(fn, *args, **kwargs)

                def finished(_):
                    nonlocal outstanding
                    with mutex:
                        outstanding -= 1

                future.add_done_callback(finished)
                return future

        def decode_one(value):
            if not release.wait(2):
                raise AssertionError("decode release timed out")
            return value

        with mock.patch.object(production, "ThreadPoolExecutor", TrackingExecutor):
            decoder = production.BoundedImageDecoder(decode_one, workers=2)
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                first = callers.submit(decoder.decode, [1, 2, 3, 4])

                def second_call():
                    second_started.set()
                    return decoder.decode([5, 6, 7, 8])

                try:
                    self.assertTrue(first_submitted.wait(2))
                    second = callers.submit(second_call)
                    self.assertTrue(second_started.wait(2))
                    self.assertFalse(exceeded_window.wait(0.05))
                finally:
                    release.set()
                self.assertEqual([s.value for s in first.result(timeout=2)], [1, 2, 3, 4])
                self.assertEqual([s.value for s in second.result(timeout=2)], [5, 6, 7, 8])
            self.assertLessEqual(maximum, 4)
            self.assertEqual(outstanding, 0)
        finally:
            release.set()
            decoder.close()

    def test_rejects_invalid_or_unbounded_input_before_decoding(self):
        decode_one = mock.Mock()
        decoder = production.BoundedImageDecoder(decode_one, workers=2)
        try:
            for samples in ([], [1, 2, 3, 4, 5]):
                with self.assertRaises(ValueError):
                    decoder.decode(samples)

            def unbounded():
                raise AssertionError("must not consume a generator")
                yield 1

            with self.assertRaises(TypeError):
                decoder.decode(unbounded())
            decode_one.assert_not_called()
        finally:
            decoder.close()

    def test_invalid_constructor_arguments_are_rejected(self):
        for invalid in (0, -1, True, 1.5, "2"):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    production.BoundedImageDecoder(lambda value: value, workers=invalid)
                with self.assertRaises(ValueError):
                    production.BoundedImageDecoder(lambda value: value, workers=1, max_batch_size=invalid)
        with self.assertRaises(TypeError):
            production.BoundedImageDecoder(None, workers=1)

    def test_base_exception_propagates_after_other_slots_drain(self):
        other_finished = Event()

        def decode_one(value):
            if value == 0:
                raise KeyboardInterrupt("do not convert to image error")
            other_finished.set()
            return value

        decoder = production.BoundedImageDecoder(decode_one, workers=2)
        try:
            with self.assertRaises(KeyboardInterrupt):
                decoder.decode([0, 1])
            self.assertTrue(other_finished.is_set())
            self.assertEqual(decoder.decode([2])[0].value, 2)
        finally:
            decoder.close()

    def test_close_waits_for_inflight_decode_and_rejects_new_work(self):
        started = Event()
        release = Event()

        def decode_one(value):
            started.set()
            if not release.wait(2):
                raise AssertionError("decode release timed out")
            return value

        decoder = production.BoundedImageDecoder(decode_one, workers=2)
        try:
            with ThreadPoolExecutor(max_workers=2) as callers:
                decoding = callers.submit(decoder.decode, [1, 2])
                try:
                    self.assertTrue(started.wait(2))
                    closing = callers.submit(decoder.close, wait=True)
                    with self.assertRaises(TimeoutError):
                        closing.result(timeout=0.05)
                finally:
                    release.set()
                self.assertEqual([s.value for s in decoding.result(timeout=2)], [1, 2])
                closing.result(timeout=2)
            decoder.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                decoder.decode([3])
        finally:
            release.set()
            decoder.close()


if __name__ == "__main__":
    unittest.main()
