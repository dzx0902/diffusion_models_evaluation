from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ms_video_eval.wan_t5_injection import first_call_context_injector


class FirstCallContextInjectorTest(unittest.TestCase):
    def test_only_positive_call_is_injected(self) -> None:
        native_calls: list[tuple[list[str], str]] = []

        def native_call(_self, texts, device):  # type: ignore[no-untyped-def]
            native_calls.append((list(texts), str(device)))
            return [torch.full((2, 4), 7.0) for _ in texts]

        injected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        patched = first_call_context_injector(native_call, injected)

        positive = patched(object(), ["positive"], "cpu")
        negative = patched(object(), ["negative"], "cpu")

        self.assertEqual(positive[0].dtype, torch.bfloat16)
        torch.testing.assert_close(positive[0].float(), injected)
        torch.testing.assert_close(negative[0], torch.full((2, 4), 7.0))
        self.assertEqual(native_calls, [(["negative"], "cpu")])


if __name__ == "__main__":
    unittest.main()
