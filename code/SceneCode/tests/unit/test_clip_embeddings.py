import sys
import types
import unittest

from unittest.mock import patch

for _key in [k for k in sys.modules if k == "open_clip" or k.startswith("open_clip.")]:
    del sys.modules[_key]

_open_clip_stub = types.ModuleType("open_clip")
_open_clip_stub.create_model_and_transforms = None
_open_clip_stub.get_tokenizer = None
sys.modules["open_clip"] = _open_clip_stub

from scenecode.agent_utils import clip_embeddings


class TestClipEmbeddings(unittest.TestCase):
    def setUp(self):
        clip_embeddings._cached_model = object()
        clip_embeddings._cached_tokenizer = object()
        clip_embeddings._cached_preprocess = object()
        clip_embeddings._device = "cuda:0"

    def test_reset_clip_model_cache_clears_state_and_cuda_cache(self):
        with patch.object(clip_embeddings.gc, "collect") as mock_gc, patch.object(
            clip_embeddings.torch.cuda, "is_available", return_value=True
        ), patch.object(
            clip_embeddings.torch.cuda, "empty_cache"
        ) as mock_empty_cache, patch.object(
            clip_embeddings.torch.cuda, "ipc_collect"
        ) as mock_ipc_collect:
            clip_embeddings.reset_clip_model_cache()

        self.assertIsNone(clip_embeddings._cached_model)
        self.assertIsNone(clip_embeddings._cached_tokenizer)
        self.assertIsNone(clip_embeddings._cached_preprocess)
        self.assertIsNone(clip_embeddings._device)
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_ipc_collect.assert_called_once()

    def test_reset_clip_model_cache_skips_cuda_cleanup_without_cuda(self):
        with patch.object(clip_embeddings.gc, "collect") as mock_gc, patch.object(
            clip_embeddings.torch.cuda, "is_available", return_value=False
        ), patch.object(
            clip_embeddings.torch.cuda, "empty_cache"
        ) as mock_empty_cache, patch.object(
            clip_embeddings.torch.cuda, "ipc_collect"
        ) as mock_ipc_collect:
            clip_embeddings.reset_clip_model_cache()

        mock_gc.assert_called_once()
        mock_empty_cache.assert_not_called()
        mock_ipc_collect.assert_not_called()
