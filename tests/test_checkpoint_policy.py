"""Check checkpoint filesystem effects without GPU or model dependencies.

Compile the real trainer checkpoint methods, following the isolated-source
approach used by the geometry tests. Model and distributed I/O are test doubles.
Run with: python -m unittest discover -s tests -p test_checkpoint_policy.py -v
"""

import ast
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint_trainer():
    path = ROOT / "src/fastwam/trainer.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Wan22Trainer")
    methods = {
        "_save_weights_checkpoint", "_save_trainer_state", "save_checkpoint",
        "_load_weight_checkpoint_before_prepare", "_resume_or_load_checkpoint",
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in methods]
    scope = dict(os=os, json=json, Path=Path, logger=logging.getLogger(__name__),
                 ensure_dir=lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    return scope["Wan22Trainer"]


Trainer = load_checkpoint_trainer()


class CheckpointPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkpoints"
        self.trainer = Trainer()
        self.trainer.weights_dir = str(self.root / "weights")
        self.trainer.state_dir = str(self.root / "state")
        Path(self.trainer.weights_dir).mkdir(parents=True)
        self.trainer.global_step = 6000
        self.trainer.epoch = 2
        self.trainer.batch_in_epoch = 17
        self.trainer.safe_checkpoint_serialization = False
        self.trainer.model = Mock()
        self.trainer.model.save_checkpoint.side_effect = (
            lambda path, **kwargs: Path(path).write_bytes(b"model weights"))
        self.trainer.accelerator = Mock(is_main_process=True)
        self.trainer.accelerator.unwrap_model.return_value = self.trainer.model

    def test_weights_only_creates_no_training_state(self):
        self.trainer.save_accelerate_state = False
        result = self.trainer.save_checkpoint()
        weight_path = self.root / "weights/step_006000.pt"
        self.assertTrue(weight_path.is_file())
        self.assertFalse((self.root / "state").exists())
        self.assertIsNone(result["state_path"])
        self.trainer.accelerator.save_state.assert_not_called()
        self.trainer.model.save_checkpoint.assert_called_once_with(
            str(weight_path), optimizer=None, step=6000)

    def test_worker_rank_does_not_write_weights_or_state_when_disabled(self):
        self.trainer.save_accelerate_state = False
        self.trainer.accelerator.is_main_process = False
        self.trainer.save_checkpoint()
        self.assertEqual(list(self.root.rglob("*")), [self.root / "weights"])
        self.trainer.model.save_checkpoint.assert_not_called()
        self.trainer.accelerator.save_state.assert_not_called()
        self.assertEqual(self.trainer.accelerator.wait_for_everyone.call_count, 3)

    def test_explicit_full_state_remains_available_on_all_ranks(self):
        self.trainer.save_accelerate_state = True
        result = self.trainer.save_checkpoint()
        state_path = Path(result["state_path"])
        self.assertEqual(json.loads((state_path / "trainer_state.json").read_text()),
                         dict(global_step=6000, epoch=2, batch_in_epoch=17, full_state=True))
        self.trainer.accelerator.save_state.assert_called_once_with(output_dir=str(state_path))
        self.trainer.accelerator.reset_mock()
        self.trainer.accelerator.is_main_process = False
        self.trainer.save_checkpoint()
        self.trainer.accelerator.save_state.assert_called_once_with(output_dir=str(state_path))

    def test_disabling_state_preserves_existing_checkpoints(self):
        old_state = self.root / "state/step_003000/optimizer.pt"
        old_state.parent.mkdir(parents=True)
        old_state.write_bytes(b"existing optimizer state")
        self.trainer.save_accelerate_state = False
        self.trainer.save_checkpoint()
        self.assertEqual(old_state.read_bytes(), b"existing optimizer state")
        self.assertFalse((self.root / "state/step_006000").exists())

    def test_weight_restart_keeps_new_training_progress(self):
        self.trainer.save_accelerate_state = False
        self.trainer.resume = self.trainer.save_checkpoint()["weights_path"]
        self.trainer.global_step = self.trainer.epoch = self.trainer.batch_in_epoch = 0
        self.trainer.model.load_checkpoint.return_value = 6000
        self.trainer._load_weight_checkpoint_before_prepare()
        self.trainer._resume_or_load_checkpoint()
        self.trainer.model.load_checkpoint.assert_called_once_with(
            self.trainer.resume, optimizer=None)
        self.trainer.accelerator.load_state.assert_not_called()
        self.assertEqual((self.trainer.global_step, self.trainer.epoch,
                          self.trainer.batch_in_epoch), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
