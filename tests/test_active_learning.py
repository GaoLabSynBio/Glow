import os
import json
import pytest
import shutil
import numpy as np
from glow.active_learning.learner import ActiveLearner
from glow.active_learning.storage import ActiveLearningLogger

class TestActiveLearningLogger:
    def test_init_creates_dir(self, tmp_path):
        storage_dir = tmp_path / "al_data_logger"
        logger = ActiveLearningLogger(storage_dir=str(storage_dir))
        assert os.path.exists(storage_dir)

    def test_log_and_load_feedback(self, tmp_path):
        storage_dir = tmp_path / "al_data_logger_io"
        logger = ActiveLearningLogger(storage_dir=str(storage_dir))
        
        sample_data = {
            "sample_name": "S1",
            "cell_line": "HeLa",
            "x_channel": "FSC-A",
            "y_channel": "SSC-A",
            "gate_type": "Polygon",
            "gate_params": {'vertices': [[1,1], [2,2], [3,3]]},
            "user_decision": "accepted"
        }
        
        logger.log_gate_feedback(**sample_data)
        
        # Check file exists
        files = list(os.listdir(storage_dir))
        assert len(files) >= 1
        assert any(f.startswith("feedback_") for f in files)
        
        # Check load
        loaded = logger.load_feedback()
        assert len(loaded) == 1
        entry = loaded[0]
        assert entry['sample_name'] == "S1"
        assert entry['gate_params']['vertices'] == [[1,1], [2,2], [3,3]]

class TestActiveLearner:
    @pytest.fixture
    def learner_env(self, tmp_path):
        storage_dir = tmp_path / "al_data_learner"
        storage_dir.mkdir()
        return storage_dir

    def test_training_logic(self, learner_env):
        logger = ActiveLearningLogger(storage_dir=str(learner_env))
        
        # Log feedback for "Live" (5 vertices)
        logger.log_gate_feedback(
            sample_name="S1", cell_line="HeLa", x_channel="FSC", y_channel="SSC",
            gate_type="Polygon", gate_params={'vertices': [[0,0], [1,0], [1,1], [0,1], [0,0.5]]}, # 5 points
            user_decision="accepted"
        )
        
        # Log feedback for "Singlet" (4 vertices)
        logger.log_gate_feedback(
            sample_name="S1", cell_line="HeLa", x_channel="FSC-A", y_channel="FSC-H",
            gate_type="Polygon", gate_params={'vertices': [[0,0], [1,0], [1,1], [0,1]]}, # 4 points
            user_decision="accepted"
        )

        learner = ActiveLearner(storage_dir=str(learner_env), model_filename="weights.json")
        learner.train()
        
        assert "HeLa" in learner.knowledge_base
        assert "Live" in learner.knowledge_base["HeLa"]
        assert len(learner.knowledge_base["HeLa"]["Live"]) == 5
        assert "Singlet" in learner.knowledge_base["HeLa"]
        assert len(learner.knowledge_base["HeLa"]["Singlet"]) == 4

    def test_predict(self, learner_env):
        learner = ActiveLearner(storage_dir=str(learner_env))
        # Manually inject knowledge
        learner.knowledge_base = {
            "CHO": {
                "Live": [[1,1], [2,2]]
            }
        }
        
        prediction = learner.predict("CHO", "Live")
        assert prediction == [[1,1], [2,2]]
        
        assert learner.predict("CHO", "Singlet") is None
        assert learner.predict("Unknown", "Live") is None

    def test_save_load_weights(self, learner_env):
        learner = ActiveLearner(storage_dir=str(learner_env), model_filename="test_weights.json")
        learner.knowledge_base = {"Test": {"Gate": [1, 2, 3]}}
        learner.save_weights()
        
        assert (learner_env / "test_weights.json").exists()
        
        # New instance
        learner2 = ActiveLearner(storage_dir=str(learner_env), model_filename="test_weights.json")
        assert "Test" in learner2.knowledge_base
        assert learner2.knowledge_base["Test"]["Gate"] == [1, 2, 3]
