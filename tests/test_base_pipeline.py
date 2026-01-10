import os
import pytest
from glow.data_loading.loader import DataLoader

class TestDataLoader:
    def test_natural_sort_key(self):
        loader = DataLoader()
        unsorted = ["A10", "A1", "A2"]
        sorted_list = sorted(unsorted, key=loader._natural_sort_key)
        assert sorted_list == ["A1", "A2", "A10"]

        filename_unsorted = ["file_10.fcs", "file_2.fcs", "file_1.fcs"]
        sorted_files = sorted(filename_unsorted, key=loader._natural_sort_key)
        assert sorted_files == ["file_1.fcs", "file_2.fcs", "file_10.fcs"]

    def test_scan_directory(self, tmp_path):
        # Setup dummy directory structure
        d = tmp_path / "data_scan"
        d.mkdir()
        
        (d / "file_2.fcs").touch()
        (d / "file_1.fcs").touch()
        (d / "file_10.fcs").touch()
        (d / "other.txt").touch() # Should be ignored

        loader = DataLoader()
        files = loader.scan_directory(str(d))
        
        assert len(files) == 3
        # Check sorting
        filenames = [os.path.basename(f) for f in files]
        assert filenames == ["file_1.fcs", "file_2.fcs", "file_10.fcs"]

    def test_parse_file_paths_from_text(self, tmp_path):
        list_file = tmp_path / "list.txt"
        
        # Create dummy fcs files first so they exist
        (tmp_path / "A.fcs").touch()
        (tmp_path / "B.fcs").touch()
        
        content = f"""
{tmp_path}/B.fcs, CustomB
{tmp_path}/A.fcs
# This is a comment

"""
        list_file.write_text(content)
        
        loader = DataLoader()
        parsed = loader.parse_file_paths_from_text(str(list_file))
        
        # Should be sorted naturally by path: A then B
        assert len(parsed) == 2
        assert os.path.basename(parsed[0]['path']) == "A.fcs"
        assert parsed[0]['name'] is None
        
        assert os.path.basename(parsed[1]['path']) == "B.fcs"
        assert parsed[1]['name'] == "CustomB"
