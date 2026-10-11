import unittest
import os
import shutil
import tempfile
import json
import numpy as np
import struct
from ribline.pipeline.train import train

class TestTrainFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.test_dir, 'vol2mm')
        self.gt_dir = os.path.join(self.test_dir, 'Centerlines_GT')
        self.output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(self.img_dir)
        os.makedirs(self.gt_dir)
        
        # Create dummy subject
        self.create_dummy_analyze("sub1")
        self.create_dummy_centerline("sub1")
        
        # Create Config
        self.config_path = os.path.join(self.test_dir, 'test_config.json')
        config = {
            "data_config": {
                "image_directory": self.img_dir,
                "centerline_directory": self.gt_dir,
                "select_files": ["sub1"]
            },
            "model_config": {
                "n_estimators": 2,
                "n_jobs": 1
            },
            "training_config": {
                "output_directory": self.output_dir,
                "sampling_rate": 0.5
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def create_dummy_analyze(self, name, shape=(20, 20, 20)):
        hdr_path = os.path.join(self.img_dir, f"{name}.hdr")
        img_path = os.path.join(self.img_dir, f"{name}.img")
        
        header = bytearray(348)
        struct.pack_into('<h', header, 40, 3) 
        struct.pack_into('<h', header, 42, shape[0]) 
        struct.pack_into('<h', header, 44, shape[1]) 
        struct.pack_into('<h', header, 46, shape[2]) 
        struct.pack_into('<h', header, 70, 16) 
        struct.pack_into('<f', header, 76, 1.0)
        struct.pack_into('<f', header, 80, 1.0)
        struct.pack_into('<f', header, 84, 1.0)
        struct.pack_into('<f', header, 88, 1.0)
        
        with open(hdr_path, 'wb') as f:
            f.write(header)
            
        data = np.random.rand(*shape[::-1]).astype(np.float32)
        data.tofile(img_path)

    def create_dummy_centerline(self, name):
        p = os.path.join(self.gt_dir, f"{name}_interp.xls")
        with open(p, 'w') as f:
            # Add some points in the middle
            f.write("R01\t10.0\t10.0\t10.0\t0\t0\n")
            f.write("R01\t10.0\t10.0\t11.0\t0\t0\n")

    def test_train_execution(self):
        # Run training
        try:
            train(self.config_path)
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")
            
        # Check output
        model_path = os.path.join(self.output_dir, "rib_forest.joblib")
        self.assertTrue(os.path.exists(model_path))

if __name__ == '__main__':
    unittest.main()
