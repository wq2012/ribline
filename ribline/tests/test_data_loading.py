import unittest
import os
import shutil
import tempfile
import numpy as np
import struct
import SimpleITK as sitk
from ribline.data.image_reader import load_image, read_analyze75_numpy
from ribline.data.centerline_parser import parse_rib_xls
from ribline.data.dataset import RibDataset

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.test_dir, 'vol2mm')
        self.gt_dir = os.path.join(self.test_dir, 'Centerlines_GT')
        os.makedirs(self.img_dir)
        os.makedirs(self.gt_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def create_dummy_analyze(self, name, shape=(10, 10, 10)):
        # Create .hdr and .img
        hdr_path = os.path.join(self.img_dir, f"{name}.hdr")
        img_path = os.path.join(self.img_dir, f"{name}.img")
        
        # Minimal Analyze 7.5 Header
        # We need to set dim (offset 40) and datatype (offset 70) and pixdim (offset 72/76)
        # Header is 348 bytes + padding? usually 348.
        header = bytearray(348)
        
        # Set dims: ndim=3, x, y, z
        struct.pack_into('<h', header, 40, 3) # ndim? offset 40 is dim[0] which is ndim
        struct.pack_into('<h', header, 42, shape[0]) # x
        struct.pack_into('<h', header, 44, shape[1]) # y
        struct.pack_into('<h', header, 46, shape[2]) # z
        
        # DataType: 4 (short), 16 (float)
        # Let's say float (16)
        struct.pack_into('<h', header, 70, 16) 
        
        # Pixdim: 1.0, 1.0, 1.0
        # pixdim starts at 76? verify previously chosen offsets
        struct.pack_into('<f', header, 76, 1.0) # ?
        struct.pack_into('<f', header, 80, 1.0) # x
        struct.pack_into('<f', header, 84, 1.0) # y
        struct.pack_into('<f', header, 88, 1.0) # z
        
        with open(hdr_path, 'wb') as f:
            f.write(header)
            
        # Create data
        data = np.random.rand(*shape[::-1]).astype(np.float32) # z, y, x
        data.tofile(img_path)
        
        return data

    def create_dummy_centerline(self, name):
        p = os.path.join(self.gt_dir, f"{name}_interp.xls")
        with open(p, 'w') as f:
            # RibName \t X \t Y \t Z ...
            f.write("R01\t1.0\t2.0\t3.0\t0\t0\n")
            f.write("R01\t1.1\t2.1\t3.1\t0\t0\n")
            f.write("L02\t5.0\t5.0\t5.0\t0\t0\n")
            
    def test_image_reader(self):
        # Create dummy file
        self.create_dummy_analyze("test_subject")
        
        img_path = os.path.join(self.img_dir, "test_subject.img")
        
        # Test Custom Reader directly
        img = read_analyze75_numpy(img_path)
        self.assertIsInstance(img, sitk.Image)
        self.assertEqual(img.GetSize(), (10, 10, 10))
        
        # Test Unified Loader
        img2 = load_image(img_path)
        self.assertEqual(img2.GetSize(), (10, 10, 10))

    def test_centerline_parser(self):
        self.create_dummy_centerline("test_subject")
        gt_path = os.path.join(self.gt_dir, "test_subject_interp.xls")
        
        data = parse_rib_xls(gt_path)
        self.assertIn("R01", data)
        self.assertIn("L02", data)
        self.assertEqual(len(data["R01"]), 2)
        self.assertEqual(len(data["L02"]), 1)
        self.assertTrue(np.allclose(data["R01"][0], [1.0, 2.0, 3.0]))

    def test_dataset(self):
        self.create_dummy_analyze("subject1")
        self.create_dummy_centerline("subject1")
        
        ds = RibDataset(self.img_dir, self.gt_dir)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds.subjects[0], "subject1")
        
        image, centerlines, sid = ds.load_item(0)
        self.assertIsInstance(image, sitk.Image)
        self.assertEqual(sid, "subject1")
        
        # Test Training Samples
        pos, neg = ds.get_training_samples(0, num_samples=10)
        self.assertTrue(len(pos) > 0 or len(neg) > 0)

    def test_empty_gt_excluded_and_mask_metadata(self):
        self.create_dummy_analyze("subject_valid")
        self.create_dummy_centerline("subject_valid")
        self.create_dummy_analyze("subject_empty")
        empty_gt = os.path.join(self.gt_dir, "subject_empty_interp.xls")
        with open(empty_gt, "w") as f:
            pass  # 0-byte file

        ds = RibDataset(self.img_dir, self.gt_dir)
        self.assertEqual(ds.subjects, ["subject_valid"])

        image, centerlines, _ = ds.load_item(0)
        image.SetSpacing((2.0, 2.0, 2.0))
        image.SetOrigin((10.0, 20.0, 30.0))
        mask_img = ds.generate_mask(image, centerlines)
        self.assertEqual(mask_img.GetSpacing(), (2.0, 2.0, 2.0))
        self.assertEqual(mask_img.GetOrigin(), (10.0, 20.0, 30.0))

    def test_model_loaders_and_aliases(self):
        import torch
        from ribline import RibCageStatisticalAtlas, RibForest, RibUNet, RibUNet3D
        from ribline.model.shape_prior import RibCageStatisticalAtlas as AtlasAlias

        self.assertIs(RibUNet3D, RibUNet)
        self.assertIs(AtlasAlias, RibCageStatisticalAtlas)

        # Test RibForest.load both on class and on instance
        rf = RibForest(n_estimators=4, max_depth=3, n_jobs=1)
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int32)
        rf.fit(X, y)
        rf_path = os.path.join(self.test_dir, "rf_test.joblib")
        rf.save(rf_path)

        rf_cls = RibForest.load(rf_path)
        self.assertIsInstance(rf_cls, RibForest)
        self.assertEqual(rf_cls.predict_proba(X).shape, (20, 2))

        rf_inst = RibForest()
        rf_inst.load(rf_path)
        self.assertEqual(rf_inst.predict_proba(X).shape, (20, 2))

        # Test RibUNet3D.load
        unet = RibUNet3D(in_channels=1, out_channels=1, base_filters=8)
        unet_path = os.path.join(self.test_dir, "unet_test.pth")
        torch.save(unet.state_dict(), unet_path)
        unet_loaded = RibUNet3D.load(unet_path)
        self.assertIsInstance(unet_loaded, RibUNet)


if __name__ == '__main__':
    unittest.main()

