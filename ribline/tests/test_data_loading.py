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

        # Test predict_proba and apply_sigmoid
        x = torch.randn(1, 1, 16, 16, 16)
        probs = unet_loaded.predict_proba(x)
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))
        unet_sig = RibUNet3D(in_channels=1, out_channels=1, base_filters=8, apply_sigmoid=True)
        unet_sig.eval()
        with torch.no_grad():
            out_sig = unet_sig(x)
        self.assertTrue(torch.all((out_sig >= 0.0) & (out_sig <= 1.0)))

    def test_nifti_discovery_and_small_volume_patch(self):
        from ribline.model.unet.dataset import RibDLDataset, RibPatchDataset3D

        self.assertIs(RibPatchDataset3D, RibDLDataset)
        # Write a small 12x12x12 NIfTI volume (.nii.gz)
        arr = np.random.randn(12, 12, 12).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((2.0, 2.0, 2.0))
        sitk.WriteImage(img, os.path.join(self.img_dir, "nifti_sub.nii.gz"))
        self.create_dummy_centerline("nifti_sub")

        ds = RibDataset(self.img_dir, self.gt_dir)
        self.assertIn("nifti_sub", ds.subjects)

        patch_ds = RibPatchDataset3D(
            ds,
            cache_dir=os.path.join(self.test_dir, "mask_cache"),
            patch_size=(16, 16, 16),
            samples_per_volume=2,
        )
        sample = patch_ds[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 16, 16, 16))
        self.assertEqual(tuple(sample["mask"].shape), (1, 16, 16, 16))

    def test_hub_load_pretrained_models(self):
        import torch
        from ribline import (
            RibCageStatisticalAtlas,
            RibForest,
            RibUNet3D,
            load_pretrained_models,
        )

        hub_dir = os.path.join(self.test_dir, "mock_hub")
        os.makedirs(hub_dir, exist_ok=True)
        atlas = RibCageStatisticalAtlas()
        atlas.save(os.path.join(hub_dir, "rib_atlas.joblib"))

        rf = RibForest(n_estimators=2, max_depth=2, n_jobs=1)
        rf.fit(np.random.randn(10, 4).astype(np.float32), np.array([0, 1] * 5, dtype=np.int32))
        rf.save(os.path.join(hub_dir, "rib_forest_anatomical.joblib"))
        with open(os.path.join(hub_dir, "feature_config_anatomical.json"), "w") as f:
            f.write('{"scales": [1.0, 2.0, 4.0], "rsid_params": {"num_pairs": 10, "max_offset": 12, "seed": 42}, "use_anatomical_features": true}')
        with open(os.path.join(hub_dir, "train_config_unet.json"), "w") as f:
            f.write('{"model_config": {"base_filters": 16}}')

        unet = RibUNet3D(in_channels=1, out_channels=1, base_filters=16)
        torch.save(unet.state_dict(), os.path.join(hub_dir, "rib_unet_3d.pth"))

        bundle = load_pretrained_models("wq2012/ribline-3d-ct-centerline", local_dir=hub_dir, device="cpu")
        self.assertEqual(bundle["local_dir"], hub_dir)
        self.assertIsInstance(bundle["atlas"], RibCageStatisticalAtlas)
        self.assertIsInstance(bundle["forest"], RibForest)
        self.assertIsInstance(bundle["unet"], RibUNet3D)


if __name__ == '__main__':
    unittest.main()

