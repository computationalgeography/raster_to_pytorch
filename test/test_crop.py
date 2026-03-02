import os
import unittest
from lib.crop import crop_northern_hemisphere
from osgeo import gdal

class TestCrop(unittest.TestCase):

    TEST_INPUT = "test/data/test_raster.tif"
    TEST_OUTPUT = "test/result/test_raster_crop_nh.tif"

    def test_crop_creates_output_file(self):
 
        self.assertTrue(os.path.exists(self.TEST_INPUT))

        crop_northern_hemisphere(self.TEST_INPUT, self.TEST_OUTPUT)

        # Check output file was created
        self.assertTrue(os.path.exists(self.TEST_OUTPUT))

        # Check output can be opened
        ds = gdal.Open(self.TEST_OUTPUT)
        self.assertIsNotNone(ds)

        ds = None


if __name__ == "__main__":
    unittest.main()