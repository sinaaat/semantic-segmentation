import unittest
from src.data.cityscapes_dataset import CityscapesDataLoader
import os
from pathlib import Path


class TestCityscapesDataLoader(unittest.TestCase):
    def test_dataloader(self):
        self.data_dir = Path(__file__).parents[2] / 'data'
        self.split = 'train'
        self.mode = 'fine'
        self.target_type = 'semantic'
        self.batch_size = 4
        self.shuffle = True

        self.cityscapes_dataloader = CityscapesDataLoader(data_dir=self.data_dir, split=self.split, mode=self.mode, target_type=self.target_type, batch_size=self.batch_size, shuffle=self.shuffle)
        self.dataloader = self.cityscapes_dataloader.get_dataloader()

    def test_cityscapes_dataloader(self):
        for images, labels in self.dataloader:
            self.assertEquals(images.shape[1:], (self.batch_size, 3, 1024, 2048))
            self.assertEquals(len(labels), 4)

if __name__ == '__main__':
    unittest.main()
