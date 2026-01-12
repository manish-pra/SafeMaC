import unittest
import numpy as np
from openweathermap.KGS_environment import GridFunction

class TestGridFunction(unittest.TestCase):

    def setUp(self):
        """Set up a common GridFunction for testing."""
        self.base_function = np.arange(16).reshape((4, 4))
        self.grid_func = GridFunction(shape=(2, 2), base_function=self.base_function)

    def test_initialization(self):
        """Test that the GridFunction is initialized correctly."""
        self.assertEqual(self.grid_func.shape, (2, 2))
        np.testing.assert_array_equal(self.grid_func.base_function, self.base_function)
        np.testing.assert_array_equal(self.grid_func.current_function, self.base_function)

    def test_upsample(self):
        """Test the upsample method."""
        self.grid_func.upsample(current=False, factors=(2, 2))
        self.assertEqual(self.grid_func.current_function.shape, (8, 8))

    def test_downsample(self):
        """Test the downsample method."""
        self.grid_func.downsample(current=False, deltas=(2, 2))
        self.assertEqual(self.grid_func.current_function.shape, (2, 2))
        expected_result = np.array([[0, 2], [8, 10]])
        np.testing.assert_array_equal(self.grid_func.current_function, expected_result)

    def test_crop(self):
        """Test the crop method."""
        self.grid_func.crop(current=False, offsets=(1, 1), shape=(2, 2))
        self.assertEqual(self.grid_func.current_function.shape, (2, 2))
        expected_result = np.array([[5, 6], [9, 10]])
        np.testing.assert_array_equal(self.grid_func.current_function, expected_result)

    def test_fit2shape_upsample_crop(self):
        """Test fit2shape when upsampling is needed."""
        self.grid_func.current_function = np.arange(4).reshape((2,2))
        self.grid_func.shape = (3,3)
        self.grid_func.fit2shape()
        self.assertEqual(self.grid_func.current_function.shape, (3, 3))

    def test_fit2shape_downsample_crop(self):
        """Test fit2shape when downsampling is needed."""
        self.grid_func.current_function = np.arange(25).reshape((5,5))
        self.grid_func.shape = (2,2)
        self.grid_func.fit2shape()
        self.assertEqual(self.grid_func.current_function.shape, (2, 2))

    def test_call(self):
        """Test the __call__ method."""
        self.grid_func.current_function = np.array([[1, 2], [3, 4]])
        self.grid_func.shape = (2, 2)
        self.assertEqual(self.grid_func(0, 1), 2)
        self.assertEqual(self.grid_func(1, 0), 3)

    def test_call_with_wrong_shape(self):
        """Test that __call__ raises an error if the shape is wrong."""
        self.grid_func.current_function = np.array([[1, 2, 3], [4, 5, 6]])
        self.grid_func.shape = (2, 2)
        with self.assertRaises(AssertionError):
            self.grid_func(0, 0)

if __name__ == '__main__':
    unittest.main()
