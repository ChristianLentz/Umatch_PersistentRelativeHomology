


import unittest
from . import dissimilarity


class TestCase(unittest.TestCase):
    def test_dissimilarity(self):
        """
        Unit testing for oat_python/oat_python.dissimilarity.py
        """
        dissimilarity.test_dissimilarity_matrix(max_grid_size=3)
        
    def test_persistent_relative_homology(self): 
        """
        Unit testing for Rust struct FactoredBoundaryMatrixVrRelative at oat_python/src/clique_filtered.rs
        """
        # Generate data 
        
        # Construct factored boundary oracle and associated barcode 
        
        




if __name__ == '__main__':
    unittest.main()        


