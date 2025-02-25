import unittest
import os
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import folium
import re
import io
import sys
from unittest.mock import patch
from IPython.display import display

class TestGISAssignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the notebook
        try:
            with open('student_submission.ipynb', 'r', encoding='utf-8') as f:
                cls.notebook = nbformat.read(f, as_version=4)
            
            # Extract code cells for testing
            cls.code_cells = {}
            for i, cell in enumerate(cls.notebook.cells):
                if cell.cell_type == 'code':
                    cls.code_cells[i] = cell
            
            # Execute the notebook
            executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
            
            # Redirect stdout during execution
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                # Execute the notebook
                executor.preprocess(cls.notebook)
                cls.execution_successful = True
            except Exception as e:
                cls.execution_successful = False
                cls.execution_error = str(e)
            finally:
                # Restore stdout
                cls.execution_output = sys.stdout.getvalue()
                sys.stdout = original_stdout
                
            # Get the globals after execution for testing
            cls.globals = {}
            for cell in cls.notebook.cells:
                if cell.cell_type == 'code':
                    exec(cell.source, cls.globals)
                    
        except FileNotFoundError:
            print("Student submission notebook not found.")
            cls.notebook = None
            cls.execution_successful = False
    
    def test_00_notebook_execution(self):
        """Test if the notebook executes without errors."""
        self.assertTrue(self.execution_successful, 
                       f"Notebook execution failed with error: {getattr(self, 'execution_error', 'Unknown error')}")
    
    def test_01_required_libraries(self):
        """Test if all required libraries are imported."""
        required_libraries = ['geopandas', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'folium', 'contextily']
        
        # Get all import statements from the notebook
        import_statements = []
        for cell in self.notebook.cells:
            if cell.cell_type == 'code':
                # Extract import statements using regex
                imports = re.findall(r'^\s*import\s+(\w+)|^\s*from\s+(\w+)', cell.source, re.MULTILINE)
                for imp in imports:
                    # Each match is a tuple (direct_import, from_import)
                    import_statements.extend([i for i in imp if i])
        
        for lib in required_libraries:
            self.assertIn(lib, import_statements, f"Required library '{lib}' is not imported")
    
    def test_02_load_shapefile(self):
        """Test if the Tanzania shapefile is loaded correctly."""
        # Check if the shapefile variable exists
        self.assertIn('tz_shapefile', self.globals, "Tanzania shapefile variable 'tz_shapefile' not found")
        
        # Check if it's a GeoDataFrame
        self.assertIsInstance(self.globals['tz_shapefile'], gpd.GeoDataFrame, 
                             "Tanzania shapefile should be a GeoDataFrame")
        
        # Check if it has some rows (not empty)
        self.assertGreater(len(self.globals['tz_shapefile']), 0, 
                          "Tanzania shapefile should not be empty")
        
        # Check if geometry column exists
        self.assertIn('geometry', self.globals['tz_shapefile'].columns, 
                     "Tanzania shapefile should have a 'geometry' column")
    
    def test_03_describe_geodataframe(self):
        """Test if the describe_geodataframe function works correctly."""
        self.assertIn('describe_geodataframe', self.globals, 
                     "Function 'describe_geodataframe' not found")
        
        # Test with a simple GeoDataFrame
        test_gdf = gpd.GeoDataFrame(
            {'col1': [1, 2], 'geometry': gpd.points_from_xy([0, 1], [0, 1])},
            crs="EPSG:4326"
        )
        
        # Run the function
        result = self.globals['describe_geodataframe'](test_gdf)
        
        # Check if it returns a dictionary
        self.assertIsInstance(result, dict, 
                             "describe_geodataframe should return a dictionary")
        
        # Check if it has the required keys
        required_keys = ['crs', 'geometry_type', 'num_features', 'attributes', 'bounds']
        for key in required_keys:
            self.assertIn(key, result, f"describe_geodataframe result should have '{key}' key")
        
        # Check specific values
        self.assertEqual(result['num_features'], 2, 
                        "describe_geodataframe should correctly count features")
        self.assertEqual(result['crs'], test_gdf.crs, 
                        "describe_geodataframe should correctly identify CRS")
    
    def test_04_reproject_data(self):
        """Test if data reprojection works correctly."""
        self.assertIn('tz_projected', self.globals, 
                     "Reprojected Tanzania shapefile variable 'tz_projected' not found")
        
        tz_shapefile = self.globals['tz_shapefile']
        tz_projected = self.globals['tz_projected']
        
        # Check if tz_projected is a GeoDataFrame
        self.assertIsInstance(tz_projected, gpd.GeoDataFrame, 
                             "tz_projected should be a GeoDataFrame")
        
        # Check if CRS was changed
        self.assertNotEqual(tz_shapefile.crs, tz_projected.crs, 
                           "The CRS should be changed after reprojection")
        
        # Check if appropriate CRS for Tanzania was used (EPSG:21037 or similar)
        # We'll check if it's a projected CRS, not necessarily exactly EPSG:21037
        self.assertTrue(tz_projected.crs.is_projected, 
                       "The reprojected data should use a projected CRS")
    
    def test_05_compare_projections(self):
        """Test if the compare_projections function works correctly."""
        self.assertIn('compare_projections', self.globals, 
                     "Function 'compare_projections' not found")
        
        # Get the function
        compare_projections = self.globals['compare_projections']
        
        # Create test data
        test_gdf1 = gpd.GeoDataFrame(
            {'col1': [1]}, 
            geometry=[gpd.box(0, 0, 1, 1)],
            crs="EPSG:4326"
        )
        test_gdf2 = test_gdf1.to_crs(epsg=3857)
        
        # Run the function
        result = compare_projections(test_gdf1, test_gdf2)
        
        # Check if it returns a dictionary
        self.assertIsInstance(result, dict, 
                