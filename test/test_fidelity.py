"""
Test geometric fidelity of generated STEP profiles.

These tests validate that:
1. Generated geometry matches reference area within tolerance
2. Generated geometry contains curved edges (fillets) at R1 and R2
3. Arc radii match the REFERENCE_PARAMS values
"""
import sys
import math
from pathlib import Path
import unittest
import logging

from build123d import import_step, Shape, Edge, GeomType, Face
from src.geometry import build_profile_face, REFERENCE_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFidelity")


class TestGeometricFidelity(unittest.TestCase):
    """Test suite for geometric fidelity of generated profiles."""
    
    @classmethod
    def setUpClass(cls):
        """Load reference STEP once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.ref_path = project_root / "ref" / "TH1-ref.stp"
        
        if not cls.ref_path.exists():
            raise unittest.SkipTest(f"Reference file not found: {cls.ref_path}")
        
        logger.info(f"Loading reference: {cls.ref_path}")
        cls.ref_shape = import_step(str(cls.ref_path))
    
    def setUp(self):
        """Generate shape for each test."""
        logger.info("Generating shape from REFERENCE_PARAMS...")
        self.gen_face = build_profile_face(REFERENCE_PARAMS)
    
    def _get_ref_face(self) -> Face:
        """Extract largest face from reference shape."""
        if isinstance(self.ref_shape, Face):
            return self.ref_shape
        
        if hasattr(self.ref_shape, "faces"):
            faces = self.ref_shape.faces()
            if not faces:
                self.fail("Reference STEP has no faces")
            return sorted(faces, key=lambda f: f.area, reverse=True)[0]
        
        self.fail(f"Unknown shape type: {type(self.ref_shape)}")
    
    def test_01_area_difference(self):
        """Test that generated shape area is within 1.0% of reference."""
        ref_face = self._get_ref_face()
        
        area_ref = ref_face.area
        area_gen = self.gen_face.area
        
        diff = abs(area_ref - area_gen)
        percent_diff = (diff / area_ref) * 100
        
        logger.info(f"Ref Area: {area_ref:.6f}")
        logger.info(f"Gen Area: {area_gen:.6f}")
        logger.info(f"Diff: {diff:.6f} ({percent_diff:.4f}%)")
        
        self.assertLess(
            percent_diff, 1.0, 
            f"Area difference {percent_diff:.4f}% exceeds 1.0% limit"
        )
    
    def test_02_has_curved_edges(self):
        """Test that generated shape has at least 2 curved edges (fillets)."""
        edges = self.gen_face.edges()
        
        # Non-LINE edges are curved (CIRCLE, BSPLINE, etc.)
        curved_edges = [e for e in edges if e.geom_type != GeomType.LINE]
        
        logger.info(f"Found {len(curved_edges)} curved edges from {len(edges)} total")
        for i, e in enumerate(curved_edges):
            logger.info(f"  Curve {i}: Type={e.geom_type}, Length={e.length:.4f}")
        
        # Expect at least 2 curved edges (R1 and R2 fillets)
        self.assertGreaterEqual(
            len(curved_edges), 2,
            f"Expected at least 2 curved edges (fillets), found {len(curved_edges)}"
        )
    
    def test_03_fillet_radius_R1(self):
        """Test that R1 fillet radius matches REFERENCE_PARAMS.R1."""
        edges = self.gen_face.edges()
        curved_edges = [e for e in edges if e.geom_type != GeomType.LINE]
        
        expected_r1 = REFERENCE_PARAMS.R1
        # Arc length for 90 degree fillet = (pi/2) * radius
        expected_arc_length_r1 = (math.pi / 2) * expected_r1
        
        found_r1 = False
        for e in curved_edges:
            arc_len = e.length
            # Check if arc length matches expected (within 5% tolerance)
            if math.isclose(arc_len, expected_arc_length_r1, rel_tol=0.05):
                found_r1 = True
                logger.info(f"Found R1 arc: length={arc_len:.4f}, expected={expected_arc_length_r1:.4f}")
                break
        
        self.assertTrue(
            found_r1,
            f"No arc found matching R1={expected_r1} (expected arc length={expected_arc_length_r1:.4f})"
        )
    
    def test_04_fillet_radius_R2(self):
        """Test that R2 fillet radius matches REFERENCE_PARAMS.R2."""
        edges = self.gen_face.edges()
        curved_edges = [e for e in edges if e.geom_type != GeomType.LINE]
        
        expected_r2 = REFERENCE_PARAMS.R2
        # Arc length for 90 degree fillet = (pi/2) * radius
        expected_arc_length_r2 = (math.pi / 2) * expected_r2
        
        found_r2 = False
        for e in curved_edges:
            arc_len = e.length
            # Check if arc length matches expected (within 5% tolerance)
            if math.isclose(arc_len, expected_arc_length_r2, rel_tol=0.05):
                found_r2 = True
                logger.info(f"Found R2 arc: length={arc_len:.4f}, expected={expected_arc_length_r2:.4f}")
                break
        
        self.assertTrue(
            found_r2,
            f"No arc found matching R2={expected_r2} (expected arc length={expected_arc_length_r2:.4f})"
        )


if __name__ == '__main__':
    unittest.main()

