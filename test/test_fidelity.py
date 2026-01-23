import sys
import math
from pathlib import Path
import unittest
import logging

import sys
from pathlib import Path
import unittest
import logging
from build123d import import_step, Shape, Edge, GeomType, Face
from src.geometry import build_profile_face, REFERENCE_PARAMS

# Add src to path if validation needed locally without module run
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFidelity")

class TestGeometricFidelity(unittest.TestCase):
    def setUp(self):
        self.ref_path = project_root / "ref" / "TH1-ref.stp"
        if not self.ref_path.exists():
            self.skipTest(f"Reference file not found: {self.ref_path}")
            
        logger.info(f"Loading reference: {self.ref_path}")
        self.ref_shape = import_step(str(self.ref_path))
        
        # Generate shape using reference parameters
        logger.info("Generating shape from REFERENCE_PARAMS...")
        self.gen_face = build_profile_face(REFERENCE_PARAMS)
        
    def test_01_area_difference(self):
        """Test that the generated shape area is within 0.5% of reference."""
        # Extract Face from reference
        if not isinstance(self.ref_shape, Face):
            # If it's a compound or solid, find the largest face (assuming it's the profile)
            # Or just take the first one if simple
            if hasattr(self.ref_shape, "faces"):
                faces = self.ref_shape.faces()
                if not faces:
                    self.fail("Reference STEP has no faces")
                # Assume the main profile is the largest face if multiple exist (e.g. end caps)
                # But typically a profile step is just one face or one wire.
                # If it's a wire, we might need to make_face. 
                # import_step usually returns a Compound of Solids or Faces.
                ref_face = sorted(faces, key=lambda f: f.area, reverse=True)[0]
            else:
                self.fail(f"Unknown shape type: {type(self.ref_shape)}")
        else:
            ref_face = self.ref_shape

        area_ref = ref_face.area
        area_gen = self.gen_face.area
        
        diff = abs(area_ref - area_gen)
        percent_diff = (diff / area_ref) * 100
        
        logger.info(f"Ref Area: {area_ref:.6f}")
        logger.info(f"Gen Area: {area_gen:.6f}")
        logger.info(f"Diff: {diff:.6f} ({percent_diff:.4f}%)")
        
        self.assertLess(percent_diff, 0.5, f"Area difference {percent_diff:.4f}% exceeds 0.5% limit")

    def test_02_fillet_curvature(self):
        """Test that fillets are curves (CIRCLE/BSPLINE), not straight lines."""
        # Get edges of generated face
        edges = self.gen_face.edges()
        
        # Count non-linear edges
        # LINE is the straight type. Anything else implies curvature.
        curved_edges = [e for e in edges if e.geom_type != GeomType.LINE]
        
        logger.info(f"Found {len(curved_edges)} curved edges from {len(edges)} total edges")
        for i, e in enumerate(curved_edges):
            logger.info(f"  Curve {i}: Type={e.geom_type}, Length={e.length}, Radius={getattr(e, 'radius', 'N/A')}")
        
        # We expect at least 2 curves (R1 and R2)
        self.assertGreaterEqual(len(curved_edges), 2, "Expected at least 2 curved edges (fillets), but found fewer.")
        
        # Verify specific radii if possible (assuming circular arcs)
        radii_found = []
        for e in curved_edges:
            if hasattr(e, 'radius'):
                 radii_found.append(e.radius)
        
        # Check for R1
        has_r1 = any(math.isclose(r, REFERENCE_PARAMS.R1, rel_tol=0.01) for r in radii_found)
        if not has_r1:
            logger.warning(f"R1 ({REFERENCE_PARAMS.R1}) not found strictly in radii: {radii_found}")
            
        # Check for R2
        has_r2 = any(math.isclose(r, REFERENCE_PARAMS.R2, rel_tol=0.01) for r in radii_found)
        if not has_r2:
             logger.warning(f"R2 ({REFERENCE_PARAMS.R2}) not found strictly in radii: {radii_found}")

        # Strict check? User asked for "Match original shape".
        # If the generated shape uses fillets, it should match the Params.
        self.assertTrue(has_r1, f"Failed to find edge with Radius R1={REFERENCE_PARAMS.R1}")
        self.assertTrue(has_r2, f"Failed to find edge with Radius R2={REFERENCE_PARAMS.R2}")

if __name__ == '__main__':
    unittest.main()
