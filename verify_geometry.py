from src.geometry import build_profile_face, REFERENCE_PARAMS
from build123d import GeomType

def check_geometry():
    print("Generating profile face...")
    face = build_profile_face(REFERENCE_PARAMS)
    print("Face generated. Checking edges...")
    
    curved_edges = [e for e in face.edges() if e.geom_type != GeomType.LINE]
    print(f"Total edges: {len(face.edges())}")
    print(f"Curved edges: {len(curved_edges)}")
    
    for i, e in enumerate(curved_edges):
        print(f"  Curve {i}: {e.geom_type}, Radius={getattr(e, 'radius', 'N/A')}")

    if len(curved_edges) < 2:
        print("FAIL: Expected at least 2 curved edges.")
    else:
        print("PASS: Found curves.")

if __name__ == "__main__":
    check_geometry()
