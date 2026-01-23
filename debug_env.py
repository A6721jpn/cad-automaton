import sys
print(sys.executable)
print(sys.path)
try:
    import build123d
    print("build123d imported successfully")
    print(build123d.__file__)
except ImportError as e:
    print(f"Import failed: {e}")
