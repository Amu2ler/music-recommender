import os
import glob

files = glob.glob("ui/pages/*")
print(f"Found files: {files}")

for f in files:
    basename = os.path.basename(f)
    if basename.startswith("1_") or basename.startswith("2_"):
        try:
            os.remove(f)
            print(f"✅ Deleted: {f}")
        except Exception as e:
            print(f"❌ Failed to delete {f}: {e}")
