import os

PackageRoot = os.path.dirname(os.path.dirname(__file__))
DataDirectoryPath = os.path.join(PackageRoot, "data")
SnapshotDirectoryPath = os.path.join(DataDirectoryPath, "snapshot")

for d in [DataDirectoryPath, SnapshotDirectoryPath]:
    if not os.path.exists(d):
        os.makedirs(d)
