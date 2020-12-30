import os

PackageRoot = os.path.dirname(os.path.dirname(__file__))
DataDirectoryPath = os.path.join(PackageRoot, "data")
SnapshotDirectoryPath = os.path.join(DataDirectoryPath, "snapshot")
DatasetsDirectoryPath = os.path.join(DataDirectoryPath, "datasets")

for d in [DataDirectoryPath, SnapshotDirectoryPath, DataDirectoryPath]:
    if not os.path.exists(d):
        os.makedirs(d)
