import h5py

file_path = r'c:\users\user\desktop\ai_test\.venv\lib\site-packages\face_recognition_model'

try:
    with h5py.File(file_path, 'r') as f:
        # Perform operations with the HDF5 file
        pass
except OSError as e:
    print(f"Unable to open HDF5 file: {e}")
