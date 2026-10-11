import os
import struct
import numpy as np
import SimpleITK as sitk

def read_custom_header(header_path):
    """
    Reads the custom text-based header.
    Format:
    dim_x dim_y dim_z
    spacing_x spacing_y spacing_z
    min_val max_val (optional/ignored for loading)
    """
    with open(header_path, 'r') as f:
        lines = f.readlines()
        
    if len(lines) < 2:
        raise ValueError("Header too short")
        
    dims = [int(x) for x in lines[0].strip().split()]
    spacing = [float(x) for x in lines[1].strip().split()]
    
    # We assume int16 based on file size analysis (2 bytes/pixel)
    dtype = np.int16
    
    # Analyze usually stores x, y, z.
    # We return shape as (x, y, z) for consistency with previous logic, 
    # but remember numpy uses (z, y, x)
    # We return origin as (0,0,0) because GT ignores the header offset
    origin = (0.0, 0.0, 0.0)
    return dims, spacing, dtype, origin

def read_analyze75_header(header_path):
    """
    Reads the header of an Analyze 7.5 file (binary) or custom text header.
    """
    # Try text first as we confirmed it for this dataset
    try:
        return read_custom_header(header_path)
    except Exception:
        pass

    with open(header_path, 'rb') as f:
        header_data = f.read()

    if len(header_data) < 348:
         raise ValueError("Header too short for Analyze 7.5")

    # Dim is at offset 40, 8 shorts (16 bytes)
    dim = struct.unpack_from('<8h', header_data, 40)
    
    # Pixdim is at offset 76, 8 floats (32 bytes)
    pixdim = struct.unpack_from('<8f', header_data, 76)
    
    # DataType is at offset 70 (short)
    datatype_code = struct.unpack_from('<h', header_data, 70)[0]
    
    dtype_map = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64
    }
    
    if datatype_code not in dtype_map:
        # Fallback or error? 
        # If we reached here, text parse failed.
        # Maybe it IS Big Endian?
        datatype_code_be = struct.unpack_from('>h', header_data, 70)[0]
        if datatype_code_be in dtype_map:
             datatype_code = datatype_code_be
             # We should switch to BE unpacking for everything, but let's assume LE for now
        else:
             raise ValueError(f"Unsupported Analyze 7.5 datatype code: {datatype_code}")
        
    dtype = dtype_map[datatype_code]
    
    ndim = dim[0]
    shape = dim[1:ndim+1]
    spacing = pixdim[1:ndim+1]
    
    return shape, spacing, dtype, (0.0, 0.0, 0.0) # Binary fallback origin

def read_analyze75_numpy(path):
    """
    Reads an Analyze 7.5 image pair (.img/.hdr) using numpy.
    Args:
        path (str): Path to .img or .hdr file.
    Returns:
        sitk.Image: SimpleITK Image object created from numpy array.
    """
    base, ext = os.path.splitext(path)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
        
    hdr_path = base + '.hdr'
    img_path = base + '.img'
    
    if not os.path.exists(hdr_path) or not os.path.exists(img_path):
        raise FileNotFoundError(f"Analyze 7.5 pair not found: {hdr_path}, {img_path}")
        
    shape, spacing, dtype, origin = read_analyze75_header(hdr_path)
    
    # Determine total elements
    num_elements = np.prod(shape)
    
    # Read raw data
    arr = np.fromfile(img_path, dtype=dtype)
    
    if arr.size != num_elements:
        raise ValueError(f"File size mismatch. Expected {num_elements} elements, got {arr.size}")
        
    # Reshape: ITK expects [z, y, x] for numpy array access
    # Shape from header is usually [x, y, z]
    # We reshape to [z, y, x]
    arr = arr.reshape(shape[::-1]) 
    
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(spacing)
    image.SetOrigin((0.0, 0.0, 0.0))  # Force origin to 0 for this dataset
    
    return image

def load_image(path):
    """
    Robust image loader that attempts SimpleITK first, then falls back to custom Analyze reader.
    """
    try:
        return sitk.ReadImage(path)
    except Exception as e:
        if path.endswith('.img') or path.endswith('.hdr'):
            try:
                # print(f"SimpleITK failed, attempting custom Analyze 7.5 reader for {path}")
                return read_analyze75_numpy(path)
            except Exception as e2:
                print(f"Custom reader also failed: {e2}")
                raise e
        raise e
