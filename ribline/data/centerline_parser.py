import csv
import collections
import numpy as np

def parse_rib_xls(filepath):
    """
    Parses the custom tab-separated .xls ground truth files for rib centerlines.
    
    Args:
        filepath (str): Path to the .xls file.
        
    Returns:
        dict: Mapping from RibName (e.g., 'R01') to numpy array of coordinates (N, 3).
    """
    ribs = collections.defaultdict(list)
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row: continue
                rib_name = row[0]
                # x, y, z are indices 1, 2, 3
                # We assume the file format is consistent: Name, X, Y, Z, ...
                try:
                    coords = [float(x) for x in row[1:4]]
                    ribs[rib_name].append(coords)
                except ValueError:
                    # Skip lines that might contain headers or malformed data
                    continue
                    
        # Convert lists to numpy arrays
        rib_dict = {}
        for r, pts in ribs.items():
            rib_dict[r] = np.array(pts)
            
        return rib_dict
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return {}
