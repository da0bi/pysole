"""
Wurtenkees Glacier Example: Running PySole Workflow from Configuration
"""

import numpy as np
import pysole

def main():
    print("--- Running PySole Wurtenkees Glacier Example ---")

    # Execute complete workflow defined in pysole_wuk.json
    config_path = "examples/wuk/pysole_wuk.json"
    bedrock_map = pysole.run_from_config(config_path)

    print("\nWurtenkees example completed successfully!")
    print(f"Final Bedrock Raster Saved: {bedrock_map.name}")
    print(f"Bedrock Elevation Range: [{np.nanmin(bedrock_map.grid):.2f} m, {np.nanmax(bedrock_map.grid):.2f} m]")


if __name__ == "__main__":
    main()
