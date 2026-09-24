"""
Goldbergkees Glacier (GOK) Example: Running PySole Workflow from Configuration
"""

import numpy as np
import pysole


def main():
    print("--- Running PySole Goldbergkees Glacier Example ---")

    # Execute complete workflow defined in pysole_gok.json
    config_path = "examples/gok/pysole_gok.json"
    bedrock_map = pysole.run_from_config(config_path)

    print("\nGoldbergkees example completed successfully!")
    print(f"Final Bedrock Raster Saved: {bedrock_map.name}")
    print(f"Bedrock Elevation Range: [{np.nanmin(bedrock_map.grid):.2f} m, {np.nanmax(bedrock_map.grid):.2f} m]")


if __name__ == "__main__":
    main()
