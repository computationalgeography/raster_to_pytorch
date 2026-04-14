#!/usr/bin/env python3

#import sys
#from lib.crop import crop_northern_hemisphere
#
#
#def main():
#    if len(sys.argv) < 4:
#        print("Usage:")
#        print("python scripts/crop_nh.py <input.tif> <output.tif> [compression]")
#        sys.exit(1)
#
#    input_file = sys.argv[1]
#    output_file = sys.argv[2]
#
#    # Optional compression argument
#    compress = sys.argv[3] if len(sys.argv) > 3 else "LZW"
#    
#    output_type=gdal.GDT_Float32
#
#    crop_northern_hemisphere(
#        in_file=input_file,
#        out_file=output_file,
#        compress=compress
#    )
#
#    print("Cropping completed successfully.")
#
#
#if __name__ == "__main__":
#    main()
#    
    
#!/usr/bin/env python3
import sys
from osgeo import gdal
from lib.crop import crop_northern_hemisphere

# Map string names to GDAL types
GDAL_TYPE_MAP = {
    "Byte": gdal.GDT_Byte,
    "UInt16": gdal.GDT_UInt16,
    "Int16": gdal.GDT_Int16,
    "UInt32": gdal.GDT_UInt32,
    "Int32": gdal.GDT_Int32,
    "Float32": gdal.GDT_Float32,
    "Float64": gdal.GDT_Float64,
}


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python crop_nh.py <input.tif> <output.tif> [optional: compression method] [optional: output_type]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Default compression
    compress = "ZSTD"

    # Default output_type = None (use input raster type)
    output_type = None
    

    # Optional compression argument
    if len(sys.argv) > 3:
        arg3 = sys.argv[3]
        if arg3.upper() in ["LZW", "ZSTD", "DEFLATE", "PACKBITS"]:
            compress = arg3
        elif arg3 in GDAL_TYPE_MAP:
            output_type = GDAL_TYPE_MAP[arg3]

    # Optional output_type argument
    if len(sys.argv) > 4:
        arg4 = sys.argv[4]
        if arg4 not in GDAL_TYPE_MAP:
            print(f"Unsupported output type: {arg4}")
            sys.exit(1)
        output_type = GDAL_TYPE_MAP[arg4]

    # Call the crop function
    crop_northern_hemisphere(
        input_file,
        output_file,
        compress=compress,
        output_type=output_type
    )

    print("Cropping completed successfully.")


if __name__ == "__main__":
    main()