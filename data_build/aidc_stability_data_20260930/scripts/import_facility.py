"""Import an explicitly supplied facility numeric meter export."""
import argparse
import json
from pathlib import Path
from aidc_data.facility import import_meter_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_meter_file(args.source, args.output), indent=2))
