import os
import re
from datetime import datetime, timedelta


def find_missing_dates(folder_path, output_path):
    pattern = re.compile(r"(\d{2})_(\d{2})_(\d{4})")
    dates = []

    for fname in os.listdir(folder_path):
        match = pattern.search(fname)
        if match:
            day, month, year = map(int, match.groups())
            dates.append(datetime(year, month, day))

    if not dates:
        print("No valid dates found in filenames.")
        return []

    dates = sorted(dates)

    start = dates[0]
    end = dates[-1]

    all_dates = set(start + timedelta(days=i) for i in range((end - start).days + 1))
    missing = sorted(all_dates - set(dates))

    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "missing_dates.txt")

    with open(output_file, "w") as f:
        for d in missing:
            f.write(d.strftime("%d_%m_%Y\n"))

    print(f"{len(missing)} missing dates saved to {output_file}")

    return missing
