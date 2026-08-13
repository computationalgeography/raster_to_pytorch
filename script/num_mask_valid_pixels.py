import numpy as np
import rasterio


def create_balanced_col_blocks(mask_file, n_tasks):
    """
    Create column blocks with approximately equal number of valid mask pixels.

    Parameters
    ----------
    mask_file : str
        Path to mask raster
    n_tasks : int
        Number of parallel tasks

    Returns
    -------
    col_blocks : list of tuples
        [(start_col, end_col), ...]
    """

    # Read mask
    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    H, W = mask.shape

    # Workload per column
    col_load = np.sum(mask == 1, axis=0)

    # cumulative workload
    cumulative = np.cumsum(col_load)

    total_load = cumulative[-1]
    target_load = total_load / n_tasks

    col_blocks = []

    start = 0

    for i in range(n_tasks):

        if i == n_tasks - 1:
            end = W
        else:
            end = np.searchsorted(
                cumulative,
                (i + 1) * target_load
            )

        col_blocks.append((start, end))
        start = end

    return col_blocks


def main():

    mask_file = "/projects/prjs1930/pilot_data/inference_mask.tif"

    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    H, W = mask.shape

    n_tasks = 16      #   8    # 16

    # ---------------------------------------
    # Uniform column splitting
    # ---------------------------------------
    col_blocks = np.array_split(np.arange(W), n_tasks)

    valid_pixels_per_task = []

    for i, cols in enumerate(col_blocks):

        n_valid = np.sum(mask[:, cols] == 1)

        valid_pixels_per_task.append(n_valid)

        print(
            f"Uniform Task {i}: columns {cols[0]}-{cols[-1]}, "
            f"valid pixels = {n_valid}"
        )

    max_task = np.argmax(valid_pixels_per_task)

    print(
        "\nUniform splitting:",
        "Maximum workload task:",
        max_task,
        "with valid pixels:",
        valid_pixels_per_task[max_task]
    )

    print(
        "Load imbalance ratio:",
        max(valid_pixels_per_task) / min(valid_pixels_per_task)
    )


    # ---------------------------------------
    # Mask-balanced column splitting
    # ---------------------------------------
    balanced_blocks = create_balanced_col_blocks(
        mask_file,
        n_tasks
    )

    print("\nBalanced splitting:")

    balanced_pixels = []

    for i, (start, end) in enumerate(balanced_blocks):

        n_valid = np.sum(mask[:, start:end] == 1)

        balanced_pixels.append(n_valid)

        print(
            f"Balanced Task {i}: columns {start}-{end-1}, "
            f"valid pixels = {n_valid}"
        )

    print(
        "\nBalanced load imbalance ratio:",
        max(balanced_pixels) / min(balanced_pixels)
    )


if __name__ == "__main__":
    main()