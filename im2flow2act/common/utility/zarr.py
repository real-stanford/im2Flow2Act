import concurrent.futures

import numpy as np
import zarr


def parallel_saving(
    group: zarr.hierarchy.Group,
    array_name: str,
    shape: tuple,
    chunks: tuple,
    dtype: type,
    overwrite: bool,
    arr_to_save: type,
    max_workers=96,
    **kwargs,
):
    new_array = group.create_dataset(
        array_name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=overwrite,
        **kwargs,
    )

    def copy(zarr_arr, zarr_idx, np_array, np_idx):
        try:
            zarr_arr[zarr_idx] = np_array[np_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            print(e)
            return False

    n = shape[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(n):
            futures.add(
                executor.submit(
                    copy,
                    new_array,
                    i,
                    arr_to_save,
                    i,
                )
            )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")


def parallel_reading(
    group: zarr.hierarchy.Group,
    array_name: str,
    max_workers=48,
):
    zarr_arr = group[array_name]
    zarr_arr_shape = zarr_arr.shape

    # create empty numpy array
    np_array = np.empty(zarr_arr_shape, dtype=zarr_arr.dtype)

    def copy(zarr_arr, zarr_idx, np_array, np_idx):
        try:
            np_array[np_idx] = zarr_arr[zarr_idx]
            # make sure we can successfully read
            _ = np_array[zarr_idx]
            return True
        except Exception as e:
            print(e)
            return False

    n = zarr_arr_shape[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(n):
            futures.add(
                executor.submit(
                    copy,
                    zarr_arr,
                    i,
                    np_array,
                    i,
                )
            )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    return np_array
