
import os
import psutil

def get_available_memory():
    # Get the available memory in bytes
    available_memory = psutil.virtual_memory().available
    # Convert to gigabytes
    available_memory_gb = available_memory / (1024 ** 3)
    return available_memory_gb

def print_memory_info():
    print("RAM usage (%):", psutil.virtual_memory().percent)
    print("RAM usage (total):", psutil.virtual_memory().total)
    print("RAM usage (available):", psutil.virtual_memory().available)
    print("RAM usage (used):", psutil.virtual_memory().used)


def check_active_conda_env():
    active_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    if active_env is not None:
        print(f"The active Conda environment is: {active_env}")
    else:
        print("No active Conda environment found.")


if __name__ == "__main__":

    # Call the function
    print('check_active_conda_env()')
    check_active_conda_env()
    print('\n')
    print('print_memory_info()')
    print_memory_info()
    print('\n')
    print('get_available_memory()')
    available_mem = get_available_memory()
    print("Available memory (GB):", available_mem)
