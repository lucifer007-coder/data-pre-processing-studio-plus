import multiprocessing

RANDOM_STATE = 42
PREVIEW_ROWS = 1000
MAX_WORKERS = min(4, multiprocessing.cpu_count())  # Dynamic thread pool size, capped at 4
