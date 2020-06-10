import os


def start_components(components):
    for component in components:
        component.start()


def is_real_system():
    return int(os.environ['IS_REAL_SYSTEM']) == 1
