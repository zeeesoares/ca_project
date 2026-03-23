import os


def list_checkpoints(dir_path):
    return sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.startswith("checkpoint_")
    ])
