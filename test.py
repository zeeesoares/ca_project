from torch_ext.checkpoint import Checkpoint

if __name__ == "__main__":
    checkpoint = Checkpoint()

    state = {
        "epoch": 1,
        "model_state_dict": None,
        "optimizer_state_dict": None
    }

    checkpoint_local_path = "/tmp/checkpoint.pt"
    checkpoint_pfs_path = "/tmp/checkpoint_pfs.pt"  # At deucalion we could use /projects dir

    checkpoint.save(state, checkpoint_local_path, checkpoint_pfs_path)
