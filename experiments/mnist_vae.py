if __name__ == "__main__":
    import jaynes
    from mnist_classification.thunk import thunk
    from ml_logger import logger
    from mnist_classification.vae import train

    # jaynes.config(mode="local")
    jaynes.config(runner=dict(n_gpu=1, n_cpu=4))
    jaynes.run(thunk(train,
                     log_dir="http://54.71.92.65:8081",
                     log_prefix=f"debug/mnist_vae/vae/{logger.now('%f')}"),
               )
    jaynes.listen()
