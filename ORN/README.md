# ORN-pytorch
Refer to excllent implemention [ORN-master](https://github.com/SDL-GuoZhao/ORN-master).
And fix a bug which makes it not supported for pytorch latest than v1.0.0(BTW, it should have been fixed in earlier version, cause it may lead to false gradient calc.).


### Getting started
1. clone the this branch: 

    ```bash
    # git version must be greater than 1.9.10
    git clone  this-repo
    ```

2. setup orn:

    ```bash
    ./build.sh
    ```

3. run the MNIST-Variants demo:

    ```bash
    # train baseline CNN on MNIST
    python main.py --net cnn --dataset MNIST
    # train ORN on MNIST
    python main.py --net orn --dataset MNIST
    # train baseline CNN on MNIST-rot
    python main.py --net cnn --dataset MNIST-rot
    # train ORN on MNIST-rot
    python main.py --net orn --dataset MNIST-rot
    ```
### Evaluate model

    ```bash
    # test model on MNIST
    python utils/eval_model.py --model-path outputs/path_to_your_ckpt.pth --dataset MNIST
    ```

