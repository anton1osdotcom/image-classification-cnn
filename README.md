# Image Classification with CNN

This project demonstrates the use of Convolutional Neural Networks (CNN) for image classification using the CIFAR-10 dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/anton1osdotcom/image-classification-cnn.git
    ```
2. Navigate to the directory:
    ```bash
    cd image-classification-cnn
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Training the Model:**
    ```bash
    python3 src/train.py
    ```
2. **Evaluating the Model:**
    ```bash
    python3 src/evaluate.py
    ```
3. **Plotting the Results:**
    ```bash
    python3 src/plot_history.py
    ```

## Results

- The model achieved an accuracy of approximately `70%` on the CIFAR-10 test dataset.
- Confusion matrix and accuracy/loss plots can be found in the `results/` directory.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
