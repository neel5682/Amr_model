Dynamic Attention Mechanism Using CNN for Automatic Modulation Recognition
Overview
This repository contains an implementation of a Dynamic Context Attention Mechanism (DCA) integrated with a Convolutional Neural Network (CNN) for Automatic Modulation Recognition (AMR). The model is designed to classify different modulation schemes from raw IQ data using deep learning techniques.

Features
✅ Utilizes a Dynamic Context Attention Mechanism (DCA) to enhance feature extraction
✅ Employs CNN layers to capture spatial and frequency domain patterns in IQ signals
✅ Incorporates KANConv1D and KANDense layers for better adaptability and efficiency
✅ Supports RML2016.10a dataset with multiple modulation types
✅ Optimized using data augmentation and hyperparameter tuning

Dataset
The model is trained on the RML2016.10a dataset, which consists of:

Modulation Types: AM-DSB, AM-SSB, BPSK, QPSK, 8PSK, QAM16, QAM64, GFSK, CPFSK, WBFM
Data Format: Complex IQ samples
SNR Range: -20 dB to +18 dB
Model Architecture
The model consists of:

1D CNN Layers: Extract spatial-temporal features from raw IQ data
Dynamic Context Attention (DCA): Enhances relevant feature representations dynamically
KANConv1D & KANDense: Adaptive layers inspired by Kolmogorov–Arnold Networks (KAN)
Fully Connected Layers: Classify modulations using learned embeddings
Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/neel5682/DynamicAttention-AMR.git
cd DynamicAttention-AMR
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Training the Model
python
Copy
Edit
python train.py --dataset RML2016.10a --epochs 50 --batch_size 64
Testing the Model
python
Copy
Edit
python test.py --model_path saved_model.pth --dataset RML2016.10a
Hyperparameter Tuning
Modify config.py to adjust parameters like learning rate, dropout, and architecture settings.

Results
📌 Achieved high accuracy on multiple modulation schemes
📌 Improved robustness in low-SNR conditions compared to baseline CNN models
📌 Better generalization using Dynamic Context Attention

Future Work
🚀 Integrate physics-informed neural networks (PINNs) for improved signal interpretation
🚀 Experiment with transformer-based architectures for enhanced sequential learning
🚀 Deploy the model for real-time SDR applications

Contributing
Feel free to contribute by submitting a pull request or reporting issues!

License
📝 This project is licensed under the MIT License.
