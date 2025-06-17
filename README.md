# ✍️ Convolutional Neural Network Application for Converting Handwriting to Digital

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-green.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 About the Project

This project It aims to automatically convert handwritten texts and documents into digital format using Convolutional Neural Networks (CNN). It offers a solution to the need to digitize handwritten notes, documents or forms encountered in daily life quickly and accurately.

Focusing on the problem of handwritten character recognition, the project has developed a special CNN model that can recognize a total of **62 different classes** in the English alphabet, including 26 uppercase letters, 26 lowercase letters and 10 digits. Thanks to the **Handwritten English Characters and Digits** dataset used in the study, the developed model has successfully performed with an impressive overall accuracy rate of 88.6%.

This success once again demonstrates the strong potential of deep learning and especially CNNs in complex computer vision problems such as handwriting recognition.

## ✨ Features

- **Multi-Class Handwriting Recognition:** Ability to recognize 62 different character classes, including 26 uppercase letters, 26 lowercase letters and 10 digits.
- **Customized CNN Architecture:** Convolutional Neural Network model designed to extract deep features from image data and effectively classify characters.
- **Development with TensorFlow and Keras:** Industry-standard libraries were used to quickly and efficiently build and train deep learning models.
- **Data Augmentation:** Data augmentation techniques were applied with `ImageDataGenerator` to increase the generalization ability of the model and reduce overfitting.
- **Model Training and Evaluation:** The performance of the model was analyzed in detail with a comprehensive training cycle and performance metrics (accuracy, loss graphs).

## 🛠️ Technologies Used

- **Python:** The main programming language of the project.
- **TensorFlow:** The main library for building and training deep learning models.
- **Keras:** High-level neural network API running on TensorFlow, providing rapid prototyping.
- **Pandas:** Used for data manipulation and analysis.
- **Seaborn & Matplotlib:** Used for visualization of training process and model performance.
- **Jupyter Notebook:** An environment for project development, coding and interactive presentation of results.

## 📊 Dataset

The dataset used in this project is the **"Handwritten English Characters and Digits"** dataset available on Kaggle.

🔗 **Dataset Link:** [Handwritten English Characters and Digits Dataset on Kaggle](https://www.kaggle.com/datasets/sachinpatel21/handwritten-english-characters-and-digits)

Please download the dataset and place it in the root directory of the project as `data/augmented_images/augmented_images1/` or update the `data_path` variable in the Jupyter Notebook with the location of your own dataset.

## 📈 Model Performance
The developed CNN model has performed successfully with an overall accuracy rate of 88.6% on the training dataset.

Detailed performance metrics and training graphs of the model are included in the Jupyter Notebook.

## 🔮 Future Work
Model Improvements: Increasing model performance with more complex network architectures or transfer learning techniques (e.g. using pre-trained models such as VGG, ResNet).

Real-Time Application: Inference optimizations for integrating the model into a web-based or mobile application.

Different Languages ​​and Handwritings: Adapting the model for handwritings in Turkish or other languages ​​and training it on new datasets.

Larger Datasets: Using larger and more diverse datasets to increase the generalization ability and accuracy of the model.

## 📄 License
This project is licensed under the MIT License.
For more information, please see the LICENSE file.

## 📧 Contact
Gülçin İŞİDOĞRU
📧 Email: gulcinisidogru06@gmail.com

---









