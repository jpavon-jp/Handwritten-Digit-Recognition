# 🧠 Handwritten Digit Recognition (CNN & MNIST)

A Deep Learning project that implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0-9) using the MNIST dataset. The model achieves a remarkable **99.24% accuracy** on the test set, leveraging advanced computer vision techniques and regularization strategies.

📄 **[Read the Final Project Report](./SS25___MLSS_A___Josue_Pavon___Handwritten_Digit_Recognition___Final_Project.pdf)**

---

## 📖 Project Overview
The goal was to build a robust image classifier capable of recognizing handwritten digits with high precision. Unlike traditional machine learning models, this project utilizes a **Deep Learning** architecture (CNN) to automatically extract spatial hierarchies of features from the images.

**Key Achievements:**
* [cite_start]Achieved **99.24% Accuracy** on unseen test data.
* [cite_start]Minimized Test Loss to **0.0235**.
* [cite_start]Implemented **Dropout (0.5)** to effectively prevent overfitting[cite: 50].

---

## ⚙️ Model Architecture

The model follows a Sequential architecture optimized for image classification:

1.  [cite_start]**Conv2D (32 filters):** Feature extraction with ReLU activation[cite: 49].
2.  [cite_start]**MaxPooling2D (2x2):** Dimensionality reduction[cite: 50].
3.  [cite_start]**Conv2D (64 filters):** Deeper feature extraction[cite: 50].
4.  [cite_start]**MaxPooling2D (2x2):** Further reduction[cite: 50].
5.  [cite_start]**Flatten:** Converting 2D maps to 1D vectors[cite: 50].
6.  [cite_start]**Dropout (0.5):** Regularization to prevent overfitting[cite: 50].
7.  [cite_start]**Dense (10 units):** Output layer with **Softmax** activation for classification[cite: 51].

* [cite_start]**Total Parameters:** 34,826[cite: 140].
* [cite_start]**Optimizer:** Adam[cite: 57].
* [cite_start]**Loss Function:** Categorical Crossentropy[cite: 58].

---

## 📊 Performance & Evaluation

[cite_start]The model was trained for 15 epochs with a batch size of 128[cite: 59].

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **99.24%** |
| **Test Loss** | 0.0235 |
| **Precision** | ~0.99 (Avg) |
| **Recall** | ~0.99 (Avg) |
| **F1-Score** | ~0.99 (Avg) |

*(See the full Confusion Matrix and Classification Report in the notebook)*

---

## 🛠️ Technical Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jpavon-jp/Handwritten-Digit-Recognition-CNN-99Acc.git](https://github.com/jpavon-jp/Handwritten-Digit-Recognition-CNN-99Acc.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn
    ```
3.  **Run the Notebook:**
    Open the `.ipynb` file in Jupyter Notebook or Google Colab and run all cells to see the training process and predictions.

---

## ⚠️ Context
Developed as the Final Project for the **Machine Learning Systems & Applications (MLSS)** course (SS25) at the University of Europe for Applied Sciences.
**Author:** Josue David Pavon Maldonado
