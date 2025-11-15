Animal Classification Using Deep Learning

This project classifies animal images using a ResNet18 model trained with PyTorch.
It supports:

Training a custom model

Predicting a single image

Predicting an entire folder of images (batch mode)

Optional Streamlit app for GUI-based classification

Note:
The trained model file (animal_classifier.ckpt) is not included in this repository due to file size.
You can train your own model or download the model from your own link (if you upload it to Google Drive).

📁 Project Structure
animal_classification_project/
│
├── train.py               # Train the model
├── inference.py           # Predict one image
├── batch_infer.py         # Predict multiple images (folder)
├── app.py                 # Streamlit web app (optional)
├── requirements.txt       # Python dependencies
└── sample_images/         # Example test images (optional)

🔧 Installation

Create a virtual environment:

python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

🏋️‍♂️ Training the Model

To train the model using your dataset:

python train.py


After training, a file named:

animal_classifier.ckpt


will be created in the project folder.

🔍 Inference (Single Image)

Run prediction on one image:

python inference.py --image path_to_image.jpg --topk 5


Example:

python inference.py --image sample_images/lion.jpg

📦 Batch Inference (Folder of Images)

Predict all images in a directory and save results to a CSV file:

python batch_infer.py --folder sample_images --topk 3 --out predictions.csv

🌐 Optional: Streamlit App

Launch a simple UI for uploading images and viewing predictions:

streamlit run app.py

📥 Model File (Not Included)

The file animal_classifier.ckpt is not included in this repository due to size limits.

You can:

Train your own using train.py, or

Download the model from your Google Drive link and place it in the project folder.

(If you want, add your own download link here.)

📸 Example Output
Predictions for: Lion_1_1.jpg
1. Lion — 97.24%
2. Tiger — 1.61%
3. Deer — 0.37%
4. Bear — 0.19%
5. Elephant — 0.13%

🤝 Contributions

Feel free to modify, improve, or extend this project.

📄 License

This project is for educational purposes.