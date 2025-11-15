#🐾 Animal Classification Using Deep Learning

This project classifies animal images using a ResNet18 model trained with PyTorch.

✨ Features

Train a custom classification model

Predict a single image

Predict a full folder of images (batch mode)

Optional Streamlit web app for GUI-based classification

Note:
The model file animal_classifier.ckpt is not included due to its size.
You can train your own using train.py or download it from your own link.

📁 Project Structure
animal_classification_project/

│
├── train.py  # Train the model

├── inference.py   # Predict one image

├── batch_infer.py   # Predict multiple images (folder)

├── app.py        # Streamlit web app (optional)

├── requirements.txt    # Python dependencies

└── sample_images/      # Example images (optional)


🔧 Installation
1. Create virtual environment
python -m venv venv

2. Activate it

Windows:

venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

🏋️‍♂️ Training the Model

Make sure your dataset is inside a folder named dataset with subfolders per class.

Run:

python train.py


This generates:

animal_classifier.ckpt

🔍 Inference (Single Image)

Run prediction on one image:

python inference.py --image sample_images/lion.jpg --topk 5

📦 Batch Inference (All Images in a Folder)

Predict all images in a directory and save results:

python batch_infer.py --folder sample_images --topk 3 --out predictions.csv

🌐 Optional: Streamlit App

Run the web app:

streamlit run app.py


Upload an image and view the model’s prediction.

📥 Model File (Not Included)

The CKPT file is not stored in this repo because it is large.

You can either:

Train your own model using train.py, or

Download your trained model from your Google Drive link (add your link here)

📸 Example Output
Predictions for: Lion_1_1.jpg
1. Lion — 97.24%
2. Tiger — 1.61%
3. Deer — 0.37%
4. Bear — 0.19%
5. Elephant — 0.13%

🤝 Contribution

Feel free to fork this repo and improve the project.

📄 License

This project is for educational and research purposes.



