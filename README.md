# Project Omicron

![Project Omicron Logo](logo.svg)

Project Omicron is a plant disease classification workspace built with PyTorch and Flask, supported by additional TensorFlow training utilities. The repository includes an inference web app, a PyTorch training pipeline for multi-crop disease classification, and a TensorFlow model trainer for plant-specific datasets.

## 📌 What’s included

- `app.py` — Flask web server for uploading images and predicting disease labels using a MobileNetV2 model.
- `train_torch.py` — PyTorch training pipeline for datasets organized by crop and disease.
- `train_model.py` — TensorFlow model trainer for plant-specific image classification.
- `classes.json` — Saved class labels used by the inference server.
- `plant_disease_model/` — Existing trained model files and metadata.
- `data/` — Organized dataset folders for `train`, `valid`, and `test`.
- `static/`, `templates/` — Web app front-end assets.
- `uploads/` — Temporary upload folder used by the Flask app.

## 🚀 Quick start

### 1. Install dependencies

```bash
pip install flask torch torchvision pillow
```

If you want to use the TensorFlow pipeline, also install:

```bash
pip install tensorflow
```

### 2. Run the web app

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000
```

### 3. Train the PyTorch model

If you want to retrain the PyTorch model using the repository dataset structure:

```bash
python train_torch.py
```

This will use the directories in `data/train` and `data/valid`, save `plant_disease_model.pth`, and create a fresh `classes.json` file.

### 4. Train the TensorFlow model

The TensorFlow trainer is designed for plant-specific directories, such as `data/train/Tomato` and `data/valid/Tomato`.

```bash
python train_model.py --plant_type Tomato
```

> Check the `train_model.py` file for additional command-line options.

## 🗂️ Expected data layout

The repository expects image data organized like this:

```
data/
  train/
    Apple/
      DiseaseA/
      DiseaseB/
    Tomato/
      DiseaseA/
      DiseaseB/
  valid/
    Apple/
    Tomato/
  test/
```

For the PyTorch pipeline, the `train_torch.py` script supports a multi-crop structure where each disease label is nested under its crop folder.

## 💡 Notes

- The Flask app uses `plant_disease_model.pth` and `classes.json` for inference.
- Make sure your upload folder exists and is writable. The app already creates `uploads/` if missing.
- The current web server runs on CPU by default.

## 🧩 File summary

- `app.py` — inference and REST API
- `train_torch.py` — multi-class PyTorch trainer
- `train_model.py` — TensorFlow classifier trainer
- `templates/index.html` — web UI
- `static/css/` — styles
- `static/js/` — JavaScript for front-end interactions

## 📄 License

This project is released under the MIT License.
