# PetVision 🐾 — AI Dog & Cat Breed Classifier

PetVision is a modern, responsive web application that identifies the breed of a dog or cat from an uploaded photo. It uses a fine-tuned PyTorch ResNet50 neural network (132 classes) and provides explainable AI insights via Grad-CAM activation heatmaps.

---

## ✨ Features

- **Accurate Breed Classification**: Evaluates uploaded pet photos against 132 dog and cat breeds trained on Stanford Dogs and Oxford-IIIT Pet datasets.
- **Top-3 Predictions & Confidence Scores**: Displays the primary match with large typography alongside animated Top-3 probability bars.
- **Explainable AI (Grad-CAM)**: Visualizes the exact convolutional features (`layer4`) that influenced the model's decision using blended heatmaps with multiple view modes (Overlay, Side-by-Side, Original).
- **FastAPI Backend**: Asynchronous Python API with strict file validation (JPG/PNG/WebP, max 5 MB).
- **Modern Responsive UI**: Clean, mobile-friendly interface built with Tailwind CSS, Inter font, smooth animations, drag-and-drop file upload, and low-confidence notices (<40%).
- **CPU Optimized**: Fast, local CPU inference without requiring an external GPU.

---

## 📁 Project Structure

```text
petvision/
├── main.py                     # FastAPI application & /predict endpoint
├── model_utils.py              # PyTorch ResNet50 loader, inference & Grad-CAM pipeline
├── generate_sample_model.py    # Utility script to generate testing weights if needed
├── requirements.txt            # Python dependencies
├── README.md                   # Windows setup & deployment guide
├── model/
│   ├── pet_model.pt            # Your PyTorch ResNet50 state_dict (132 classes)
│   └── class_names.json        # JSON array of 132 class names (dog_<Breed>, cat_<Breed>)
└── static/
    └── index.html              # Modern single-page UI (Tailwind CSS CDN + Vanilla JS)
```

---

## 🚀 Getting Started on Windows 11

### 1. Prerequisites
- **Python 3.10+** installed on Windows 11.
- Command Prompt (`cmd.exe`) or Windows PowerShell.

### 2. Create and Activate a Virtual Environment

Open PowerShell or Command Prompt in the `petvision` directory:

```powershell

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment on Windows
venv\Scripts\activate
```

> **Note for PowerShell Users:** If script execution is restricted, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 🧠 Placing Your Model Files

Place your trained model files directly inside the `model/` folder:

1. **`model/pet_model.pt`**:
   - PyTorch `state_dict` of a ResNet50 where the final `fc` layer was replaced with 132 outputs:
     `model.fc = nn.Linear(model.fc.in_features, 132)`.
2. **`model/class_names.json`**:
   - A JSON array of 132 class strings formatted with prefixes, for example:
     ```json
     [
       "cat_Abyssinian",
       "cat_Persian",
       "dog_German_Shepherd",
       "dog_Golden_Retriever",
       ...
     ]
     ```

> **Testing Out-of-the-Box:**
> If you want to test the full website UI, file validation, and Grad-CAM visualization before copying over your final weights, run:
> ```powershell
> python generate_sample_model.py
> ```
> This creates a matching initialized `model/pet_model.pt` so the app is immediately runnable. When your trained weights are ready, simply overwrite `model/pet_model.pt`.

---

## 🏃 Running the Application

Start the local FastAPI development server:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Once started, open your web browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

- Interactive API documentation (Swagger UI) is available at: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Health check endpoint: **[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)**

---

## 🌐 Exposing Publicly with ngrok (for Demos)

To share PetVision with colleagues or test from a mobile phone without deploying to the cloud, use **ngrok**:

1. **Download and Install ngrok** (if not already installed):
   - Download the Windows executable from [https://ngrok.com/download](https://ngrok.com/download) or install via winget / choco:
     ```powershell
     winget install ngrok
     ```
2. **Connect Your ngrok Account**:
   ```powershell
   ngrok config add-authtoken YOUR_AUTHTOKEN
   ```
3. **Start the Tunnel**:
   With your FastAPI server running on port `8000`, open a second terminal and run:
   ```powershell
   ngrok http 8000
   ```
4. **Access the Public URL**:
   ngrok will display a forwarding URL such as:
   `https://xxxx-xx-xx-xx.ngrok-free.app`
   Share this URL or open it on any mobile device to demo PetVision live.

---

## 📡 API Specification

### `POST /predict`
Uploads a pet photo as `multipart/form-data`.

- **Field name**: `file`
- **Supported types**: `.jpg`, `.jpeg`, `.png`, `.webp`
- **Max file size**: 5 MB

#### Sample Response:
```json
{
  "animal": "Dog",
  "top3": [
    {
      "breed": "German Shepherd",
      "confidence": 0.9124
    },
    {
      "breed": "Belgian Malinois",
      "confidence": 0.0541
    },
    {
      "breed": "Doberman",
      "confidence": 0.0182
    }
  ],
  "gradcam": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

#### Error Handling:
- File size exceeds 5 MB: `HTTP 400 Bad Request` with `{"detail": "File size exceeds the 5 MB limit..."}`
- Invalid format or corrupted image: `HTTP 400 Bad Request` with descriptive error.
