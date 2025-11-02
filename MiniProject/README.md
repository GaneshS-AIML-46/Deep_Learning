## Black & White Image Colorization (OpenCV DNN)

This project is to  colorize grayscale images using OpenCV's DNN module and the Berkeley colorization model.

### 1) Create and activate a virtual environment (Windows PowerShell)
```powershell
cd "C:\Users\GANESH\Desktop\DL"
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
pip install -r requirements.txt
```

If you encounter build issues with `opencv-python` on Windows, install the prebuilt wheel first:
```powershell
pip install --upgrade pip
pip install --only-binary opencv-python opencv-python==4.10.0.84
```

### 3) Download model files
```powershell
python download_models.py
```
This fetches the following into the `models` directory:
- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

### 4A) CLI colorization
```powershell
mkdir out
python colorize.py path\to\grayscale.jpg out\colorized.jpg
```

### 4B) Web frontend (Flask)
```powershell
set FLASK_APP=app.py
python app.py
```
Open `http://127.0.0.1:5000` in your browser. Upload an image and view/download the result.

Notes:
- Input should be a grayscale (or B/W) photo; color images are supported but results vary.
- The first run may take a few seconds while OpenCV loads the network.

### Troubleshooting
- If `download_models.py` times out, re-run. Corporate proxies may block the `.caffemodel` URL.
- If files are missing, ensure they exist in `models` and re-run.
- If the app errors about model size, re-run `python download_models.py` to fetch a full `.caffemodel`.
