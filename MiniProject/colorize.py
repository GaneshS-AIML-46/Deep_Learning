import os
import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

PROTOTXT = os.path.join(MODELS_DIR, "colorization_deploy_v2.prototxt")
CAFFE_MODEL = os.path.join(MODELS_DIR, "colorization_release_v2.caffemodel")
PTS = os.path.join(MODELS_DIR, "pts_in_hull.npy")


def _ensure_models_exist() -> None:
	missing = [p for p in [PROTOTXT, CAFFE_MODEL, PTS] if not os.path.isfile(p)]
	if missing:
		raise FileNotFoundError(
			"Missing model files. Run 'python download_models.py' first. Missing: "
			+ ", ".join(os.path.basename(m) for m in missing)
		)
	# basic sanity size for caffemodel (~100MB+)
	if os.path.getsize(CAFFE_MODEL) < 100_000_000:
		raise RuntimeError(
			f"Model file seems incomplete: {os.path.basename(CAFFE_MODEL)} is too small. Re-run download."
		)


def _inject_cluster_centers(net: cv2.dnn_Net, pts_npy_path: str) -> None:
	pts = np.load(pts_npy_path)
	pts = pts.transpose().reshape(2, 313, 1, 1).astype(np.float32)
	class8 = net.getLayerId("class8_ab")
	conv8 = net.getLayerId("conv8_313_rh")
	if class8 == -1 or conv8 == -1:
		raise RuntimeError(
			"Expected layers 'class8_ab' and 'conv8_313_rh' not found in the network. "
			"Ensure you have the correct 'colorization_deploy_v2.prototxt'."
		)
	# Try modern API: setParam; fallback to blobs for older OpenCV builds
	try:
		net.setParam(class8, 0, pts)
		net.setParam(conv8, 0, np.full([1, 313], 2.606, dtype=np.float32))
	except Exception:
		layer_class8 = net.getLayer(class8)
		layer_conv8 = net.getLayer(conv8)
		layer_class8.blobs = [pts]
		layer_conv8.blobs = [np.full([1, 313], 2.606, dtype=np.float32)]


def colorize_image(input_path: str, output_path: str) -> None:
	_ensure_models_exist()

	net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)
	_inject_cluster_centers(net, PTS)

	bgr = cv2.imread(input_path)
	if bgr is None:
		raise FileNotFoundError(f"Could not read image: {input_path}")

	img = bgr.astype(np.float32) / 255.0
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	L = lab[:, :, 0]

	L_rs = cv2.resize(L, (224, 224))
	L_rs -= 50

	net.setInput(cv2.dnn.blobFromImage(L_rs))
	ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
	ab_dec_us = cv2.resize(ab_dec, (bgr.shape[1], bgr.shape[0]))

	L = L[:, :, np.newaxis]
	lab_out = np.concatenate((L, ab_dec_us), axis=2)
	bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
	bgr_out = np.clip(bgr_out, 0, 1)
	bgr_out = (255 * bgr_out).astype(np.uint8)

	out_dir = os.path.dirname(output_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	ok = cv2.imwrite(output_path, bgr_out)
	if not ok:
		raise RuntimeError(f"Failed to save output image to: {output_path}")
	print(f"Saved: {os.path.abspath(output_path)}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Colorize a black-and-white image using OpenCV DNN model.")
	parser.add_argument("input", help="Path to grayscale input image")
	parser.add_argument("output", help="Path to write colorized image")
	args = parser.parse_args()

	colorize_image(args.input, args.output)
