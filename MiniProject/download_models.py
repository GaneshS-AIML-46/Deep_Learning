import os
import requests

MODELS = [
	{
		"filenames": ["colorization_deploy_v2.prototxt"],
		"urls": [
			"https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt",
			"https://raw.githubusercontent.com/PySimpleGUI/PySimpleGUI-Photo-Colorizer/master/model/colorization_deploy_v2.prototxt",
			"https://raw.githubusercontent.com/harishanand95/opencv-colorizer/master/models/colorization_deploy_v2.prototxt",
			"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization/colorization_deploy_v2.prototxt",
		],
	},
	{
		"filenames": ["colorization_release_v2.caffemodel"],
		"urls": [
			"http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
			"https://pjreddie.com/media/files/colorization_release_v2.caffemodel",
			"https://huggingface.co/spaces/Shashank009/Black_and_white_image_colorization/resolve/main/colorization_release_v2.caffemodel",
		],
	},
	{
		"filenames": ["pts_in_hull.npy"],
		"urls": [
			"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/colorization/pts_in_hull.npy",
			"https://raw.githubusercontent.com/PySimpleGUI/PySimpleGUI-Photo-Colorizer/master/model/pts_in_hull.npy",
			"https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy",
			"https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
		],
	},
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CAFFE_MIN_BYTES = 100_000_000  # ~100MB sanity threshold


def download_first_success(urls: list[str], dest: str) -> None:
	last_error = None
	for url in urls:
		try:
			print(f"Trying {url} ...")
			with requests.get(url, stream=True, timeout=120) as r:
				r.raise_for_status()
				with open(dest, "wb") as f:
					for chunk in r.iter_content(chunk_size=8192):
						if chunk:
							f.write(chunk)
			if dest.lower().endswith(".caffemodel") and os.path.getsize(dest) < CAFFE_MIN_BYTES:
				print("Downloaded file seems too small, trying next mirror...")
				try:
					os.remove(dest)
				except OSError:
					pass
				continue
			print(f"Saved {os.path.basename(dest)}")
			return
		except Exception as e:
			last_error = e
			print(f"Failed: {e}")
	raise RuntimeError(f"All mirrors failed for {os.path.basename(dest)}: {last_error}")


def main() -> None:
	os.makedirs(MODELS_DIR, exist_ok=True)

	for item in MODELS:
		for fname in item["filenames"]:
			path = os.path.join(MODELS_DIR, fname)
			# Re-validate existing files; for caffemodel require a minimum size
			if os.path.isfile(path):
				if path.lower().endswith(".caffemodel"):
					if os.path.getsize(path) >= CAFFE_MIN_BYTES:
						print(f"OK {fname} (already present)")
						continue
					else:
						print(f"Existing {fname} is too small, re-downloading...")
				else:
					if os.path.getsize(path) > 0:
						print(f"OK {fname} (already present)")
						continue
			print(f"Downloading {fname} ...")
			download_first_success(item["urls"], path)


if __name__ == "__main__":
	main()
