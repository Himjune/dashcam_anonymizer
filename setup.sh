pip install ultralytics
pip install pybboxes
pip install opencv-python
conda install opencv=4.9.0.80
pip uninstall numpy
pip install numpy==1.26.4
pip install natsorted
pip install rich
pip install gdown
mkdir model
echo "Downloading the YOLO model..."
gdown 1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe
mv best.pt model/
echo "Setup complete!"
