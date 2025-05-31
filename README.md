# ASLALphabetDetector
An application for American Sign Language classification built in Python. It includes the training two custom modes using Tensorflow, one for hand detection and the other for gesture classification. The models were trained on a custom dataset created with the included scripts.

## Requirements
- Python 3.9+
- pip

## Usage
Clone the repositroy with:
```bash
git clone https://github.com/Ksagit/ASLAlphabetDetector.git
```

Install required pacakges with:
```bash
pip install -r requirements.txt
```

### Hand detection 
Navigate to the appropriate directory:
```bash
cd hand
```

Create the dataset with:
```bash
python3 gathering.py
```

Extend the dataset with:
```bash
python3 processing.py
```

Train the model with:
```bash
python3 trainig.py
```

Run the detector with:
```bash
python3 detector.py
```

### Gesture classification
Create the dataset with:
```bash
python3 gathering.py
```

Extend the dataset with:
```bash
python3 processing.py
```

Train the model with:
```bash
python3 trainig.py
```
