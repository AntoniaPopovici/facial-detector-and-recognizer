# Facial Detection and Recognition System

This project implements facial detection and recognition algorithms using the Histogram of Oriented Gradients (HOG) method along with a sliding window approach. The system is divided into two tasks: facial detection and facial recognition.

## Task 1 - Facial Detection

For the first task, facial detection is implemented using the sliding window technique and HOG descriptors. Positive descriptors are generated from annotated training images, while negative descriptors are generated from random images. Additional negative images with skin tone presence are also included. Various data augmentation techniques are applied to increase the dataset size.

### Implementation Details

- Implemented using Python and OpenCV.
- Positive descriptors: 41862
- Negative descriptors: 99426
- Descriptor features: 2916
- Model training: Linear Support Vector Classifier (LinearSVC)
- Average Precision: ~50%

### Running Task 1

To run Task 1, execute the following files:

```
imports.py
params.py
utils.py
evals.py
generator.py
FacialDetector.py
task1.py
```

### Additional Information

For more detailed information, refer to the "documentation.pdf" provided in the repository.

## Task 2 - Facial Recognition

The second task focuses on facial recognition, specifically recognizing the face of a specific individual (Barney). Positive images contain Barney's face, while negative images contain other faces and generated images from Task 1.

### Implementation Details

- Similar approach as Task 1 but with Barney's face as the target.
- Model trained using Linear Support Vector Classifier (LinearSVC).
- Average Precision: 50.8%

### Running Task 2

To run Task 2, execute the following files:

```
imports.py
params.py
utils.py
evals.py
generator.py
FacialRecognizer.py
task2.py
```

### Additional Information

For more detailed information, refer to the "documentation.pdf" provided in the repository.

## Dependencies

- OpenCV 4.8.1
- NumPy 1.26.2
- TensorFlow 2.15.0
- scikit-learn (sklearn)
- PIL (Python Imaging Library)

## Note

To avoid regenerating images and descriptors, comment out the image and descriptor generation sections in the respective task files.
