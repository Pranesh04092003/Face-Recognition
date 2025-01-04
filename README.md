# Real-time Face Recognition System

A real-time face recognition system built with Python, Flask, and face_recognition library. This system can detect and recognize faces from your computer's webcam in real-time.

## Features

- Real-time face detection and recognition
- Web-based interface
- Support for multiple faces
- Easy to add new faces to the dataset
- Live display of recognized names

## Setup Instructions

### Prerequisites
- Docker
- Docker Compose

### Running the Application with Docker

1. **Build the Docker image:**
   ```bash
   docker-compose build
   ```

2. **Run the application:**
   ```bash
   docker-compose up
   ```

3. **Access the application:**
   - Open your web browser and navigate to `http://localhost:5000`

### Troubleshooting

- **Docker not found:** Ensure Docker is installed and running on your system. You can download it from [docker.com](https://www.docker.com/get-started).

## Setup

1. Create your dataset:
   - Create a folder for each person inside the `dataset` directory
   - Name each folder with the person's name (e.g., "john", "sarah")
   - Add multiple clear face photos of each person in their respective folders
   - Supported formats: jpg, jpeg, png

Example:
```
dataset/
    john/
        photo1.jpg
        photo2.jpg
    sarah/
        photo1.jpg
        photo2.jpg
```

2. Generate face encodings:
```bash
docker-compose run face_recognizer python encode_dataset.py
```
This will create a `face_encodings.pkl` file containing the encoded face data.

## License
[Your chosen license]

## Acknowledgments
- face_recognition library
- Flask
- OpenCV

