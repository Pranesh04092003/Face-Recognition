import face_recognition
import pickle
import os

# Directory containing subdirectories of images for each person
dataset_dir = "dataset"

encodings = []
names = []

# Loop through each person in the dataset
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    
    # Loop through each image of the person
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        image_encodings = face_recognition.face_encodings(image)
        
        if image_encodings:
            # Add each encoding to the list
            for encoding in image_encodings:
                encodings.append(encoding)
                names.append(person_name)
            print(f"Success: Encoded {image_name} for {person_name}")
        else:
            print(f"Failed: No face found in {image_name} for {person_name}")

# Save encodings and names
with open("face_encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)