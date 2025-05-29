import requests
import uuid
import os
import re

# Server URL
url = "http://localhost:8001/process_image"

# UUID for the request
test_uuid = str(uuid.uuid4())

# Path to local image file
image_path = "./image/test1.jpg"

# Check if image file exists
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found")
    exit(1)

# Prepare the request
with open(image_path, "rb") as image_file:
    files = {
        "image": ("test1.jpg", image_file, "image/jpeg"),
        "uuid": (None, test_uuid)
    }
    
    # Send POST request
    response = requests.post(url, files=files)
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {response.headers}")
    
    # Check response
    if response.status_code == 200:
        print("Request successful")
        
        # Extract boundary from content-type
        content_type = response.headers.get("content-type", "")
        match = re.search(r'boundary=(.+)', content_type)
        if not match:
            print("Error: Could not find boundary in Content-Type header")
            exit(1)
        boundary = match.group(1).encode('utf-8')
        print(f"Boundary: {boundary.decode('utf-8')}")
        
        # Split the response content by boundary
        parts = response.content.split(b'--' + boundary)
        expected_parts = [f"{test_uuid}_mask.png", f"{test_uuid}_overlay.png", 
                         f"{test_uuid}_transparent.png", f"{test_uuid}_boundary.png"]
        saved_files = []
        
        for part in parts[1:-1]:  # Skip first (empty) and last (--boundary--)
            part = part.strip()
            if not part:
                continue
                
            # Split headers and body
            try:
                headers, body = part.split(b'\r\n\r\n', 1)
                headers = headers.decode('utf-8')
            except ValueError:
                print(f"Skipping invalid part: {part[:50]}...")
                continue
                
            # Extract filename from Content-Disposition
            filename_match = re.search(r'filename="([^"]+)"', headers)
            if filename_match:
                filename = filename_match.group(1)
                print(f"Found filename: {filename}")
                
                if filename in expected_parts:
                    # Save the image content
                    with open(filename, "wb") as f:
                        # Remove trailing \r\n if present
                        body = body.rstrip(b'\r\n')
                        f.write(body)
                    saved_files.append(filename)
                    print(f"Saved {filename}")
                else:
                    print(f"Skipping unexpected filename: {filename}")
            else:
                print("No filename found in part headers")
        
        # Verify all expected files were saved
        missing_files = [f for f in expected_parts if f not in saved_files]
        if not missing_files:
            print("All expected image files saved successfully")
        else:
            print(f"Error: Missing files {missing_files}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        try:
            print(f"Error message: {response.json()}")
        except:
            print("Error message: Could not decode response as JSON")

# Clean up temporary files
for file in os.listdir():
    if file.startswith("temp_") and file.endswith(".jpg"):
        os.unlink(file)
        print(f"Cleaned up temporary file: {file}")