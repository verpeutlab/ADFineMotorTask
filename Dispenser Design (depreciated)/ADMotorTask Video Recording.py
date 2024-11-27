# This runs on Python version 3.7
"""
Created on Tue Aug 22 15:11:50 2023

@author: verpeutlab, davidlukacik
"""

import os
import time
import cv2
from PySpin import PySpin
import serial

# Set the save location for the videos
save_folder = "D:\FineMotorVideos\PySpinTestTemp"

# Set the desired frames per second
fps = 32

# Set the serial port and baud rate
serial_port = "COM3"
baud_rate = 9600

# Specify the path to the ffplay executable
ffplay_path = r"C:\Users\verpeutlab\Documents\ff\ffplay.exe"

# Initialize serial communication
ser = serial.Serial(serial_port, baud_rate)

# Initialize PySpin system
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

if cam_list.GetSize() == 0:
    print("No cameras found.")
    system.ReleaseInstance()
    ser.close()  # Close serial communication
    exit(1)

# Function to initialize and start the camera
def init_and_start_camera():
    try:
        # Select the first camera
        camera = cam_list.GetByIndex(0)

        # Initialize camera
        camera.Init()

        # Configure camera settings
        camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(fps)

        # Start the camera stream
        camera.BeginAcquisition()

        return camera

    except PySpin.SpinnakerException as e:
        print(f"Error initializing camera: {e}")
        return None

# Function to start recording for a specified duration
def start_recording(camera, duration):
    try:
        # Create a VideoWriter object to save the recording
        current_time = time.strftime("%Y%m%d_%H%M%S")
        video_name = os.path.join(save_folder, f"pyspintest_recording_{current_time}.avi")

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(video_name, fourcc, fps, (camera.Width.GetValue(), camera.Height.GetValue()))

        # Record for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # Get the next image from the camera
                image_result = camera.GetNextImage()

                # Convert the image to a numpy array
                image_data = image_result.GetNDArray()

                # Convert the image to BGR format for compatibility with OpenCV
                image_bgr = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)

                # Write the image frame to the video file
                video_writer.write(image_bgr)

                # Release the image
                image_result.Release()

            except PySpin.SpinnakerException as e:
                print(f"Error: {e}")
                break

        # End the camera acquisition
        camera.EndAcquisition()

        # Release the video writer
        video_writer.release()

    except PySpin.SpinnakerException as e:
        print(f"Error: {e}")

# Main loop
while True:
    # Wait for the "cr" signal from Arduino to start the recording
    while ser.read(2) != b"cr":
        pass

    # Initialize and start the camera
    camera = init_and_start_camera()

    if camera is not None:
        # Start recording for 30 seconds
        start_recording(camera, 30)

        # Close the camera
        camera.DeInit()

# Close serial communication
ser.close()

# Release the camera list and system resources
cam_list.Clear()
system.ReleaseInstance()
