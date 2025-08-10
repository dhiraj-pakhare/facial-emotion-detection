import cv2

# Attempt to access the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to access the camera.")
else:
    print("Camera is ready!")
    cap.release()















# import cv2

# def test_camera():
#     camera_index = 0  # Start with default camera
#     cap = None
    
#     # Try multiple camera indices (0, 1, 2) in case the default one fails
#     for i in range(3):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             camera_index = i
#             print(f"Camera found at index {i}")
#             break
#         cap.release()
    
#     if not cap or not cap.isOpened():
#         print("❌ Failed to access any camera.")
#         return
    
#     # Set resolution (optional)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     print("✅ Camera is ready! Press 'q' to exit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Failed to grab frame.")
#             break

#         cv2.imshow("Camera Test", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Exiting...")
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     test_camera()
