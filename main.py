import cv2
from ultralytics import solutions

def count_objects_in_region(video_path,model_path):
    """Count objects in a specific region within a video."""
    count=0
    cap = cv2.VideoCapture(video_path)
    region_points = (225,1),(225,497)
    counter = solutions.ObjectCounter(show=False, region=region_points, model=model_path)

    # Callback function for mouse events
    def mouse_move(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse moved to: ({x}, {y})")

    # Create a window and set the mouse callback
    window_name = "Object Counting"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_move)

    while cap.isOpened():
        success, im0 = cap.read()
        count += 1
        if count % 2 != 0:
           continue
        if not success:
            break
        im0=cv2.resize(im0,(1020,500))
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)

        # Display the frame
        cv2.imshow(window_name, im0)
       

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Call the function
count_objects_in_region("vid1.avi","best.pt")
