import cv2

for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Camera {idx}: not opened")
        continue

    ret, frame = cap.read()
    if ret:
        print(f"Camera {idx}: opened, shape={frame.shape}")
        cv2.imshow(f"Camera {idx}", frame)
        print(f"正在显示 Camera {idx}，按任意键看下一个")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Camera {idx}: opened but no frame")

    cap.release()