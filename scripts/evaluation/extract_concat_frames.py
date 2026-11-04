import cv2
import sys
import os


def extract_and_concat(video_path, output_path):
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 3:
        print("Error: Video too short.")
        return

    # 프레임 인덱스
    first_idx = 0
    middle_idx = total_frames // 2
    last_idx = total_frames - 1

    frames = []

    for idx in [first_idx, middle_idx, last_idx]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Cannot read frame {idx}")
            return

        frames.append(frame)

    cap.release()

    # 높이 맞추기 (혹시 다를 경우 대비)
    min_height = min(f.shape[0] for f in frames)
    resized_frames = [
        cv2.resize(f, (int(f.shape[1] * min_height / f.shape[0]), min_height))
        for f in frames
    ]

    # 가로로 붙이기
    concat_img = cv2.hconcat(resized_frames)

    # 저장
    cv2.imwrite(output_path, concat_img)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_concat_frames.py input.mp4 output.jpg")
        sys.exit(1)

    input_video = sys.argv[1]
    output_image = sys.argv[2]

    extract_and_concat(input_video, output_image)