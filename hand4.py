import cv2
import mediapipe as mp
import numpy as np
import requests

class HandGestureImageSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Ссылки на картинки
        self.image_urls = {
            "absolute_cinema": "G-mT1ZJbQAEtMIC.jpg",
            "one_finger": "http://boro.da33.ru/22145/",

            # Левая рука
            "left_1": "",
            "left_2": "",
            "left_3": "",
            "left_4": "",
            "left_5": "",

            # Правая рука
            "right_1": "",
            "right_2": "",
            "right_3": "",
            "right_4": "",
            "right_5": "",
        }

        self.loaded_images = {}
        self.load_images()

    def load_images(self):
        for key, path in self.image_urls.items():
            if not path:
                continue

            try:
                if path.startswith("http"):
                    response = requests.get(path, timeout=5)
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                else:
                    img = cv2.imread(path)

                if img is not None:
                    self.loaded_images[key] = img
                else:
                    print(f"Не удалось загрузить: {key}")

            except Exception as e:
                print(f"Ошибка загрузки {key}: {e}")

    def count_fingers(self, hand_landmarks, hand_label):
        tips = [4, 8, 12, 16, 20]
        fingers = []

        # Большой палец
        if hand_label == "Right":
            fingers.append(hand_landmarks.landmark[tips[0]].x <
                           hand_landmarks.landmark[tips[0]-1].x)
        else:
            fingers.append(hand_landmarks.landmark[tips[0]].x >
                           hand_landmarks.landmark[tips[0]-1].x)

        # Остальные пальцы
        for tip in tips[1:]:
            fingers.append(
                hand_landmarks.landmark[tip].y <
                hand_landmarks.landmark[tip-2].y
            )

        return sum(fingers)

    def show_image(self, frame, image):
        if image is None:
            return frame

        h, w = frame.shape[:2]

        # Масштабируем картинку на весь экран
        image_resized = cv2.resize(image, (w, h))

        return image_resized

    def run(self):
        cv2.namedWindow("Hand Gesture Meme System", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Hand Gesture Meme System",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb)

            left_fingers = 0
            right_fingers = 0

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    hand_label = handedness.classification[0].label
                    finger_count = self.count_fingers(hand_landmarks, hand_label)

                    if hand_label == "Left":
                        left_fingers = finger_count
                    else:
                        right_fingers = finger_count

            total_fingers = left_fingers + right_fingers

            image_to_show = None

            # ОБЕ РУКИ ОТКРЫТЫ
            if total_fingers == 10:
                image_to_show = self.loaded_images.get("absolute_cinema")

            # Один палец
            elif total_fingers == 1:
                image_to_show = self.loaded_images.get("one_finger")

            # Левая рука
            elif left_fingers > 0:
                image_to_show = self.loaded_images.get(f"left_{left_fingers}")

            # Правая рука
            elif right_fingers > 0:
                image_to_show = self.loaded_images.get(f"right_{right_fingers}")

            frame = self.show_image(frame, image_to_show)

            cv2.putText(
                frame,
                f"Left: {left_fingers} Right: {right_fingers}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Hand Gesture Meme System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = HandGestureImageSystem()
    system.run()