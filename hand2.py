import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ─────────────────────────────────────────────
#  AR 3D CUBE BUILDER — Hand Gesture Control
#  Создание 3D кубиков жестами рук
# ─────────────────────────────────────────────

class Cube3D:
    """3D куб с проекцией на 2D экран"""

    VERTICES = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # задняя грань
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],  # передняя грань
    ], dtype=float)

    EDGES = [
        (0,1),(1,2),(2,3),(3,0),  # задняя грань
        (4,5),(5,6),(6,7),(7,4),  # передняя грань
        (0,4),(1,5),(2,6),(3,7),  # рёбра соединения
    ]

    FACES = [
        ([0,1,2,3], 0),  # зад
        ([4,5,6,7], 1),  # перед
        ([0,1,5,4], 2),  # низ
        ([2,3,7,6], 3),  # верх
        ([0,3,7,4], 4),  # лево
        ([1,2,6,5], 5),  # право
    ]

    PALETTES = [
        # (основной, грань0..5)
        [(50, 180, 255), (30,120,200), (80,200,255), (20,100,180), (60,160,220), (40,140,210), (70,190,245)],
        [(80, 255, 140), (40,180, 90), (100,255,160),(30,160, 80), (60,210,120),(50,195,110),(90,240,150)],
        [(255, 100,  80), (200, 60, 50),(255,130,110),(180, 50, 40),(220, 80, 60),(210, 70, 55),(240,110, 90)],
        [(255, 220,  60), (200,170, 30),(255,235, 90),(180,155, 20),(220,200, 50),(210,185, 40),(240,215, 75)],
        [(200,  80, 255), (150, 40,200),(220,110,255),(130, 30,180),(170, 60,220),(160, 50,210),(190, 90,240)],
    ]

    def __init__(self, cx, cy, size=40, color_idx=0):
        self.cx = cx          # центр X на экране
        self.cy = cy          # центр Y на экране
        self.size = size      # размер
        self.color_idx = color_idx % len(self.PALETTES)
        self.rx = 0.4         # угол поворота X (рад)
        self.ry = 0.6         # угол поворота Y
        self.rz = 0.0         # угол поворота Z
        self.spin_x = 0.008   # авто-вращение X
        self.spin_y = 0.012   # авто-вращение Y
        self.birth = time.time()
        self.alpha = 0.0      # прозрачность (0→1 при появлении)
        self.alive = True
        self.grabbed = False
        self.grab_dx = 0
        self.grab_dy = 0

    def _rotate(self):
        """Матрица поворота RxRyRz"""
        cx, sx = math.cos(self.rx), math.sin(self.rx)
        cy, sy = math.cos(self.ry), math.sin(self.ry)
        cz, sz = math.cos(self.rz), math.sin(self.rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx

    def project(self, v3, fov=400):
        """Перспективная проекция 3D→2D"""
        z = v3[2] + 4.0
        if z <= 0.001: z = 0.001
        px = int(v3[0] * fov / z + self.cx)
        py = int(v3[1] * fov / z + self.cy)
        return px, py

    def update(self):
        if not self.grabbed:
            self.ry += self.spin_y
            self.rx += self.spin_x
        # появление
        age = time.time() - self.birth
        self.alpha = min(1.0, age / 0.4)

    def draw(self, frame):
        self.update()
        palette = self.PALETTES[self.color_idx]
        R = self._rotate()
        verts_3d = [(R @ (v * self.size / 50)).tolist() for v in self.VERTICES]
        verts_2d = [self.project(v) for v in verts_3d]
        a = self.alpha

        # Сортировка граней по глубине (painter's algorithm)
        face_depths = []
        for face_verts, fi in self.FACES:
            z_avg = sum(verts_3d[i][2] for i in face_verts) / len(face_verts)
            face_depths.append((z_avg, face_verts, fi))
        face_depths.sort(key=lambda x: x[0])

        overlay = frame.copy()
        for _, fv, fi in face_depths:
            pts = np.array([verts_2d[i] for i in fv], dtype=np.int32)
            # Нормаль грани
            v0 = np.array(verts_3d[fv[0]])
            v1 = np.array(verts_3d[fv[1]])
            v2 = np.array(verts_3d[fv[2]])
            n = np.cross(v1 - v0, v2 - v0)
            light = np.array([0.5, -0.8, 1.0])
            if np.linalg.norm(n) > 0:
                n = n / np.linalg.norm(n)
            diffuse = max(0.3, float(np.dot(n, light / np.linalg.norm(light))))
            base_c = palette[fi + 1]
            col = tuple(int(c * diffuse) for c in base_c)
            cv2.fillConvexPoly(overlay, pts, col)
            cv2.polylines(overlay, [pts], True, (255,255,255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, a * 0.85, frame, 1 - a * 0.85, 0, frame)

        # Рёбра поверх
        for e in self.EDGES:
            p1, p2 = verts_2d[e[0]], verts_2d[e[1]]
            cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    def contains_point(self, px, py, radius=35):
        return math.hypot(px - self.cx, py - self.cy) < radius + self.size * 0.5


# ─────────────────────────────────────────────
#  Детектор жестов
# ─────────────────────────────────────────────

class GestureDetector:
    # Индексы кончиков пальцев
    TIPS = [4, 8, 12, 16, 20]
    # Индексы сгибов для проверки «вытянут ли палец»
    PIPS = [3, 6, 10, 14, 18]

    def finger_states(self, lm, w, h):
        """Возвращает список bool — вытянут ли каждый палец [большой, указат., средн., безым., мизинец]"""
        tips = [(int(lm[t].x*w), int(lm[t].y*h)) for t in self.TIPS]
        pips = [(int(lm[p].x*w), int(lm[p].y*h)) for p in self.PIPS]
        states = []
        # Большой палец: сравниваем X
        states.append(tips[0][0] > pips[0][0])
        # Остальные: сравниваем Y
        for i in range(1, 5):
            states.append(tips[i][1] < pips[i][1])
        return states

    def pinch_distance(self, lm, w, h):
        """Расстояние между большим и указательным пальцами"""
        t = lm[4]
        i = lm[8]
        return math.hypot((t.x - i.x)*w, (t.y - i.y)*h)

    def pinch_center(self, lm, w, h):
        """Центр щипка"""
        t = lm[4]
        i = lm[8]
        return int((t.x + i.x)/2*w), int((t.y + i.y)/2*h)

    def index_tip(self, lm, w, h):
        return int(lm[8].x*w), int(lm[8].y*h)

    def palm_center(self, lm, w, h):
        wrist = lm[0]
        mid = lm[9]
        return int((wrist.x+mid.x)/2*w), int((wrist.y+mid.y)/2*h)


# ─────────────────────────────────────────────
#  Главное приложение
# ─────────────────────────────────────────────

class AR3DCubeBuilder:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = GestureDetector()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cubes = []
        self.color_idx = 0
        self.pinch_cooldown = 0.0   # не создавать куб сразу снова
        self.grabbed_cube = None    # куб, который тащит правая рука
        self.delete_cooldown = 0.0

        # UI
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.last_action = ""
        self.last_action_time = 0.0

        print("\n" + "="*55)
        print("   🧊  AR 3D CUBE BUILDER  🧊")
        print("="*55)
        print("  🤏  Щипок (большой + указат.) — создать куб")
        print("  ✊  Кулак над кубом — схватить и переместить")
        print("  ☝️   Один палец (указат.) над кубом — удалить")
        print("  🖐  Все пальцы — стереть все кубики")
        print("  SPACE — сменить цвет  |  Q — выход")
        print("="*55 + "\n")

    def _show_action(self, text):
        self.last_action = text
        self.last_action_time = time.time()

    def _draw_ui(self, frame, h, w):
        # Панель сверху
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        palette_name = ["💙 СИНИЙ","💚 ЗЕЛЁНЫЙ","❤️ КРАСНЫЙ","💛 ЖЁЛТЫЙ","💜 ФИОЛЕТ"][self.color_idx % 5]
        cv2.putText(frame, f"🧊 КУБИКИ: {len(self.cubes)}   Цвет: {palette_name}",
                    (15, 36), self.font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, "Q-выход | SPACE-цвет | 5пальцев-очистить",
                    (w - 460, 36), self.font, 0.5, (130,130,130), 1, cv2.LINE_AA)

        # Последнее действие
        if time.time() - self.last_action_time < 2.0:
            alpha = min(1.0, (2.0 - (time.time() - self.last_action_time)) / 0.5)
            col = (int(100*alpha), int(255*alpha), int(180*alpha))
            cv2.putText(frame, self.last_action, (w//2 - 200, h - 30),
                        self.font, 0.9, col, 2, cv2.LINE_AA)

        # Цветовая пилюля
        pal = Cube3D.PALETTES[self.color_idx % 5]
        cv2.circle(frame, (w - 490, 28), 14, pal[0], -1)
        cv2.circle(frame, (w - 490, 28), 14, (255,255,255), 1)

    def _draw_gesture_hint(self, frame, cx, cy, label, color=(0,255,200)):
        cv2.circle(frame, (cx, cy), 18, color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (cx - 10, cy - 25),
                    self.font, 0.5, color, 1, cv2.LINE_AA)

    def run(self):
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            now = time.time()

            # ── Рисуем кубики ───────────────────────────────────────────
            for cube in self.cubes:
                cube.draw(frame)

            # ── Обработка рук ───────────────────────────────────────────
            if results.multi_hand_landmarks:
                for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    lm = hand_lm.landmark
                    fs = self.detector.finger_states(lm, w, h)
                    pinch_d = self.detector.pinch_distance(lm, w, h)
                    px, py = self.detector.pinch_center(lm, w, h)
                    ix, iy = self.detector.index_tip(lm, w, h)

                    # Рисуем скелет руки (тонко)
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec((80,80,80), 1, 1),
                        self.mp_draw.DrawingSpec((60,60,60), 1, 1),
                    )

                    # ── ЖЕСТ: ЩИПОК → создать куб ───────────────────────
                    if pinch_d < 35 and now > self.pinch_cooldown:
                        # проверяем что средний палец НЕ вытянут (иначе путается с захватом)
                        if not fs[2]:
                            new_cube = Cube3D(px, py, size=50, color_idx=self.color_idx)
                            self.cubes.append(new_cube)
                            self.pinch_cooldown = now + 0.7
                            self._show_action(f"✨ Куб создан! (всего: {len(self.cubes)})")
                            cv2.circle(frame, (px, py), 30, (100, 255, 200), 3, cv2.LINE_AA)
                        self._draw_gesture_hint(frame, px, py, "Щипок", (100,255,200))

                    # ── ЖЕСТ: КУЛАК → схватить куб ──────────────────────
                    fist = not any(fs[1:])  # все пальцы кроме большого согнуты
                    palm_x, palm_y = self.detector.palm_center(lm, w, h)
                    if fist:
                        if self.grabbed_cube is None:
                            for cube in reversed(self.cubes):
                                if cube.contains_point(palm_x, palm_y):
                                    self.grabbed_cube = cube
                                    cube.grabbed = True
                                    cube.grab_dx = cube.cx - palm_x
                                    cube.grab_dy = cube.cy - palm_y
                                    self._show_action("✊ Куб схвачен!")
                                    break
                        if self.grabbed_cube:
                            self.grabbed_cube.cx = palm_x + self.grabbed_cube.grab_dx
                            self.grabbed_cube.cy = palm_y + self.grabbed_cube.grab_dy
                            # вращение пальцами
                            self.grabbed_cube.ry += (lm[8].x - lm[12].x) * 3
                            self.grabbed_cube.rx += (lm[8].y - lm[12].y) * 3
                            self._draw_gesture_hint(frame, palm_x, palm_y, "Тащу", (255,200,50))
                    else:
                        if self.grabbed_cube:
                            self.grabbed_cube.grabbed = False
                            self.grabbed_cube = None

                    # ── ЖЕСТ: ОДИН УКАЗАТЕЛЬНЫЙ → удалить ──────────────
                    only_index = fs[1] and not fs[2] and not fs[3] and not fs[4]
                    if only_index and now > self.delete_cooldown:
                        for cube in list(self.cubes):
                            if cube.contains_point(ix, iy, 10):
                                self.cubes.remove(cube)
                                self.delete_cooldown = now + 0.5
                                self._show_action("🗑️  Куб удалён")
                                break
                        self._draw_gesture_hint(frame, ix, iy, "Удалить", (255, 80, 80))

                    # ── ЖЕСТ: ВСЕ ПАЛЬЦЫ → очистить ────────────────────
                    all_open = all(fs[1:])
                    if all_open and now > self.pinch_cooldown:
                        if len(self.cubes) > 0:
                            self.cubes.clear()
                            self.grabbed_cube = None
                            self.pinch_cooldown = now + 1.5
                            self._show_action("🧹 Все кубики удалены!")

            # ── UI ────────────────────────────────────────────────────
            self._draw_ui(frame, h, w)

            cv2.imshow("🧊 AR 3D Cube Builder", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.color_idx = (self.color_idx + 1) % 5
                names = ["синий","зелёный","красный","жёлтый","фиолет"]
                self._show_action(f"🎨 Цвет изменён: {names[self.color_idx]}")
            elif key == ord('r'):
                self.cubes.clear()
                self.grabbed_cube = None
                self._show_action("🔄 Сцена очищена")

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print(f"\nСоздано кубиков за сессию: {len(self.cubes)} осталось на экране")
        print("Спасибо за игру! 🧊")


# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("   🧊  AR 3D CUBE BUILDER  🧊")
    print("="*55)
    print("Установка зависимостей:")
    print("  pip install opencv-python mediapipe numpy")
    print("="*55)
    try:
        app = AR3DCubeBuilder()
        app.run()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Убедитесь что установлены: opencv-python mediapipe numpy")

if __name__ == '__main__':
    main()
    