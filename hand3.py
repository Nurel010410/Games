import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ═══════════════════════════════════════════════════════════
#   AR 3D CUBE BUILDER  —  строй конструкции из кубиков!
#   Кубики маленькие, прикрепляются к сетке, можно строить
# ═══════════════════════════════════════════════════════════

GRID_SIZE = 40          # размер одного кубика в пикселях
CUBE_COLOR  = (255, 60, 180)   # розово-маджента (BGR)
CUBE_COLOR2 = (255, 120, 220)  # светлее для граней
EDGE_COLOR  = (255, 180, 255)  # контур
GHOST_COLOR = (180,  40, 130)  # призрак под пальцем

# Вершины единичного куба (от 0 до 1)
UNIT_VERTS = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],   # нижняя грань
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],   # верхняя грань
], dtype=float)

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# Грани (для закрашивания)
FACES = [
    ([4,5,6,7], 1.0),   # верх   — самая светлая
    ([0,1,5,4], 0.65),  # перед
    ([1,2,6,5], 0.55),  # право
    ([0,3,7,4], 0.55),  # лево
    ([2,3,7,6], 0.65),  # зад
    ([0,1,2,3], 0.4),   # низ
]

# Изометрическая проекция (угол 30°)
ISO_ANGLE = math.radians(30)
cos30 = math.cos(ISO_ANGLE)
sin30 = math.sin(ISO_ANGLE)

def iso_project(x3, y3, z3, scale, ox, oy):
    """Изометрическая проекция 3D → 2D"""
    sx = (x3 - z3) * cos30
    sy = (x3 + z3) * sin30 - y3
    return int(ox + sx * scale), int(oy + sy * scale)


class SmallCube:
    """Один маленький кубик в сетке"""

    def __init__(self, gx, gy, gz, color_scheme=0):
        self.gx = gx   # позиция в сетке
        self.gy = gy
        self.gz = gz
        self.color_scheme = color_scheme
        self.birth = time.time()

    def draw(self, frame, origin, scale, alpha=1.0):
        ox, oy = origin
        age = time.time() - self.birth
        fade = min(1.0, age / 0.25)

        # Вычисляем вершины в изо-проекции
        pts = []
        for v in UNIT_VERTS:
            wx = self.gx + v[0]
            wy = self.gy + v[1]
            wz = self.gz + v[2]
            px, py = iso_project(wx, wy, wz, scale, ox, oy)
            pts.append((px, py))

        schemes = [
            (CUBE_COLOR, CUBE_COLOR2, EDGE_COLOR),
            ((60, 255, 180), (120, 255, 210), (180, 255, 240)),  # зелёный
            ((60, 180, 255), (120, 210, 255), (180, 240, 255)),  # синий
            ((255, 220, 60), (255, 235, 120), (255, 245, 180)),  # жёлтый
            ((255, 100, 60), (255, 140, 100), (255, 180, 160)),  # оранжевый
        ]
        c_dark, c_light, c_edge = schemes[self.color_scheme % len(schemes)]

        # Рисуем грани
        overlay = frame.copy()
        for face_verts, brightness in FACES:
            poly = np.array([pts[i] for i in face_verts], dtype=np.int32)
            col = tuple(int(ch * brightness * fade) for ch in c_dark)
            cv2.fillConvexPoly(overlay, poly, col)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Рёбра
        for e in EDGES:
            p1, p2 = pts[e[0]], pts[e[1]]
            col_e = tuple(int(ch * fade) for ch in c_edge)
            cv2.line(frame, p1, p2, col_e, 1, cv2.LINE_AA)


def draw_ghost_cube(frame, gx, gy, gz, origin, scale):
    """Призрак кубика под пальцем"""
    ox, oy = origin
    pts = []
    for v in UNIT_VERTS:
        px, py = iso_project(gx+v[0], gy+v[1], gz+v[2], scale, ox, oy)
        pts.append((px, py))
    for e in EDGES:
        p1, p2 = pts[e[0]], pts[e[1]]
        cv2.line(frame, p1, p2, GHOST_COLOR, 1, cv2.LINE_AA)


def draw_grid_floor(frame, origin, scale, grid_w=12, grid_d=10):
    """Рисует сетку-пол"""
    ox, oy = origin
    col = (60, 30, 50)
    for x in range(grid_w + 1):
        p1 = iso_project(x, 0, 0, scale, ox, oy)
        p2 = iso_project(x, 0, grid_d, scale, ox, oy)
        cv2.line(frame, p1, p2, col, 1)
    for z in range(grid_d + 1):
        p1 = iso_project(0, 0, z, scale, ox, oy)
        p2 = iso_project(grid_w, 0, z, scale, ox, oy)
        cv2.line(frame, p1, p2, col, 1)


# ──────────────────────────────────────────────
#  Детектор жестов
# ──────────────────────────────────────────────

class GestureDetector:
    TIPS = [4, 8, 12, 16, 20]
    PIPS = [3, 6, 10, 14, 18]

    def finger_states(self, lm, w, h):
        states = [lm[4].x > lm[3].x]  # большой
        for i in range(1, 5):
            states.append(lm[self.TIPS[i]].y < lm[self.PIPS[i]].y)
        return states

    def pinch_dist(self, lm, w, h):
        t, i = lm[4], lm[8]
        return math.hypot((t.x-i.x)*w, (t.y-i.y)*h)

    def pinch_center(self, lm, w, h):
        t, i = lm[4], lm[8]
        return int((t.x+i.x)/2*w), int((t.y+i.y)/2*h)

    def index_tip(self, lm, w, h):
        return int(lm[8].x*w), int(lm[8].y*h)


# ──────────────────────────────────────────────
#  Главное приложение
# ──────────────────────────────────────────────

class CubeBuilderApp:
    GRID_W = 12
    GRID_D = 10
    GRID_H = 20   # максимальная высота стопки

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands_model = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = GestureDetector()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Сетка кубиков: (gx, gz) → высота стека
        self.grid = {}          # (gx,gz) → list of SmallCube
        self.color_idx = 0
        self.scale = 32         # масштаб изо-проекции

        # Параметры камеры изо-проекции (смещение начала координат)
        self.origin = (400, 500)  # подберём под размер окна

        self.last_place_time = 0.0
        self.last_del_time   = 0.0
        self.place_cd = 0.4    # секунды между укладкой
        self.del_cd   = 0.4

        self.last_action = ""
        self.last_action_t = 0.0

        self.mode = "BUILD"  # BUILD / DELETE
        self.ghost_pos = None

        print("\n" + "═"*55)
        print("   🧱  AR CUBE BUILDER — Строй что хочешь!  🧱")
        print("═"*55)
        print("  🤏  Щипок  — положить кубик")
        print("  ✌️   Два пальца над кубиком — удалить верхний")
        print("  🖐   Все пальцы — режим удаления / стройки (toggle)")
        print("  SPACE — сменить цвет   |   C — очистить   |   Q — выйти")
        print("═"*55 + "\n")

    def _action(self, txt):
        self.last_action = txt
        self.last_action_t = time.time()

    def _screen_to_grid(self, sx, sy):
        """Конвертируем экранные координаты в сетку (гx, gz)"""
        ox, oy = self.origin
        sc = self.scale
        # Обратная изометрия (y=0 плоскость)
        # sx = ox + (gx - gz)*cos30*sc
        # sy = oy + (gx + gz)*sin30*sc
        rx = (sx - ox) / sc
        ry = (sy - oy) / sc
        gz_f = (ry / sin30 - rx / cos30) / 2
        gx_f = rx / cos30 + gz_f
        gx = int(math.floor(gx_f))
        gz = int(math.floor(gz_f))
        return gx, gz

    def _valid(self, gx, gz):
        return 0 <= gx < self.GRID_W and 0 <= gz < self.GRID_D

    def _stack_height(self, gx, gz):
        return len(self.grid.get((gx, gz), []))

    def _place_cube(self, gx, gz):
        now = time.time()
        if not self._valid(gx, gz): return
        if now - self.last_place_time < self.place_cd: return
        if self._stack_height(gx, gz) >= self.GRID_H: return
        gy = self._stack_height(gx, gz)  # ставим на вершину стека
        cube = SmallCube(gx, gy, gz, self.color_idx)
        self.grid.setdefault((gx, gz), []).append(cube)
        self.last_place_time = now
        total = sum(len(v) for v in self.grid.values())
        self._action(f"🧱 Кубик [{gx},{gz}] h={gy+1}  (всего: {total})")

    def _delete_top(self, gx, gz):
        now = time.time()
        if not self._valid(gx, gz): return
        if now - self.last_del_time < self.del_cd: return
        stack = self.grid.get((gx, gz), [])
        if stack:
            stack.pop()
            if not stack:
                del self.grid[(gx, gz)]
            self.last_del_time = now
            self._action(f"🗑️  Удалён кубик [{gx},{gz}]")

    def _draw_ui(self, frame, h, w):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w, 58), (5,5,5), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        colors = ["🌸РОЗОВ","💚ЗЕЛЁН","💙СИНИЙ","💛ЖЁЛТ","🧡ОРАНЖ"]
        mode_col = (80,255,160) if self.mode=="BUILD" else (80,80,255)
        cv2.putText(frame,
            f"Режим: {self.mode}  |  Цвет: {colors[self.color_idx]}  |  Кубиков: {sum(len(v) for v in self.grid.values())}",
            (12, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, mode_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "SPACE=цвет  C=очистить  Q=выход",
            (w-370, 36), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100,100,100), 1)

        if time.time() - self.last_action_t < 2.5:
            fade = min(1.0, (2.5-(time.time()-self.last_action_t))/0.4)
            col = (int(120*fade), int(255*fade), int(200*fade))
            cv2.putText(frame, self.last_action,
                (w//2-220, h-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 1, cv2.LINE_AA)

        # Подсказки жестов
        hints = [
            "🤏 Щипок = положить",
            "✌️  2 пальца = удалить",
            "🖐 5 пальцев = режим",
        ]
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (10, h - 80 + i*22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140), 1)

    def _draw_finger_cursor(self, frame, sx, sy, gx, gz, gesture):
        """Визуальный курсор руки"""
        valid = self._valid(gx, gz)
        col = (100, 255, 180) if valid else (60, 60, 180)
        cv2.circle(frame, (sx, sy), 12, col, 2, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),  4, col, -1, cv2.LINE_AA)
        label = {"pinch": "📦 ставить", "two": "🗑 удалить", "open": "👋"}.get(gesture, "")
        if label:
            cv2.putText(frame, label, (sx+15, sy-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    def run(self):
        ret, frame = self.cap.read()
        if ret:
            h_frame, w_frame = frame.shape[:2]
            # Центрируем начало координат изо-сетки
            # Центр сетки в середине экрана
            cx_grid = iso_project(self.GRID_W/2, 0, self.GRID_D/2,
                                  self.scale, 0, 0)
            self.origin = (w_frame//2 - cx_grid[0], int(h_frame*0.72))

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok: break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_model.process(rgb)

            # Тёмный фон сцены
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0),(w,h),(0,0,0),-1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

            # Сетка-пол
            draw_grid_floor(frame, self.origin, self.scale, self.GRID_W, self.GRID_D)

            # Кубики (сортировка по изо-глубине: большой gx+gz рисуем первым)
            all_cubes = []
            for (gx,gz), stack in self.grid.items():
                for cube in stack:
                    all_cubes.append((gx + gz, cube))
            all_cubes.sort(key=lambda x: x[0])
            for _, cube in all_cubes:
                cube.draw(frame, self.origin, self.scale)

            # Призрак
            if self.ghost_pos:
                gx, gz = self.ghost_pos
                if self._valid(gx, gz):
                    gy = self._stack_height(gx, gz)
                    draw_ghost_cube(frame, gx, gy, gz, self.origin, self.scale)

            # ── Руки ──────────────────────────────────────────────────
            self.ghost_pos = None
            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    lm = hand_lm.landmark
                    fs = self.detector.finger_states(lm, w, h)
                    pd = self.detector.pinch_dist(lm, w, h)
                    ix, iy = self.detector.index_tip(lm, w, h)
                    pcx, pcy = self.detector.pinch_center(lm, w, h)

                    gx, gz = self._screen_to_grid(ix, iy)
                    self.ghost_pos = (gx, gz)

                    # Определяем жест
                    pinch   = pd < 35
                    two_up  = fs[1] and fs[2] and not fs[3] and not fs[4]
                    all_open = all(fs[1:])
                    gesture = "pinch" if pinch else ("two" if two_up else ("open" if all_open else ""))

                    # Скелет руки
                    self.mp_draw.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec((40,40,40),1,1),
                        self.mp_draw.DrawingSpec((30,30,30),1,1),
                    )

                    # Щипок → положить кубик
                    if pinch:
                        pgx, pgz = self._screen_to_grid(pcx, pcy)
                        self.ghost_pos = (pgx, pgz)
                        self._place_cube(pgx, pgz)
                        self._draw_finger_cursor(frame, pcx, pcy, pgx, pgz, "pinch")

                    # Два пальца → удалить верхний
                    elif two_up:
                        self._delete_top(gx, gz)
                        self._draw_finger_cursor(frame, ix, iy, gx, gz, "two")

                    # Все пальцы открыты → toggle режима (для справки в UI)
                    elif all_open:
                        self._draw_finger_cursor(frame, ix, iy, gx, gz, "open")
                    else:
                        self._draw_finger_cursor(frame, ix, iy, gx, gz, "")

            self._draw_ui(frame, h, w)
            cv2.imshow("🧱 AR Cube Builder", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.color_idx = (self.color_idx + 1) % 5
                names = ["розовый","зелёный","синий","жёлтый","оранжевый"]
                self._action(f"🎨 Цвет: {names[self.color_idx]}")
            elif key == ord('c'):
                self.grid.clear()
                self._action("🧹 Очищено!")
            elif key == ord('z'):
                # Отмена последнего действия — удалить последний поставленный куб
                for (gx,gz) in reversed(list(self.grid.keys())):
                    stack = self.grid[(gx,gz)]
                    if stack:
                        stack.pop()
                        if not stack: del self.grid[(gx,gz)]
                        self._action("↩️  Отмена")
                        break

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands_model.close()
        total = sum(len(v) for v in self.grid.values())
        print(f"\nПостроено кубиков: {total}")
        print("До свидания! 🧱")


def main():
    print("\n" + "═"*55)
    print("   🧱  AR CUBE BUILDER  🧱")
    print("═"*55)
    print("pip install opencv-python mediapipe numpy")
    print("═"*55)
    try:
        app = CubeBuilderApp()
        app.run()
    except Exception as e:
        import traceback
        print(f"❌ Ошибка: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()