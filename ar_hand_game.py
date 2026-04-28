import cv2
import mediapipe as mp
import numpy as np
import random
import time

class ARHandGame:
    def __init__(self):
        # Инициализация MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Параметры игры
        self.score = 0
        self.level = 1
        self.game_time = 0
        self.start_time = time.time()
        self.objects = []
        self.spawn_rate = 0.05
        self.speed = 2
        
        # Цвета для объектов
        self.colors = [
            (255, 0, 0),    # Синий
            (0, 255, 0),    # Зелёный
            (0, 0, 255),    # Красный
            (255, 255, 0),  # Голубой
            (255, 0, 255),  # Магента
            (0, 255, 255)   # Жёлтый
        ]
        
        self.hand_positions = {}
    
    def spawn_object(self, frame_width, frame_height):
        """Создаёт новый объект для ловли"""
        x = random.randint(50, frame_width - 50)
        y = random.randint(50, frame_height - 50)
        radius = random.randint(15, 30)
        color = random.choice(self.colors)
        points = random.randint(1, 5)
        vx = random.uniform(-self.speed, self.speed)
        vy = random.uniform(-self.speed, self.speed)
        
        return {
            'x': x,
            'y': y,
            'radius': radius,
            'color': color,
            'points': points,
            'vx': vx,
            'vy': vy,
            'caught': False
        }
    
    def update_objects(self, frame_width, frame_height):
        """Обновляет позиции объектов"""
        objects_to_remove = []
        
        for i, obj in enumerate(self.objects):
            # Движение объекта
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # Отскок от стен
            if obj['x'] - obj['radius'] < 0 or obj['x'] + obj['radius'] > frame_width:
                obj['vx'] *= -1
                obj['x'] = np.clip(obj['x'], obj['radius'], frame_width - obj['radius'])
            
            if obj['y'] - obj['radius'] < 0 or obj['y'] + obj['radius'] > frame_height:
                obj['vy'] *= -1
                obj['y'] = np.clip(obj['y'], obj['radius'], frame_height - obj['radius'])
            
            # Удаление объектов, вышедших за пределы (дополнительная проверка)
            if obj['caught']:
                objects_to_remove.append(i)
        
        # Удаляем поймано объекты в обратном порядке
        for i in reversed(objects_to_remove):
            self.objects.pop(i)
    
    def check_collision(self, hand_x, hand_y, hand_size=20):
        """Проверяет столкновение руки с объектами"""
        caught = False
        
        for obj in self.objects:
            distance = np.sqrt((obj['x'] - hand_x) ** 2 + (obj['y'] - hand_y) ** 2)
            
            if distance < obj['radius'] + hand_size:
                self.score += obj['points']
                obj['caught'] = True
                caught = True
                
                # Увеличиваем уровень каждые 50 очков
                self.level = self.score // 50 + 1
                self.spawn_rate = min(0.15, 0.05 + self.level * 0.01)
                self.speed = 2 + self.level * 0.5
        
        return caught
    
    def get_hand_position(self, hand_landmarks, frame_width, frame_height):
        """Получает позицию указательного пальца"""
        # Точка 8 - кончик указательного пальца
        index_finger = hand_landmarks.landmark[8]
        
        x = int(index_finger.x * frame_width)
        y = int(index_finger.y * frame_height)
        
        return x, y
    
    def draw_game_interface(self, frame, frame_height, frame_width):
        """Рисует игровой интерфейс"""
        # Фон для статистики (полупрозрачный)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Статистика
        cv2.putText(frame, f'SCORE: {self.score}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f'LEVEL: {self.level}', (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Время игры
        elapsed = time.time() - self.start_time
        cv2.putText(frame, f'TIME: {int(elapsed)}s', (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Счёт объектов
        cv2.putText(frame, f'Objects: {len(self.objects)}', (frame_width - 300, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        # Инструкции
        cv2.putText(frame, 'Use your hand to catch objects!', (frame_width // 2 - 200, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        # Управление (маленький текст)
        cv2.putText(frame, 'Press Q to quit | Press SPACE to pause', 
                   (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def draw_objects(self, frame):
        """Рисует объекты для ловли"""
        for obj in self.objects:
            # Основной круг
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 
                      obj['radius'], obj['color'], -1)
            
            # Контур
            cv2.circle(frame, (int(obj['x']), int(obj['y'])), 
                      obj['radius'], (255, 255, 255), 2)
            
            # Показываем очки в центре объекта
            cv2.putText(frame, str(obj['points']), 
                       (int(obj['x'] - 10), int(obj['y'] + 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def draw_hand_indicators(self, frame, hand_positions):
        """Рисует индикаторы рук"""
        for hand_id, (x, y) in hand_positions.items():
            # Большой круг указание руки
            cv2.circle(frame, (x, y), 30, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 25, (0, 255, 0), -1)
            
            # Точка указания
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            
            # Текст обозначение руки
            hand_label = "Left" if hand_id == 0 else "Right"
            cv2.putText(frame, hand_label, (x - 20, y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def run(self):
        """Основной цикл игры"""
        paused = False
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Не удалось прочитать кадр с камеры")
                break
            
            # Поворот кадра для удобства
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            
            # Обработка кадра
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Обновление позиций рук
            self.hand_positions = {}
            if results.multi_hand_landmarks and not paused:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    x, y = self.get_hand_position(hand_landmarks, frame_width, frame_height)
                    self.hand_positions[hand_id] = (x, y)
                    
                    # Проверка столкновения
                    self.check_collision(x, y)
            
            # Спавн новых объектов
            if not paused and random.random() < self.spawn_rate:
                self.objects.append(self.spawn_object(frame_width, frame_height))
            
            # Обновление объектов
            if not paused:
                self.update_objects(frame_width, frame_height)
            
            # Рисуем объекты
            self.draw_objects(frame)
            
            # Рисуем индикаторы рук
            self.draw_hand_indicators(frame, self.hand_positions)
            
            # Рисуем игровой интерфейс
            frame = self.draw_game_interface(frame, frame_height, frame_width)
            
            # Пауза текст
            if paused:
                overlay = frame.copy()
                cv2.rectangle(overlay, (frame_width // 2 - 150, frame_height // 2 - 50),
                            (frame_width // 2 + 150, frame_height // 2 + 50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                cv2.putText(frame, 'PAUSED', (frame_width // 2 - 80, frame_height // 2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Вывод
            cv2.imshow('AR Hand Game', frame)
            
            # Управление
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                # Перезагрузка игры
                self.score = 0
                self.level = 1
                self.objects = []
                self.start_time = time.time()
        
        # Очистка
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Финальная статистика
        print(f"\n{'='*40}")
        print(f"GAME OVER!")
        print(f"{'='*40}")
        print(f"Final Score: {self.score}")
        print(f"Final Level: {self.level}")
        print(f"Game Time: {int(time.time() - self.start_time)}s")
        print(f"{'='*40}\n")

def main():
    print("="*50)
    print("     🎮 AR HAND DETECTION GAME 🎮")
    print("="*50)
    print("\nInstructions:")
    print("- Use your hand to catch the colored objects")
    print("- Each object has different point values")
    print("- Complete levels to increase difficulty")
    print("\nControls:")
    print("- Move hand to catch objects")
    print("- SPACE: Pause/Resume")
    print("- R: Restart game")
    print("- Q: Quit game")
    print("\n" + "="*50 + "\n")
    
    try:
        game = ARHandGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed:")
        print("pip install opencv-python mediapipe numpy")

if __name__ == '__main__':
    main()