import tkinter as tk
from tkinter import messagebox
import random
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.root.title("🐍 ЗМЕЙКА")
        self.root.resizable(False, False)
        self.root.configure(bg='#2c3e50')
        
        # Параметры игры
        self.grid_size = 20
        self.grid_width = 20
        self.grid_height = 20
        self.canvas_width = self.grid_width * self.grid_size
        self.canvas_height = self.grid_height * self.grid_size
        
        # Создаём интерфейс
        self.create_widgets()
        
        # Инициализация игры
        self.reset_game()
        
        # Привязка клавиш
        self.root.bind('<Key>', self.on_key_press)
        
        # Запуск игры
        self.game_loop()
    
    def create_widgets(self):
        # Верхняя панель с информацией
        top_frame = tk.Frame(self.root, bg='#2c3e50')
        top_frame.pack(pady=10)
        
        # Счёт
        self.score_label = tk.Label(top_frame, text='Счёт: 0', 
                                     font=('Arial', 18, 'bold'), 
                                     bg='#2c3e50', fg='#ecf0f1')
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        # Уровень
        self.level_label = tk.Label(top_frame, text='Уровень: 1', 
                                     font=('Arial', 18, 'bold'), 
                                     bg='#2c3e50', fg='#ecf0f1')
        self.level_label.pack(side=tk.LEFT, padx=20)
        
        # Скорость
        self.speed_label = tk.Label(top_frame, text='Скорость: 🟢 Нормальная', 
                                     font=('Arial', 18, 'bold'), 
                                     bg='#2c3e50', fg='#ecf0f1')
        self.speed_label.pack(side=tk.LEFT, padx=20)
        
        # Canvas для игры
        self.canvas = tk.Canvas(self.root, 
                               width=self.canvas_width, 
                               height=self.canvas_height,
                               bg='#1a1a1a',
                               highlightthickness=2,
                               highlightbackground='#34495e')
        self.canvas.pack(pady=10)
        
        # Нижняя панель с инструкциями и кнопками
        bottom_frame = tk.Frame(self.root, bg='#2c3e50')
        bottom_frame.pack(pady=10)
        
        # Инструкции
        instructions = tk.Label(bottom_frame, 
                               text='⬆️ ⬇️ ⬅️ ➡️ или WASD - управление | P - пауза | R - перезагрузка',
                               font=('Arial', 10),
                               bg='#2c3e50', 
                               fg='#95a5a6')
        instructions.pack()
        
        # Кнопки управления
        buttons_frame = tk.Frame(self.root, bg='#2c3e50')
        buttons_frame.pack(pady=10)
        
        self.start_button = tk.Button(buttons_frame, text='▶ Начать/Пауза', 
                                     command=self.toggle_pause,
                                     bg='#3498db', fg='white',
                                     font=('Arial', 10, 'bold'),
                                     padx=10, pady=5)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        restart_button = tk.Button(buttons_frame, text='🔄 Перезагрузка',
                                  command=self.restart_game,
                                  bg='#e74c3c', fg='white',
                                  font=('Arial', 10, 'bold'),
                                  padx=10, pady=5)
        restart_button.pack(side=tk.LEFT, padx=5)
    
    def reset_game(self):
        # Инициализация змейки
        self.snake = [(10, 10), (9, 10), (8, 10)]
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        
        # Еда
        self.food = self.generate_food()
        
        # Статистика
        self.score = 0
        self.level = 1
        self.speed = 100  # ms
        self.game_over = False
        self.paused = False
        
        self.update_stats()
    
    def generate_food(self):
        while True:
            food = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if food not in self.snake:
                return food
    
    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.start_button.config(text='⏸ Пауза (активна)')
        else:
            self.start_button.config(text='▶ Начать/Пауза')
    
    def restart_game(self):
        self.reset_game()
        self.draw()
    
    def on_key_press(self, event):
        key = event.keysym.lower()
        
        # Управление
        if key in ('up', 'w'):
            if self.direction != Direction.DOWN:
                self.next_direction = Direction.UP
        elif key in ('down', 's'):
            if self.direction != Direction.UP:
                self.next_direction = Direction.DOWN
        elif key in ('left', 'a'):
            if self.direction != Direction.RIGHT:
                self.next_direction = Direction.LEFT
        elif key in ('right', 'd'):
            if self.direction != Direction.LEFT:
                self.next_direction = Direction.RIGHT
        elif key == 'p':
            self.toggle_pause()
        elif key == 'r':
            self.restart_game()
    
    def update(self):
        if self.game_over or self.paused:
            return
        
        # Обновляем направление
        self.direction = self.next_direction
        
        # Вычисляем новую позицию головы
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Проверка столкновения со стеной
        if (new_head[0] < 0 or new_head[0] >= self.grid_width or
            new_head[1] < 0 or new_head[1] >= self.grid_height):
            self.end_game("Столкновение со стеной!")
            return
        
        # Проверка столкновения с собой
        if new_head in self.snake:
            self.end_game("Змейка столкнулась с собой!")
            return
        
        # Добавляем новую голову
        self.snake.insert(0, new_head)
        
        # Проверка поедания еды
        if new_head == self.food:
            self.score += 10
            self.level = self.score // 100 + 1
            self.speed = max(20, 100 - (self.level - 1) * 5)
            self.food = self.generate_food()
        else:
            # Удаляем хвост если еда не поедена
            self.snake.pop()
        
        self.update_stats()
    
    def draw(self):
        self.canvas.delete('all')
        
        # Сетка (опционально)
        self.canvas.configure(bg='#1a1a1a')
        
        # Рисуем еду
        fx, fy = self.food
        self.canvas.create_oval(
            fx * self.grid_size + 2,
            fy * self.grid_size + 2,
            (fx + 1) * self.grid_size - 2,
            (fy + 1) * self.grid_size - 2,
            fill='#e74c3c', outline='#c0392b'
        )
        
        # Рисуем змейку
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # Голова
                self.canvas.create_oval(
                    x * self.grid_size + 1,
                    y * self.grid_size + 1,
                    (x + 1) * self.grid_size - 1,
                    (y + 1) * self.grid_size - 1,
                    fill='#2ecc71', outline='#27ae60'
                )
                # Глаза
                if self.direction == Direction.RIGHT:
                    self.canvas.create_oval(
                        x * self.grid_size + 12, y * self.grid_size + 6,
                        x * self.grid_size + 15, y * self.grid_size + 9,
                        fill='white'
                    )
                elif self.direction == Direction.LEFT:
                    self.canvas.create_oval(
                        x * self.grid_size + 5, y * self.grid_size + 6,
                        x * self.grid_size + 8, y * self.grid_size + 9,
                        fill='white'
                    )
                elif self.direction == Direction.UP:
                    self.canvas.create_oval(
                        x * self.grid_size + 6, y * self.grid_size + 5,
                        x * self.grid_size + 9, y * self.grid_size + 8,
                        fill='white'
                    )
                elif self.direction == Direction.DOWN:
                    self.canvas.create_oval(
                        x * self.grid_size + 6, y * self.grid_size + 12,
                        x * self.grid_size + 9, y * self.grid_size + 15,
                        fill='white'
                    )
            else:  # Тело
                color = '#27ae60' if i % 2 == 0 else '#229954'
                self.canvas.create_rectangle(
                    x * self.grid_size + 1,
                    y * self.grid_size + 1,
                    (x + 1) * self.grid_size - 1,
                    (y + 1) * self.grid_size - 1,
                    fill=color, outline='#1e8449'
                )
        
        # Пауза текст
        if self.paused:
            self.canvas.create_text(
                self.canvas_width // 2,
                self.canvas_height // 2,
                text='ПАУЗА',
                font=('Arial', 40, 'bold'),
                fill='#e74c3c'
            )
        
        # Game Over текст
        if self.game_over:
            self.canvas.create_text(
                self.canvas_width // 2,
                self.canvas_height // 2 - 30,
                text='GAME OVER',
                font=('Arial', 40, 'bold'),
                fill='#e74c3c'
            )
            self.canvas.create_text(
                self.canvas_width // 2,
                self.canvas_height // 2 + 30,
                text=f'Финальный счёт: {self.score}',
                font=('Arial', 20),
                fill='#ecf0f1'
            )
    
    def update_stats(self):
        self.score_label.config(text=f'Счёт: {self.score}')
        self.level_label.config(text=f'Уровень: {self.level}')
        
        if self.level <= 3:
            speed_text = '🟢 Медленная'
        elif self.level <= 6:
            speed_text = '🟡 Нормальная'
        elif self.level <= 10:
            speed_text = '🔴 Быстрая'
        else:
            speed_text = '⚫ Экстремальная'
        
        self.speed_label.config(text=f'Скорость: {speed_text}')
    
    def end_game(self, message):
        self.game_over = True
        messagebox.showinfo('Game Over', f'{message}\n\nВаш счёт: {self.score}\nУровень: {self.level}')
    
    def game_loop(self):
        self.update()
        self.draw()
        self.root.after(self.speed, self.game_loop)

def main():
    root = tk.Tk()
    root.geometry('500x600')
    
    # Установка иконки окна (опционально)
    # root.iconbitmap('snake.ico')
    
    game = SnakeGame(root)
    root.mainloop()

if __name__ == '__main__':
    main()