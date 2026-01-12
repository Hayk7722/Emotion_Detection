import numpy as np
import cv2
import pickle

# Словарь эмоций
out = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

# Загружаем модель
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Открываем камеру
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        break

    # Подготовка кадра для модели
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    image = resized / 255.0
    image = image.reshape(1, 48, 48, 1)

    # Предсказание
    y_pred = model.predict(image)
    label = out[np.argmax(y_pred)]

    # Отображаем предсказанную эмоцию на изображении
    cv2.putText(
        frame,
        f"Emotion: {label}",
        (30, 50),                 # координаты текста
        cv2.FONT_HERSHEY_SIMPLEX, # шрифт
        1,                        # размер
        (0, 255, 0),              # цвет (зеленый)
        2,                        # толщина линии
        cv2.LINE_AA               # сглаживание
    )

    # Показываем кадр с текстом
    cv2.imshow("Emotion Detection", frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()