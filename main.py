#                                                       WARNING

#                               В некоторых комментариях присутствует нецензурная брань
#                       Проношу свои извенения и прошу удалить их при необходимости демонстрации кода
#                                               вышестоящим начальством

#                                                       WARNING


#Подключение библиотек
from turtle import st, fd
import psycopg2

from config import host, user, password, db_name
import datetime
import os
import glob
import tkinter.filedialog as fd

import cv2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import time
import tkinter as tk
from tkinter import *

from tkinter.messagebox import showerror
import tkinter.scrolledtext as st
import numpy as np



yolo_net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg') #загружает модель YOLOv4 из файлов .weights и .cfg с помощью функции readNet из библиотеки OpenCV.
# В результате создается объект yolo_net, представляющий нейронную сеть
output_layer_names = yolo_net.getUnconnectedOutLayersNames() #Здесь определяются имена всех выходных слоев в нейронной сети YOLOv4.
# Эта информация будет использоваться позже для получения выходных данных с этих слоев после передачи изображения через сеть.

connection = None #означает отсутствие связи или неопределенное состояние c БД.

# Создание графического интерфейса (надписи кнопачки, размерчики, расположение, шрифтики и тд.)
win = tk.Tk()
win.title('Cистема анализа траектории движения электросамоката')
win.geometry("500x500")
win.resizable(False, False)
tk.Label(win, text= 'Cистема анализа траектории движения электросамоката', font=("Times New Roman", 14)).place(x=20, y=50)
tk.Label(win,text = 'Введите ФИО').place(x=210, y=210)
Fio = tk.Entry(win, width=30)
Fio.place(x=160, y=230)
buttonload = tk.Button(win, text="Выбрать файл",width=20,height=2, command=lambda:Entry(name=Fio.get(), nev=Fio.get()))
buttonload.place(x=178,y=260)
tk.Label(win, text='Научный руководитель: Малинский С.В.', font=("Times New Roman", 10)).place(x=10, y=450)
tk.Label(win, text='Команда разработчиков: Угурчиева Д.М.', font=("Times New Roman", 10)).place(x=10, y=470)


# Создание графического интерфейса для панели администратора (тоже кнопочки шрифтики и т.д.)
def AdminPanel():
    global new_window, label2
    new_window = Toplevel(win)
    new_window.title('Панель Администратора')
    new_window.geometry("600x600")
    new_window.resizable(False, False)

    tk.Label(new_window, text='ID', font=("Times New Roman", 12)).place(x=70, y=35)
    slot_id = tk.Entry(new_window, width=10)
    slot_id.place(x=50, y=60)
    tk.Label(new_window, text='ФИО', font=("Times New Roman", 12)).place(x=212, y=35)
    slot_name = tk.Entry(new_window, width=10)
    slot_name.place(x=200, y=60)
    tk.Label(new_window, text='Дата', font=("Times New Roman", 12)).place(x=362, y=35)
    slot_date = tk.Entry(new_window, width=10)
    slot_date.place(x=350, y=60)
    tk.Label(new_window, text='Результат', font=("Times New Roman", 12)).place(x=497, y=35)
    slot_verdict = tk.Entry(new_window, width=10)
    slot_verdict.place(x=500, y=60)
    buttonDelete = tk.Button(new_window,
                                text="Просмотр",
                                width=10,
                                height=1, command=lambda: AdminWork(id=slot_id.get(),name=slot_name.get(), data=slot_date.get(), verdict=slot_verdict.get()))
    buttonDelete.place(x=200, y=550)
    buttonGet = tk.Button(new_window,
                               text="Удаление",
                               width=10,
                               height=1, command=lambda: AdminDelete(id=slot_id.get(),name=slot_name.get(), data=slot_date.get(), verdict=slot_verdict.get()))
    buttonGet.place(x=320, y=550)
    #buttonDelete и buttonGet это кнопочки для при нажатии на которые вызываются функции с
    # получением указаных данных Например:(command=lambda: AdminDelete(id=slot_id.get(),name=slot_name.get(), data=slot_date.get(), verdict=slot_verdict.get()))




#Переменная хранящая соединение с базой данных
def AdminWork(id=None, name=None, data=None, verdict=None):
    global connection


    # удаляет все файлы из директории (для очистки старых файлов), нахуй не надо если честно
    files = glob.glob('D:/OneDrive/Desktop/AdminPhotos')
    for f in files:
        os.remove(f)



    # Создается блок с возможностью прокрутки (если в бд много данных)
    list1 = st.ScrolledText(new_window, wrap=tk.WORD, width=95, height=25, font=("Times New Roman", 10))
    list1.place(x=10, y=130)


    # устанавливается соединение с базой данных PostgreSQL (host, user, password, и db_name в файле config.py)
    # также включается автоматическое подтверждение (autocommit) транзакций.
    try:
        connection = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=db_name
        )
        connection.autocommit = True



        # Этот блок кода используется для выполнения SQL-запроса к базе данных PostgreSQL для получения версии сервера Тоже не особо надо
        with connection.cursor() as cursor:
            cursor.execute("SELECT version();")
            print(f"Server version: {cursor.fetchone()}")



            # Здесь создаются переменные db_get, conditions и values. db_get содержит SQL-запрос
            # для выборки всех данных из таблицы project, а conditions и values будут использоваться
            # для формирования условий выборки
            db_get = "SELECT * FROM project"
            conditions = []
            values = []



            #В этом блоке кода проверяются параметры id, name, data и verdict, и если они не
            #равны None, то для каждого из них создается соответствующее условие выборки и
            #добавляется значение к списку values. Если значение data может быть преобразовано в
            #дату, то это значение преобразуется в объект datetime и используется в запросе.
            if id:
                conditions.append("id = %s")
                values.append(id)
            if name:
                conditions.append("name = %s")
                values.append(name)
            if data:
                try:
                    date_obj = datetime.datetime.strptime(data, '%Y-%m-%d')
                    conditions.append("data = %s")
                    values.append(date_obj)
                except ValueError as ex:
                    print(f"Error converting data to timestamp: {ex}")
            if verdict:
                conditions.append("verdict = %s")
                values.append(verdict)



            #Здесь формируется итоговый SQL-запрос, добавляя к основному запросу условия выборки, если они есть.
            # После этого запрос выполняется с использованием метода execute, и результаты сохраняются в переменной gets.
            if conditions:
                db_get += " WHERE " + " AND ".join(conditions)

            cursor.execute(db_get, tuple(values))

            gets = cursor.fetchall()




            #Здесь результаты запроса обрабатываются и добавляются в текстовое поле list1.
            # Каждая строка таблицы отображает id, date, name и verdict из базы данных.
            # После того как все строки добавлены, текстовое поле блокируется для редактирования.
            for row in gets:
                id = row[0]
                name = row[1]
                date = row[2]
                verdict = row[3]

                list1.insert(tk.INSERT, "{:^5}".format(id) + "{:^25}".format(
                    date.strftime('%Y-%m-%d')) + "{:^30}".format(name))
                verdict = str(verdict)
                list1.insert(tk.INSERT, "{:^15}".format(verdict) + "\n")
            list1.configure(state='disabled')



    #Этот блок кода используется для обработки исключений, которые могут возникнуть при работе с базой данных PostgreSQL.
    # При возникновении ошибки соединение закрывается, чтобы избежать утечек ресурсов.
    except psycopg2.Error as ex:
        print("[INFO] Error while working with PostgreSQL", ex)
    finally:
        if connection:
            connection.close()
            print("[INFO] PostgreSQL connection closed")



#определяет функцию AdminDelete, используется для удаления записей из базы данных PostgreSQL
def AdminDelete(id=None, name=None, data=None, verdict=None):


    # устанавливается соединение с базой данных PostgreSQL (host, user, password, и db_name в файле config.py)
    # также включается автоматическое подтверждение (autocommit) транзакций
    try:
        connection = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=db_name
        )
        connection.autocommit = True


        #В этом блоке кода формируется SQL-запрос на удаление данных из таблицы project в базе данных.
        # Условия удаления зависят от переданных в функцию параметров. Если параметры заданы, они добавляются в условие запроса.
        # После формирования запроса, он выполняется с помощью метода execute.

        with connection.cursor() as cursor:
            db_get = "DELETE FROM project"
            conditions = []
            values = []

            if id:
                conditions.append("id = %s")
                values.append(id)
            if name:
                conditions.append("name = %s")
                values.append(name)
            if data:
                try:
                    date_obj = datetime.datetime.strptime(data, '%Y-%m-%d')
                    conditions.append("data = %s")
                    values.append(date_obj)
                except ValueError as ex:
                    print(f"Error converting data to timestamp: {ex}")
            if verdict:
                conditions.append("verdict = %s")
                values.append(verdict)

            if conditions:
                db_get += " WHERE " + " AND ".join(conditions)

            cursor.execute(db_get, tuple(values))


    #В этом блоке обрабатываются исключения, которые могут возникнуть при работе с базой данных PostgreSQL.
    # В случае ошибки соединение закрывается, чтобы избежать утечек ресурсов.
    except psycopg2.Error as ex:
        print("[INFO] Error while working with PostgreSQL", ex)
    finally:
        if connection:
            connection.close()
            print("[INFO] PostgreSQL connection closed")


#   Начинается самое ебучее. Ну погнали


#определение функции Choose_file с двумя параметрами: name и nev. (Забавно, что nev не используется, но если ее убрать но ничего работать не будет)
def Choose_file(name, nev):

    #Объявление глобальных переменных video_filename, verdict и connection,
    # которые будут использоваться внутри функции.
    global video_filename
    global verdict
    global connection

    #Отображение диалогового окна для выбора видеофайла с помощью fd.askopenfilename.
    # Выбранный путь сохраняется в переменную video_filename.
    video_filename = fd.askopenfilename(title="Открыть видео файл", initialdir="D:/path/to/videos")

    if video_filename: #Проверка, был ли выбран файл.

        date_now = datetime.datetime.now()
        date1 = date_now.strftime('%Y-%m-%d %H:%M:%S') #Получение текущей даты и времени и форматирование в строку

        output_layer_names = yolo_net.getUnconnectedOutLayersNames()
        trajectory = [] #   Инициализация списка trajectory для хранения траектории объекта на видео.
        cap = cv2.VideoCapture(video_filename) #    Открытие видеофайла для чтения с помощью OpenCV.
        frame_number = 0 #  Инициализация переменных для отслеживания номера кадра
        processing_time = 0 #   времени обработки
        last_frame = None # последнего кадра
        draw_line = False # флага рисования линии
        frame_skip = 15 # количества пропускаемых кадров
        current_frame = 0  # текущего кадра.


        #Цикл для обработки каждого кадра видео.
        # Кадр считывается с помощью cap.read(), проверяется успешность чтения (ret),
        # а затем проверяется, нужно ли пропустить этот кадр (frame_skip). Если кадр нужно пропустить,
        # происходит переход к следующему итерации цикла.
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            current_frame += 1

            if current_frame % frame_skip != 0:
                continue

            frame_number += 1


        #Получение размеров и каналов изображения кадра и измерение времени начала обработки кадра.
            height, width, channels = frame.shape
            start_time = time.time()


        #Преобразование кадра в блоб, это тип данных для хранения изображения в СУБД (Как итог оно нахуй не используется).
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            outs = yolo_net.forward(output_layer_names)



            #Обработка выходов модели YOLO.
            # Получение координат ограничивающих рамок (boxes),
            # уверенности (confidences) и идентификаторов классов (class_ids) для объектов, обнаруженных на кадре
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)




            #Применение метода Non-Maximum Suppression (NMS) для удаления дублирующихся рамок.
            # Затем происходит проход по отфильтрованным рамкам и добавление координат центра
            # объекта (если это объект с классом "0") в список trajectory для отслеживания траектории объекта
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(class_ids[i])
                    confidence = confidences[i]

                    if label == "0":
                        trajectory.append((center_x, center_y))




            #Проверка условия для рисования линии траектории объекта и рисование линии на кадре с помощью OpenCV,
            # если условие выполняется. Затем текущий кадр сохраняется в переменной last_frame.
            # Также вычисляется время обработки кадра.
            if len(trajectory) > 1:
                draw_line = True

            if draw_line:
                if len(trajectory) > 2:
                    pts = np.array(trajectory, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    frame = cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

            last_frame = frame

            end_time = time.time()
            processing_time += (end_time - start_time)

            verdict = ""


            #Проверка нажатия клавиши 'q'. Если клавиша 'q' нажата, цикл прерывается (Закрытие кадра с результатом).
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        #Закрытие видеофайла после завершения обработки всех кадров
        cap.release()


        #Преобразование списка координат траектории в массив NumPy и разделение координат на отдельные массивы x и y
        trajectory = np.array(trajectory)
        x = trajectory[:, 0]
        y = trajectory[:, 1]



        #Вот тут вообще нихуя не понятно, но по фатку если бы не она, то все бы было очень плохо

        #Создание объекта PolynomialFeatures степени 2 для преобразования признаков. З
        # атем координаты x преобразуются в полиномиальные признаки с помощью метода fit_transform.
        # Создается и обучается модель линейной регрессии (LinearRegression), используя преобразованные признаки x_poly и координаты y.
        # Предсказанные значения y вычисляются для преобразованных признаков.
        # Коэффициент детерминации (R-квадрат) рассчитывается для оценки качества модели.
        # В зависимости от значения R-квадрата, присваивается вердикт о плавном или резком движении объекта.
        # БЛА-БЛА-БЛА хуйня, но звучит умно

        poly_features = PolynomialFeatures(degree=2)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))

        poly_regression_model = LinearRegression()
        poly_regression_model.fit(x_poly, y)

        predicted_y_poly = poly_regression_model.predict(x_poly)

        r_squared_poly = poly_regression_model.score(x_poly, y)

        if r_squared_poly > 0.5:
            verdict = "Smooth motion"
        else:
            verdict = "Rough motion"



        #Если в траектории есть более двух точек, то рисуется кривая, отображающая траекторию объекта на последнем кадре.
        # Затем кадр помечается вердиктом ("Smooth motion" или "Rough motion") с помощью OpenCV функции putText.
        if len(trajectory) > 2:
            pts = np.array(trajectory, np.int32)
            pts = pts.reshape((-1, 1, 2))
            last_frame = cv2.polylines(last_frame, [pts], isClosed=False, color=(220, 20, 60), thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(last_frame, verdict, (20, 50), font, 2, (220, 20, 60), 2)


        # Вывод финального кадра
        #Проверка, что последний кадр существует и имеет ненулевые размеры.
        # Если условие выполняется, то последний кадр масштабируется и отображается на экране с помощью OpenCV функции imshow.
        # Также кадр сохраняется в файл с именем, содержащим вердикт, номер кадра и время обработки.
        if last_frame is not None and last_frame.shape[0] > 0 and last_frame.shape[1] > 0:
            small_last_frame = cv2.resize(last_frame, (680, 1000))
            cv2.imshow("Final Frame", small_last_frame)
            cv2.imwrite(f"{verdict}_{frame_number}_{processing_time}.jpeg", small_last_frame)



        #Устанавливается соединение с базой данных PostgreSQL с использованием переданных параметров.
        # Затем выполняется вставка новой записи в таблицу project с полями data, name и verdict,
        # где data содержит текущую дату и время, name - переданное имя, а verdict - вычисленный вердикт о движении объекта.
        try:
            connection = psycopg2.connect(
                host=host,
                user=user,
                password=password,
                database=db_name
            )
            connection.autocommit = True

            with connection.cursor() as cursor:
                xyz = """INSERT INTO project(data, name, verdict) VALUES (%s, %s, %s);"""
                cursor.execute(xyz, (date1, name, verdict))
        except Exception as _ex:
            print("[INFO] Ошибка при подключении к PostgreSQL:", _ex)
            print("Подключение к PostgreSQL не удалось.")

        finally:
            if connection:
                connection.close()
                print("PostgreSQL connection closed")


    #Если файл видео не был выбран, вызывается функция Error() для обработки ошибки.

    else:
        Error()



#отображает диалоговое окно с сообщением об ошибке, указывая на то, что файл не был выбран
def Error():
    showerror(title="Ошибка", message="Вы не выбрали файл")

#отображает диалоговое окно с сообщением об ошибке, указывая на то, что имя не было введено
def No_Name_Error():
    showerror(title="Ошибка", message="Вы не ввели имя")

# отображает диалоговое окно с сообщением об ошибке, указывая на то, что данные для удаления не были введены
def No_Data_Error():
    showerror(title="Ошибка", message="Вы не ввели данные для удаления")




    # вызывает окно в зависимости от введеного имени (если ничего не ввели вывод функции No_Name_Error),
    # (Если введено имя admin, открывает панель администратора, функция AdminPanel())
    # (Если введено что-то другое вызывает функцию для выбора файла Choose_file(name, nev))
def Entry(name, nev):
    if name == '':
        No_Name_Error()
    elif name == "admin":
        AdminPanel()
    else:
        Choose_file(name, nev)

win.mainloop()  # Закрывает окно приложения


#                                                   На этом все! :-) По дополнительным вопросам обращаться в тг - @blufuf