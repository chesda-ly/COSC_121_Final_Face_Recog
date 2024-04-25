import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import time
from PIL import Image, ImageTk
import sqlite3

from read import insertOrUpdate
from train import get_image_with_id
from detect import getProfile

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def start_face_recognition():
    cam = cv2.VideoCapture(0)
    Id = id_entry.get()
    Name = name_entry.get().capitalize()
    age = age_entry.get()

    if not Id or not Name or not age:
        messagebox.showerror("Error", "Please fill all fields")
        return
    
    if not Id.isdigit() or not age.isdigit() or not Name.isalpha():
        messagebox.showerror("Error", "ID and Age must be in numbers. Name must be in characters.")
        return
    
    if int(Id) < 2015000 or int(Id) > 2025000:
        messagebox.showerror("Error", "Invalid ID number.")
        return

    # Add inputs to database
    insertOrUpdate(Id, Name, age)

    id_entry.delete(0, tk.END)
    name_entry.delete(0, tk.END)
    age_entry.delete(0, tk.END)

    sampleNum = 0
    no_face_detected_time = None

    # Start capturing images
    while True:
        _, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5);
        if len(faces) == 0:

            if no_face_detected_time is None:
                no_face_detected_time = time.time()

            elif time.time() - no_face_detected_time >= 5:
                print("No face detected for 5 seconds. Exiting...")
                break

        else:
            no_face_detected_time = None

        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1;
            cv2.imwrite(f"dataset/{Name}." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])        
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(100)
            
        cv2.imshow("Registering Face", img)
        cv2.waitKey(1);
        if sampleNum == 20:
            break;
    
    cv2.destroyAllWindows()
        
    # Train the model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'dataset'
    ids, faces = get_image_with_id(path)
    cv2.destroyAllWindows()
    recognizer.train(faces, ids)
    recognizer.save('recognizer/trainingdata.yml')

    # Show detection/recognition window
    recognizer.read('recognizer/trainingdata.yml')
    while(True):
        _, img = cam.read();
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, _ = recognizer.predict(gray[y:y+h, x:x+w])
            profile = getProfile(id)

            if profile != None:
                cv2.putText(img, "ID: " + str(profile[0]), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)
                cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)
                cv2.putText(img, "Age: " + str(profile[2]), (x, y+h+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(10) == ord('q'):
            break;
        
    cam.release()
    cv2.destroyAllWindows()

# Function to view database
def view_database():
    db_window = tk.Toplevel(root)
    db_window.title("Students Database")

    tree = ttk.Treeview(db_window)
    tree["columns"] = ("ID", "Name", "Age")
    tree.column("#0", width=0, stretch=tk.NO)
    tree.column("ID", anchor=tk.W, width=100)
    tree.column("Name", anchor=tk.W, width=100)
    tree.column("Age", anchor=tk.W, width=100)

    tree.heading("#0", text="", anchor=tk.W)
    tree.heading("ID", text="ID", anchor=tk.W)
    tree.heading("Name", text="Name", anchor=tk.W)
    tree.heading("Age", text="Age", anchor=tk.W)

    conn = sqlite3.connect('sqlite.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    rows = cursor.fetchall()
    for row in rows:
        tree.insert("", tk.END, values=row)

    tree.pack()
    conn.close()

# Function to begin face recognition
def begin_face_recognition():
    cam = cv2.VideoCapture(0)
    try:
        recognizer.read('recognizer/trainingdata.yml')
        while(True):
            _, img = cam.read();
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id, _ = recognizer.predict(gray[y:y+h, x:x+w])
                profile = getProfile(id)
                if profile != None:
                    cv2.putText(img, "ID: " + str(profile[0]), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)
                    cv2.putText(img, "Age: " + str(profile[2]), (x, y+h+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 2)

            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(10) == ord('q'):
                break;
    finally:
        cam.release()
        cv2.destroyAllWindows()

# Function to focus next widget
def focus_next_widget(event):
    event.widget.tk_focusNext().focus()
    return "break"

root = tk.Tk()

root.geometry('1280x720')
root.title('Facial Recognition')

background = Image.open('logos/bglogo.jpg')
background = background.resize((1280, 720))
background = ImageTk.PhotoImage(background)

background_label = tk.Label(root, image=background)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

image = Image.open('logos/AUPP Vertical Logo.png')
image = image.resize((200, 217))
image = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=image, bg='white', fg='black')
image_label.place(relx=0.5, rely=0, anchor= "n")

label = tk.Label(root, text='Welcome to AUPP Facial Recognition System', font=('Times New Roman', 24), fg='#2F3A6A', bg="white")
label.place(relx=0.5, rely=220/720, anchor= "n")

id_label = tk.Label(root, text="Enter your ID:", font=('Times New Roman', 16), bg="white", fg='black')
id_label.place(relx=0.5, rely=(220/720)+35/720, anchor= "n")

id_entry = tk.Entry(root, fg='black', bg='white')
id_entry.place(relx=0.5, rely=(220/720)+70/720, anchor= "n")
id_entry.bind("<Return>", focus_next_widget)


name_label = tk.Label(root, text="Enter your first name:", font=('Times New Roman', 16), bg="white", fg='black')
name_label.place(relx=0.5, rely=(220/720)+105/720, anchor= "n")

name_entry = tk.Entry(root, fg='black', bg='white')
name_entry.place(relx=0.5, rely=(220/720)+140/720, anchor= "n")
name_entry.bind("<Return>", focus_next_widget)

age_label = tk.Label(root, text="Enter your age:", font=('Times New Roman', 16), bg="white", fg='black')
age_label.place(relx=0.5, rely=(220/720)+175/720, anchor= "n")

self_label = tk.Label(root, text="Press \"ENTER\" to start. \"q\" to quit.", font=('Times New Roman', 16),fg="black", bg="white")
self_label.place(relx=0.5, rely=(220/720)+290/720, anchor= "n")

age_entry = tk.Entry(root, fg='black', bg='white')
age_entry.place(relx=0.5, rely=(220/720)+210/720, anchor= "n")
age_entry.bind("<Return>", focus_next_widget)

start_button = tk.Button(root, text="Start Facial Registration", font=('Times New Roman', 16), command=start_face_recognition, width = 20)
start_button.place(relx=0.5, rely=(220/720)+250/720, anchor= "n")
start_button.bind("<Return>", lambda event: start_face_recognition())

view_db_button = tk.Button(root, text="View Database", font=('Times New Roman', 16), command=view_database, width = 20)
view_db_button.place(relx=0.48, rely=(220/720)+350/720, anchor= "e")

start_detection_button = tk.Button(root, text="View Face Detection", font=('Times New Roman', 16), comman = begin_face_recognition, width = 20)
start_detection_button.place(relx=0.53, rely=(220/720)+350/720, anchor= "w")

root.mainloop()