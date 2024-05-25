import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import os
import time
from tkinter import PhotoImage
import re
import subprocess


class ProgramRunnerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Automated attendance")

        image_path = "./img/download(1).png"

        try:
            self.background_image = PhotoImage(file=image_path)
            label = tk.Label(master, image=self.background_image)
            label.pack()
        except tk.TclError as e:
            print(f"Error loading image: {e}")
            
        # Add a frame for better layout
        main_frame = tk.Frame(master, bg='dark grey', bd=5)
        main_frame.place(relx=0.5, rely=0.5, relwidth=0.75, relheight=0.7, anchor='center')
        main_frame = tk.Frame(master)
        main_frame.place(relx=0.5, rely=0.5, relwidth=0.75, relheight=0.7, anchor='center')

        # Load the background image
        try:
            background_image = tk.PhotoImage(file="./img/Online-Attendance-Management-System-768x722.png")
            background_label = tk.Label(main_frame, image=background_image)
            background_label.place(relwidth=1, relheight=1)
            background_label.image = background_image  # Keep a reference to prevent garbage collection
        except tk.TclError as e:
            print(f"Error loading image: {e}")

        button_style = {'bg': 'red', 'fg': 'black', 'font': ('Arial', 12, 'bold')}

        # Button for running Main Program
        self.main_program_button = tk.Button(master, text="Run attendance mapping", command=self.run_main_program,**button_style)
        self.main_program_button.pack(pady=10)

        # Button for stopping Main Program
        self.stop_main_program_button = tk.Button(master, text="Stop attendance mapping", command=self.stop_main_program,**button_style)
        self.stop_main_program_button.pack(pady=10)

        # Button for running Program 2
        self.program2_button = tk.Button(master, text="Register new face", command=self.run_program2,**button_style)
        self.program2_button.pack(pady=10)

        # Variable to signal stopping Main Program
        self.stop_main_program_var = tk.BooleanVar(value=False)

        # Entry for providing the user name to delete
        self.delete_user_name_entry = tk.Entry(master)
        self.delete_user_name_entry.pack(pady=10)

        # Button for deleting user's database
        self.delete_user_button = tk.Button(master, text="Delete User Name", command=self.delete_user_database,**button_style)
        self.delete_user_button.pack(pady=10)

        

        # Entry for providing the ID to delete
        self.delete_id_entry = tk.Entry(master)
        self.delete_id_entry.pack(pady=10)

        # Button for deleting ID from name_to_id_mapping in main.py
        self.delete_id_button = tk.Button(master, text="Delete User ID", command=self.delete_id_from_mapping,**button_style)
        self.delete_id_button.pack(pady=10)

        self.program3_button = tk.Button(master, text="Attendance Report", command=self.run_program3,**button_style)
        self.program3_button.pack(pady=10)

        self.program4_button = tk.Button(master, text="Emotion", command=self.run_program4,**button_style)
        self.program4_button.pack(pady=10)

        self.program5_button = tk.Button(master, text="Video player", command=self.run_program5,**button_style)
        self.program5_button.pack(pady=10)
  
    
    def run_main_program(self):
        self.stop_main_program_var.set(False)  # Reset the stop variable
        threading.Thread(target=self.run_main_program_thread).start()
        

    def run_main_program_thread(self):
        try:
            self.main_process = subprocess.Popen(["python", "main.py"])
            self.main_process.communicate()
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Failed to run the Main Program.")
        finally:
            self.stop_main_program_var.set(True)  # Ensure stop variable is set to True after process completes

    def stop_main_program(self):
        if hasattr(self, 'main_process') and self.main_process.poll() is None:
            self.main_process.terminate()

    def run_program2(self):
        # Replace 'program2.py' with the actual command to run Program 2
        subprocess.run(["python", "register.py"])
    def run_program3(self):
        # Replace 'program2.py' with the actual command to run Program 2
        subprocess.run(["python", "attendance.py"])
    def run_program4(self):
        # Replace 'program2.py' with the actual command to run Program 2
        subprocess.run(["python", "entry.py"])

    def run_program5(self):
        # Replace 'program2.py' with the actual command to run Program 2
        subprocess.run(["python", "video player.py"])
             

    def delete_user_database(self):
        user_name = self.delete_user_name_entry.get()

        if not user_name:
            tk.messagebox.showerror("Error", "Please enter a user name.")
            return

        # Path to the databases directory
        databases_path = "./databases"

        # Full path to the user's database file
        user_database_path = os.path.join(databases_path, f"{user_name}.npz")

        # Check if the file exists before attempting to delete
        if os.path.exists(user_database_path):
            os.remove(user_database_path)
            tk.messagebox.showinfo("Success", f"Database for user '{user_name}' deleted successfully.")
            threading.Thread(target=self.delete_id_from_mapping, args=(user_name,)).start()  # Pass user_name to delete_id_from_mapping
        else:
            tk.messagebox.showerror("Error", f"Database for user '{user_name}' not found.")

    def delete_id_from_mapping(self):
    # Rest of the code remains unchanged
     id_to_delete = self.delete_id_entry.get()

     if not id_to_delete:
         tk.messagebox.showerror("Error", "Please enter an ID to delete.")
         return


     # Path to the main.py file
     main_py_path = "./main.py"

     # Read the content of main.py
     with open(main_py_path, "r") as f:
         main_py_content = f.read()

     # Check if the ID exists in the main.py file before attempting to delete
     if f": {id_to_delete}," in main_py_content:
         # Use regular expressions to match the entire line containing the ID
         pattern = rf"\".*\": {id_to_delete},\n"
         main_py_content = re.sub(pattern, "", main_py_content)

         # Write the modified content back to main.py
         with open(main_py_path, "w") as f:
             f.write(main_py_content)

         tk.messagebox.showinfo("Success", f"ID '{id_to_delete}' deleted from mapping in main.py successfully.")

         

     else:
         tk.messagebox.showerror("Error", f"ID '{id_to_delete}' not found in mapping in main.py.")

      

if __name__ == "__main__":
    root = tk.Tk()
    app = ProgramRunnerGUI(root)
    root.mainloop()
