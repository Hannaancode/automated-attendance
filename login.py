import requests
import subprocess
import tkinter as tk
from tkinter import messagebox

def login():
    username = username_entry.get()
    password = password_entry.get()

    data = {
        "Username": username,
        "Password": password
    }
    #{"Username":"naziya.pathan@cloudxperte.com","Password":"Redbytes@123"}
    url = 'http://18.188.42.141:8080/api/authentication/authenticate'
    response = requests.post(url, json=data)

    if response.status_code == 200:
        res = response.json()
        if res["Success"]:
           token = res["Data"]["Token"]
           with open("token.txt", "w") as file:
                file.write(token)
           command = ["python", "gui.py"]
           subprocess.run(command)
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")
    else:
        messagebox.showerror("Error", "Failed to connect to the server.")

root = tk.Tk()
root.title("Login")

# Username Label and Entry
username_label = tk.Label(root, text="Username:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

# Password Label and Entry
password_label = tk.Label(root, text="Password:")
password_label.pack()
password_entry = tk.Entry(root, show="*")
password_entry.pack()

# Login Button
login_button = tk.Button(root, text="Login", command=login)
login_button.pack()

# Run the Tkinter event loop
root.mainloop()
