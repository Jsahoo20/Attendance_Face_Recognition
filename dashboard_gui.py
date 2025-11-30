import tkinter as tk
from tkinter import messagebox, Toplevel
import subprocess
import sys
import os
import pickle

# =========================
# 🎓 PROJECT CONFIGURATION
# =========================
PROJECT_TITLE = "5G LAB - SMART ATTENDANCE SYSTEM"
COLLEGE_NAME = "SVNIT, SURAT"
TEAM_MEMBERS = ["Jyotishankar Sahoo", "Abhisar Kumar", "Shivansh Tyagi"]

# --- GUIDES ---
GUIDE_NAME = "Prof. Sandeep Mishra"
CO_GUIDE_NAME = "Dr. Rahavendra Pal"  
DATABASE_FILE = "DeepFaceEncodings.pkl"

# =========================
# 🛠️ HELPER FUNCTIONS
# =========================
def run_script(script_name):
    """
    Runs external python scripts (registration.py, main.py)
    without freezing the dashboard.
    """
    python_exec = sys.executable
    script_path = os.path.join(os.getcwd(), script_name)

    if not os.path.exists(script_path):
        messagebox.showerror("Error", f"❌ {script_name} not found in directory.")
        return

    try:
        # Popen starts the script as a separate process
        subprocess.Popen([python_exec, script_path])
    except Exception as e:
        messagebox.showerror("Error", f"⚠️ Failed to start {script_name}.\n{e}")

def open_attendance_folder():
    """Opens the folder where CSVs are saved."""
    folder = "attendance_records"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Windows
    if os.name == 'nt':
        os.startfile(folder)
    # Linux/Mac (for Edge Server)
    else:
        subprocess.call(['xdg-open', folder])

def show_credits():
    """Displays the Team and Guide names in a popup."""
    credits_text = (
        f"🎓 {PROJECT_TITLE}\n"
        f"🏛️ {COLLEGE_NAME}\n\n"
        f"--- DEVELOPERS ---\n" + "\n".join([f"• {m}" for m in TEAM_MEMBERS]) + "\n\n"
        f"--- GUIDANCE ---\n"
        f"👨‍🏫 Guide: {GUIDE_NAME}\n"
        f"👨‍🏫 Co-Guide: {CO_GUIDE_NAME}"
    )
    messagebox.showinfo("Project Credits", credits_text)

# =========================
# 🗑️ DEREGISTER LOGIC (INTERNAL)
# =========================
def open_deregister_popup():
    """
    Opens a popup window inside the dashboard to remove students.
    """
    # Create a Popup Window (Toplevel)
    popup = Toplevel(root)
    popup.title("Remove Student")
    popup.geometry("350x250")
    popup.config(bg="#f8f9fa")
    
    # Label
    tk.Label(popup, text="Enter Roll Number to Delete:", 
             font=("Segoe UI", 10), bg="#f8f9fa").pack(pady=(20, 5))
    
    # Input Field
    entry_id = tk.Entry(popup, font=("Segoe UI", 12), justify="center")
    entry_id.pack(pady=5)
    
    # Delete Function
    def perform_delete():
        target_id = entry_id.get().strip().upper() # Force Uppercase
        
        if not target_id:
            messagebox.showwarning("Input Error", "Please enter a Roll Number.", parent=popup)
            return

        if not os.path.exists(DATABASE_FILE):
            messagebox.showerror("Error", "Database file not found.", parent=popup)
            return

        # Load Database
        try:
            with open(DATABASE_FILE, 'rb') as f:
                encodeListKnown, studentIds, studentNames = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Corrupt Database: {e}", parent=popup)
            return

        # Find Indices
        indices_to_delete = [i for i, x in enumerate(studentIds) if x == target_id]

        if not indices_to_delete:
            messagebox.showinfo("Not Found", f"No student found with Roll No: {target_id}", parent=popup)
            return

        # Confirm Deletion
        student_name = studentNames[indices_to_delete[0]]
        confirm = messagebox.askyesno(
            "Confirm Delete", 
            f"Found {len(indices_to_delete)} photo(s) for:\n\n"
            f"👤 Name: {student_name}\n"
            f"🆔 Roll No: {target_id}\n\n"
            "Are you sure you want to delete this student?",
            parent=popup
        )
        
        if confirm:
            # Delete in reverse order to keep indices valid
            for index in sorted(indices_to_delete, reverse=True):
                del encodeListKnown[index]
                del studentIds[index]
                del studentNames[index]

            # Save back to file
            with open(DATABASE_FILE, 'wb') as f:
                pickle.dump((encodeListKnown, studentIds, studentNames), f)
            
            messagebox.showinfo("Success", f"Deleted {target_id} ({student_name}) successfully.", parent=popup)
            entry_id.delete(0, tk.END) # Clear box

    # Delete Button
    tk.Button(popup, text="🗑️ Delete Student", bg="#DC3545", fg="white",
              font=("Segoe UI", 11, "bold"), command=perform_delete, 
              width=20).pack(pady=20)
    
    # Close Button
    tk.Button(popup, text="Cancel", command=popup.destroy, bg="#6c757d", fg="white").pack(pady=5)

# =========================
# 🖥️ MAIN DASHBOARD GUI
# =========================
root = tk.Tk()
root.title("🎓 5G Smart Lab Attendance System")
root.geometry("500x650") # Increased height slightly for new button
root.config(bg="#101820") # Dark Theme
root.resizable(False, False)

# --- Header Section ---
tk.Label(root, text=COLLEGE_NAME, font=("Helvetica", 12, "italic"), 
         bg="#101820", fg="#FEE715").pack(pady=(25, 0))

tk.Label(root, text=PROJECT_TITLE, font=("Helvetica", 16, "bold"), 
         bg="#101820", fg="white").pack(pady=(5, 25))

# --- Button Handlers ---
def start_register():
    run_script("registration.py")

def start_attendance():
    if not os.path.exists(DATABASE_FILE):
        messagebox.showwarning("Warning", "No Database found! Please register students first.")
    else:
        run_script("main.py")

# --- Button Styling ---
# A nice professional style for buttons
btn_style = {"font": ("Segoe UI", 11, "bold"), "width": 35, "height": 2, "bd": 0, "cursor": "hand2"}

# 1. Register Button
tk.Button(root, text="➕  Register New Student", bg="#007BFF", fg="white",
          command=start_register, **btn_style).pack(pady=10)

# 2. Deregister Button
tk.Button(root, text="🗑️  Deregister / Remove Student", bg="#FFC107", fg="#101820",
          command=open_deregister_popup, **btn_style).pack(pady=10)

# 3. Attendance Button
tk.Button(root, text="🎥  START 5G ATTENDANCE", bg="#28A745", fg="white",
          command=start_attendance, **btn_style).pack(pady=10)

# 4. Records Button
tk.Button(root, text="📂  Open Attendance Sheets", bg="#17a2b8", fg="white",
          command=open_attendance_folder, **btn_style).pack(pady=10)

# 5. Credits Button (NEW - To show Guides)
tk.Button(root, text="ℹ️  Project Credits", bg="#6C757D", fg="white",
          command=show_credits, **btn_style).pack(pady=10)

# 6. Exit Button
tk.Button(root, text="❌  Exit System", bg="#DC3545", fg="white",
          command=root.destroy, font=("Segoe UI", 10, "bold"), width=20).pack(pady=20)

# Footer
footer_text = "Powered by YuNet & FaceNet | 5G MEC Enabled Project"
tk.Label(root, text=footer_text, font=("Arial", 8), bg="#101820", fg="#666").pack(side="bottom", pady=10)

# Run the App
root.mainloop()