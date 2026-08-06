import customtkinter as ctk
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class AppGUI(ctk.CTk):
    def __init__(self, camera_mgr, face_tracker, eye_tracker, logger):
        super().__init__()
        
        self.camera_mgr = camera_mgr
        self.face_tracker = face_tracker
        self.eye_tracker = eye_tracker
        self.logger = logger
        
        # Configure window
        self.title("Face & Eye Tracking AI")
        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.start_time = time.time()
        
        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar for stats
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(20, weight=1)
        
        logo_label = ctk.CTkLabel(self.sidebar_frame, text="Tracking Dashboard", font=ctk.CTkFont(size=20, weight="bold"))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Stat Labels Dictionary
        self.stat_labels = {}
        row_idx = 1
        stat_names = [
            "Right Face Count", "Left Face Count", "Up Count", "Down Count",
            "Left Blink Count", "Right Blink Count", "Both Blink Count",
            "Eye Left Count", "Eye Right Count", "Eye Up Count", "Eye Down Count",
            "Current Face Direction", "Current Eye Direction", "Current FPS", "Session Time"
        ]
        
        for name in stat_names:
            lbl_title = ctk.CTkLabel(self.sidebar_frame, text=f"{name}:", font=ctk.CTkFont(weight="bold"))
            lbl_title.grid(row=row_idx, column=0, padx=20, pady=(5, 0), sticky="w")
            
            lbl_val = ctk.CTkLabel(self.sidebar_frame, text="0" if "Count" in name else "-", text_color="cyan")
            lbl_val.grid(row=row_idx+1, column=0, padx=20, pady=(0, 5), sticky="w")
            
            self.stat_labels[name] = lbl_val
            row_idx += 2
            
        # Main Video Frame
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill="both")
        
        # Live Graph Frame (Bottom Right)
        self.graph_frame = ctk.CTkFrame(self, height=250)
        self.graph_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.fig, self.ax = plt.subplots(figsize=(8, 2), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")
        
        # History Data list for graph
        self.graph_data_face = []
        self.graph_data_eye = []
        
        # Bottom controls and table frame
        self.bottom_frame = ctk.CTkFrame(self, height=200)
        self.bottom_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        
        # Buttons
        self.btn_frame = ctk.CTkFrame(self.bottom_frame)
        self.btn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.btn_start = ctk.CTkButton(self.btn_frame, text="Start Counting", command=self.start_counting, fg_color="green")
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)
        
        self.btn_stop = ctk.CTkButton(self.btn_frame, text="Stop Counting", command=self.stop_counting, fg_color="red", state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkButton(self.btn_frame, text="Export CSV", command=self.export_csv).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.btn_frame, text="Export Excel", command=self.export_excel).grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkButton(self.btn_frame, text="Clear Data", command=self.clear_data).grid(row=2, column=0, padx=5, pady=5)
        ctk.CTkButton(self.btn_frame, text="Reset Counts", command=self.reset_counts).grid(row=2, column=1, padx=5, pady=5)
        
        ctk.CTkButton(self.btn_frame, text="Screenshot", command=self.take_screenshot).grid(row=3, column=0, padx=5, pady=5)
        
        self.btn_record = ctk.CTkButton(self.btn_frame, text="Start Recording", command=self.toggle_recording)
        self.btn_record.grid(row=3, column=1, padx=5, pady=5)
        
        # Data Table (Treeview)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", borderwidth=0)
        style.map('Treeview', background=[('selected', '#1f538d')])
        
        columns = ("Timestamp", "Event", "Count")
        self.tree = ttk.Treeview(self.bottom_frame, columns=columns, show="headings", height=8)
        self.tree.heading("Timestamp", text="Timestamp")
        self.tree.heading("Event", text="Event")
        self.tree.heading("Count", text="Count")
        self.tree.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Update loop setup
        self.last_frame_time = time.time()
        self.update_interval = 1000 // 30  # Target ~30 FPS
        self.update_gui()
        
    def update_gui(self):
        """Main update loop called by Tkinter."""
        if not getattr(self, 'is_running', True):
            return
            
        ret, frame = self.camera_mgr.read_frame()
        
        if ret:
            # Process Frame
            frame, face_results, face_dir = self.face_tracker.process_frame(frame)
            eye_dir = self.eye_tracker.process_eyes(frame, face_results, face_dir)
            
            # Update FPS
            current_time = time.time()
            fps = int(1.0 / (current_time - self.last_frame_time))
            self.last_frame_time = current_time
            self.stat_labels["Current FPS"].configure(text=str(fps))
            
            # Update Session Time
            session_time = int(current_time - self.start_time)
            mins, secs = divmod(session_time, 60)
            self.stat_labels["Session Time"].configure(text=f"{mins:02d}:{secs:02d}")
            
            # Update Labels
            self.stat_labels["Current Face Direction"].configure(text=face_dir)
            self.stat_labels["Current Eye Direction"].configure(text=eye_dir)
            
            self.stat_labels["Right Face Count"].configure(text=str(self.face_tracker.counts["Right"]))
            self.stat_labels["Left Face Count"].configure(text=str(self.face_tracker.counts["Left"]))
            self.stat_labels["Up Count"].configure(text=str(self.face_tracker.counts["Up"]))
            self.stat_labels["Down Count"].configure(text=str(self.face_tracker.counts["Down"]))
            
            self.stat_labels["Left Blink Count"].configure(text=str(self.eye_tracker.counts["Left Blink"]))
            self.stat_labels["Right Blink Count"].configure(text=str(self.eye_tracker.counts["Right Blink"]))
            self.stat_labels["Both Blink Count"].configure(text=str(self.eye_tracker.counts["Both Blink"]))
            
            self.stat_labels["Eye Left Count"].configure(text=str(self.eye_tracker.counts["Eye Left"]))
            self.stat_labels["Eye Right Count"].configure(text=str(self.eye_tracker.counts["Eye Right"]))
            self.stat_labels["Eye Up Count"].configure(text=str(self.eye_tracker.counts["Eye Up"]))
            self.stat_labels["Eye Down Count"].configure(text=str(self.eye_tracker.counts["Eye Down"]))
            
            # Draw frame to Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Update graph and table periodically
            if session_time % 2 == 0 and getattr(self, 'last_update_time', -1) != session_time:
                self.update_table()
                self.update_graph()
                self.last_update_time = session_time
                
        self._update_id = self.after(self.update_interval, self.update_gui)
        
    def update_table(self):
        """Update the Treeview table with latest logs."""
        for i in self.tree.get_children():
            self.tree.delete(i)
        logs = self.logger.get_logs()
        # Show last 100 entries reversed (newest first)
        for row in reversed(logs[-100:]):
            self.tree.insert("", "end", values=row)
            
    def update_graph(self):
        """Update the live matplotlib graph."""
        total_face = sum(self.face_tracker.counts.values())
        total_eye = sum(self.eye_tracker.counts.values())
        
        self.graph_data_face.append(total_face)
        self.graph_data_eye.append(total_eye)
        
        if len(self.graph_data_face) > 50:
            self.graph_data_face.pop(0)
            self.graph_data_eye.pop(0)
            
        self.ax.clear()
        self.ax.plot(self.graph_data_face, color='cyan', label='Face Movements')
        self.ax.plot(self.graph_data_eye, color='orange', label='Eye Movements')
        self.ax.legend(loc="upper left")
        self.ax.set_title("Live Movement Activity", color="white")
        self.canvas.draw()
        
    def export_csv(self):
        filename = self.logger.export_csv()
        print(f"Exported to {filename}")
        
    def export_excel(self):
        filename = self.logger.export_excel()
        print(f"Exported to {filename}")
        
    def clear_data(self):
        self.logger.clear_data()
        self.update_table()
        
    def reset_counts(self):
        self.face_tracker.reset_counts()
        self.eye_tracker.reset_counts()
        
    def take_screenshot(self):
        ret, frame = self.camera_mgr.read_frame()
        if ret:
            self.camera_mgr.take_screenshot(frame)
            
    def toggle_recording(self):
        is_rec, _ = self.camera_mgr.toggle_recording()
        if is_rec:
            self.btn_record.configure(text="Stop Recording", fg_color="red")
        else:
            self.btn_record.configure(text="Start Recording", fg_color=["#3a7ebf", "#1f538d"])
            
    def start_counting(self):
        self.face_tracker.is_counting = True
        self.eye_tracker.is_counting = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
    def stop_counting(self):
        self.face_tracker.is_counting = False
        self.eye_tracker.is_counting = False
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
            
    def on_closing(self):
        """Clean up resources on close."""
        self.is_running = False
        if hasattr(self, '_update_id'):
            self.after_cancel(self._update_id)
        self.camera_mgr.release()
        self.quit()
        self.destroy()
