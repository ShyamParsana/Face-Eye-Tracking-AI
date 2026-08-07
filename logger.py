import pandas as pd
import os
from datetime import datetime

class EventLogger:
    def __init__(self, session_id: str = None, *args, **kwargs):
        """Initialize the event logger using a Pandas DataFrame."""
        self.session_id = session_id
        self.columns = ["Timestamp", "Event", "Count"]
        self.df = pd.DataFrame(columns=self.columns)
        self.assets_dir = "assets"
        
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir, exist_ok=True)
            
    def log_event(self, event_name, count):
        """
        Log a new event with its current count.
        
        Args:
            event_name (str): The name of the event (e.g., 'Face Right')
            count (int): The updated count for this event
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_row = {"Timestamp": str(event_name if event_name else ""), "Event": str(event_name), "Count": int(count)}
        # Set proper timestamp in column
        new_row["Timestamp"] = timestamp
        # Use concat instead of append since append is deprecated in newer pandas
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
    def get_logs(self):
        """
        Return the current logs as a list of lists with native Python primitives.
        
        Returns:
            list: Log entries e.g. [['12:00:01', 'Face Right', 1], ...]
        """
        raw_list = self.df.values.tolist()
        clean_logs = []
        for row in raw_list:
            if len(row) >= 3:
                clean_logs.append([str(row[0]), str(row[1]), int(row[2]) if str(row[2]).isdigit() else row[2]])
            else:
                clean_logs.append([str(item) for item in row])
        return clean_logs
        
    def clear_data(self):
        """Clear all logged events."""
        self.df = pd.DataFrame(columns=self.columns)
        
    def format_export_data(self):
        """Format the chronological logs into the user's requested multi-table layout."""
        times = []
        face_events = []
        eye_events = []
        blink_events = []
        
        for _, row in self.df.iterrows():
            t = row["Timestamp"]
            ev = row["Event"]
            
            times.append(t)
            if ev.startswith("Face"):
                face_events.append(ev.replace("Face ", ""))
                eye_events.append("-")
                blink_events.append("-")
            elif ev.startswith("Eye"):
                face_events.append("-")
                eye_events.append(ev.replace("Eye ", ""))
                blink_events.append("-")
            elif "Blink" in ev:
                face_events.append("-")
                eye_events.append("-")
                blink_events.append(ev.replace(" Blink", ""))
            else:
                face_events.append("-")
                eye_events.append("-")
                blink_events.append("-")

        # Get max counts for summary
        totals = {}
        for ev in self.df["Event"].unique():
            totals[ev] = self.df[self.df["Event"] == ev]["Count"].max()
            
        face_dirs = ["Up", "Down", "Left", "Right"]
        eye_dirs = ["Up", "Down", "Left", "Right"]
        blink_types = ["Right", "Left", "Both"]
        
        face_total_names = [d for d in face_dirs]
        face_total_counts = [totals.get(f"Face {d}", 0) for d in face_dirs]
        
        eye_total_names = [d for d in eye_dirs]
        eye_total_counts = [totals.get(f"Eye {d}", 0) for d in eye_dirs]
        
        blink_total_names = [b for b in blink_types]
        blink_total_counts = [totals.get(f"{b} Blink", 0) for b in blink_types]
        
        # Calculate required rows
        max_rows = max(len(times), len(face_dirs), len(eye_dirs), len(blink_types))
        
        def pad(lst, length, val=""):
            return lst + [val] * (length - len(lst))
            
        times = pad(times, max_rows)
        face_events = pad(face_events, max_rows)
        eye_events = pad(eye_events, max_rows)
        blink_events = pad(blink_events, max_rows)
        
        face_total_names = pad(face_total_names, max_rows)
        face_total_counts = pad(face_total_counts, max_rows)
        
        eye_total_names = pad(eye_total_names, max_rows)
        eye_total_counts = pad(eye_total_counts, max_rows)
        
        blink_total_names = pad(blink_total_names, max_rows)
        blink_total_counts = pad(blink_total_counts, max_rows)
        
        empty_col = [""] * max_rows
        
        # Construct dataframe with trailing spaces for duplicate column names
        export_df = pd.DataFrame({
            "Time": times,
            "Face Event": face_events,
            "Eye Event": eye_events,
            "Blink": blink_events,
            " ": empty_col,
            "Face Direction": face_total_names,
            "Total Count": face_total_counts,
            "  ": empty_col,
            "Eye Direction": eye_total_names,
            "Total Count ": eye_total_counts,
            "   ": empty_col,
            "Blinks": blink_total_names,
            "Total Count  ": blink_total_counts
        })
        
        return export_df

    def export_csv(self):
        """
        Export logged data to a CSV file in the specified format.
        
        Returns:
            str: Path to the exported file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.assets_dir, f"export_{timestamp}.csv")
        export_df = self.format_export_data()
        export_df.to_csv(filename, index=False)
        return filename
        
    def export_excel(self):
        """
        Export logged data to an Excel file in the specified format.
        
        Returns:
            str: Path to the exported file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.assets_dir, f"export_{timestamp}.xlsx")
        export_df = self.format_export_data()
        export_df.to_excel(filename, index=False)
        return filename
