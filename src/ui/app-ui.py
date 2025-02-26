import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from models.model_training import train_or_load_model

class HeartDiseasePredictionUI:
    def __init__(self, root, model):
        """
        Initialize the UI for heart disease prediction.
        
        Parameters:
        -----------
        root : tk.Tk
            Root window for the UI
        model : sklearn estimator
            Trained model for prediction
        """
        self.root = root
        self.root.title("پیش‌بینی بیماری قلبی")
        self.root.geometry("800x900")
        self.root.configure(bg='#f0f0f0')
        
        self.model = model

        # Create scrollable frame
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Title
        title_label = ttk.Label(self.scrollable_frame, 
                              text="سیستم پیش‌بینی بیماری قلبی", 
                              font=('Tahoma', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10)

        # Create input fields
        self.entries = {}
        self.create_input_fields()

        # Prediction button
        predict_btn = ttk.Button(self.scrollable_frame, 
                               text="پیش‌بینی",
                               command=self.predict)
        predict_btn.grid(row=len(self.field_info) + 1, column=0, columnspan=2, pady=20)

        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def create_input_fields(self):
        """Create input fields for all features."""
        self.field_info = [
            ("جنسیت (1=مرد، 0=زن)", "male", 0, 1),
            ("سن", "age", 20, 90),
            ("تحصیلات (1-4)", "education", 1, 4),
            ("سیگار کشیدن (1=بله، 0=خیر)", "currentSmoker", 0, 1),
            ("سیگارهای مصرفی روزانه", "cigsPerDay", 0, 100),
            ("مصرف داروهای فشار خون (1=بله، 0=خیر)", "BPMeds", 0, 1),
            ("سابقه سکته مغزی (1=بله، 0=خیر)", "prevalentStroke", 0, 1),
            ("سابقه فشار خون (1=بله، 0=خیر)", "prevalentHyp", 0, 1),
            ("دیابت (1=بله، 0=خیر)", "diabetes", 0, 1),
            ("کلسترول تام", "totChol", 100, 600),
            ("فشار خون سیستولیک", "sysBP", 90, 200),
            ("فشار خون دیاستولیک", "diaBP", 60, 140),
            ("شاخص توده بدنی", "BMI", 15, 50),
            ("ضربان قلب", "heartRate", 40, 150),
            ("گلوکز", "glucose", 40, 400)
        ]

        for i, (label, key, min_val, max_val) in enumerate(self.field_info):
            ttk.Label(self.scrollable_frame, text=f"{label}:", font=('Tahoma', 10)).grid(
                row=i+1, column=0, pady=5, padx=10, sticky='e')
            
            entry = ttk.Entry(self.scrollable_frame, width=30)
            entry.grid(row=i+1, column=1, pady=5, padx=10, sticky='w')
            
            # Add default values as placeholder
            default_value = (min_val + max_val) / 2
            entry.insert(0, f"{default_value}")
            
            self.entries[key] = entry

    def predict(self):
        """Perform prediction based on input values."""
        try:
            # Collect input data
            data = {}
            for _, key, _, _ in self.field_info:
                value = self.entries[key].get().strip()
                if value:
                    data[key] = float(value)
                else:
                    raise ValueError(f"لطفاً مقداری برای {key} وارد کنید")
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([data])
            
            # Make prediction
            prediction = self.model.predict(input_df)
            probability = self.model.predict_proba(input_df)[0][1]
            
            # Show result
            result = "بله" if prediction[0] == 1 else "خیر"
            message = f"احتمال ابتلا به بیماری قلبی در ۱۰ سال آینده: {result}\n"
            message += f"احتمال: {probability:.2%}"
            
            messagebox.showinfo("نتیجه پیش‌بینی", message)
            
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در پیش‌بینی: {str(e)}")
