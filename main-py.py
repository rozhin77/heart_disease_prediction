import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add the project directory to the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from src.data.data_processing import load_and_preprocess_data
from src.models.model_training import train_or_load_model
from src.ui.app import HeartDiseasePredictionUI

def main():
    """Main entry point for the application."""
    try:
        # Load and preprocess data
        try:
            X, y = load_and_preprocess_data()
            print("داده‌ها با موفقیت بارگذاری شدند")
        except Exception as e:
            print(f"خطا در بارگذاری داده‌ها: {str(e)}")
            print("ادامه عملیات با تلاش برای بارگذاری مدل ذخیره شده...")
            X, y = None, None
        
        # Train or load model
        try:
            model = train_or_load_model(X, y)
            print("مدل با موفقیت آماده شد")
        except Exception as e:
            print(f"خطا در آماده‌سازی مدل: {str(e)}")
            messagebox.showerror("خطا", f"خطا در آماده‌سازی مدل: {str(e)}")
            return
        
        # Launch UI
        root = tk.Tk()
        app = HeartDiseasePredictionUI(root, model)
        root.mainloop()
        
    except Exception as e:
        print(f"خطای غیرمنتظره: {str(e)}")
        messagebox.showerror("خطا", f"خطای غیرمنتظره: {str(e)}")

if __name__ == '__main__':
    main()
