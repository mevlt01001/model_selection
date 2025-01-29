import time
import os
from src.clear import clear
from src.log import log
import tkinter as tk
from tkinter import filedialog

def get_data_dir():
    try:
        # Tkinter uygulamasını gizli çalıştırmak için bir kök pencere oluştur ve gizle
        root = tk.Tk()
        root.withdraw()

        # Kullanıcıdan bir klasör seçmesini iste
        data_dir = filedialog.askdirectory(title="Veri dizinini seçin")

        if not data_dir:
            error_message = "[DIRECTORY INPUT ERROR]: Kullanıcı bir dizin seçmedi. Tekrar sorulacak."
            log("FUNC_GET_DATA_DIR", error_message)
            print("Hatalı dizin girdiniz veya seçim yapmadınız. Lütfen tekrar deneyin.")
            return get_data_dir()

        if not os.path.exists(data_dir):
            error_message = f"[DIRECTORY INPUT ERROR]: Veri dizini '{data_dir}' bulunamadı. Kullanıcıya tekrar sorulacak."
            log("FUNC_GET_DATA_DIR", error_message)
            print("Seçilen dizin geçersiz. Lütfen tekrar deneyin.")
            return get_data_dir()

        script_dir = os.getcwd()  # Çalıştırılan script'in bulunduğu dizin
        data_dir = os.path.relpath(data_dir, script_dir)
        success_message = f"[SUCCESS]: Veri dizini '{data_dir}' başarıyla bulundu."
        log("FUNC_GET_DATA_DIR", success_message)
        print(success_message)
        return data_dir

    except PermissionError as pe:
        exception_message = f"[PERMISSION ERROR]: İzin hatası meydana geldi: {str(pe)}"
        log("FUNC_GET_DATA_DIR", exception_message)
        print("İzin hatası meydana geldi. Lütfen farklı bir dizin seçin.")
        return get_data_dir()

    except Exception as e:
        exception_message = f"[EXCEPTION]: Hata meydana geldi: {str(e)}"
        log("FUNC_GET_DATA_DIR", exception_message)
        print("Bir hata meydana geldi. Lütfen tekrar deneyin.")
        return get_data_dir()

    finally:
        log("FUNC_GET_DATA_DIR", "get_data_dir fonksiyonu tamamlandı.")