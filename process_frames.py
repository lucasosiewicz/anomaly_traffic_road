import os
import subprocess
import glob
import argparse
import sys
import cv2
import pathlib # Dodajemy pathlib dla łatwiejszej manipulacji ścieżkami

# --- Konfiguracja Argumentów ---
def parse_args():
    parser = argparse.ArgumentParser(description='Przetwarza rekurencyjnie foldery klatek JPG za pomocą skryptu v2e i wyodrębnia klatki z wynikowych wideo DVS.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Główny folder wejściowy (np. "frames"), zawierający podfoldery train/val/test/.../images.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Główny folder wyjściowy, w którym zostanie odtworzona struktura wejściowa.')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Oryginalna liczba klatek na sekundę dla danych wejściowych (stosowana globalnie).')
    # Można dodać więcej argumentów argparse, aby kontrolować parametry V2E
    return parser.parse_args()

# --- Funkcja do uruchamiania V2E dla pojedynczego folderu ---
def run_v2e_for_folder(v2e_script_path, input_images_dir, output_base_dir, fps):
    """Uruchamia v2e i ekstrakcję klatek dla pojedynczego folderu 'images'."""
    print(f"\n=== Przetwarzanie folderu: {input_images_dir} ===")

    frame_files = sorted(glob.glob(os.path.join(input_images_dir, "*.jpg")))
    if not frame_files:
        print(f"Ostrzeżenie: Nie znaleziono plików .jpg w {input_images_dir}. Pomijanie.")
        return

    num_frames = len(frame_files)
    print(f"Znaleziono {num_frames} plików JPG.")

    # --- Konfiguracja Parametrów V2E (Poprawiona dla nagrań wideo) ---
    overwrite = True
    output_video_filename = "dvs-video.avi"
    skip_video_output = False # Nadal potrzebujemy wideo do ekstrakcji
    disable_slomo = True
    
    # Parametry czasowe poprawione dla nagrań wideo
    dvs_exposure_duration = 0.005  # Stała ekspozycja 5ms (typowa dla DVS)
    input_rate = fps  # Używamy rzeczywistego FPS
    avi_rate = fps    # Używamy rzeczywistego FPS
    
    # Parametry DVS dla lepszej jakości
    pos_thres = 0.2   # Próg dla pozytywnych zdarzeń
    neg_thres = 0.2   # Próg dla negatywnych zdarzeń
    sigma_thres = 0.03  # Próg szumu
    dvs_vid_full_scale = "1"

    # --- Budowanie Polecenia V2E --- 
    command = [sys.executable, v2e_script_path]
    command.extend(["--input", input_images_dir])
    command.extend(["--output_folder", output_base_dir])
    if overwrite:
        command.append("--overwrite")
    command.extend(["--dvs_vid", output_video_filename])
    command.append("--no_preview")
    if skip_video_output:
         command.append("--skip_video_output")
    else:
        command.extend(["--dvs_exposure", "duration", str(dvs_exposure_duration)])
        command.extend(["--dvs_vid_full_scale", dvs_vid_full_scale]) 
    
    # Ustawienia wejścia powiązane z liczbą klatek
    command.extend(["--input_frame_rate", str(input_rate)])
    # Ustawienie --avi_frame_rate (wymagało int, użyjemy input_rate)
    # Sprawdzimy, czy v2e nadal wymaga int dla avi_frame_rate
    try:
        avi_rate_int = int(avi_rate)
        command.extend(["--avi_frame_rate", str(avi_rate_int)])
    except ValueError:
        print(f"Ostrzeżenie: Nie można przekonwertować liczby klatek ({avi_rate}) na int dla --avi_frame_rate. Używanie domyślnej wartości 30.")
        command.extend(["--avi_frame_rate", "30"])

    # Ustawienia Slomo
    if disable_slomo:
        command.append("--disable_slomo")
    
    # Parametry DVS dla lepszej jakości i usunięcia szarych klatek
    command.extend(["--pos_thres", str(pos_thres)])
    command.extend(["--neg_thres", str(neg_thres)])
    command.extend(["--sigma_thres", str(sigma_thres)])
    
    # Dodatkowe parametry dla stabilności
    command.extend(["--auto_timestamp_resolution", "True"])
    command.extend(["--dvs_emulator_seed", "0"])  # Dla powtarzalności

    # --- Uruchomienie V2E --- 
    print(f"Uruchamianie polecenia V2E dla {input_images_dir}:\n    {' '.join(command)}")
    try:
        # Upewnij się, że folder wyjściowy istnieje PRZED uruchomieniem v2e
        os.makedirs(output_base_dir, exist_ok=True)
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print("--- v2e Output ---")
        # print(result.stdout) # Można odkomentować dla pełnego logu v2e
        print(f"--- v2e zakończył działanie pomyślnie dla {input_images_dir} ---")

        # --- Ekstrakcja klatek --- 
        dvs_video_path = os.path.join(output_base_dir, output_video_filename)
        if os.path.exists(dvs_video_path):
            print(f"--- Rozpoczynanie ekstrakcji klatek z {dvs_video_path} ---")
            # Zmieniamy folder docelowy na 'images' wewnątrz output_base_dir
            extracted_frames_dir = os.path.join(output_base_dir, "images") 
            os.makedirs(extracted_frames_dir, exist_ok=True)
            cap = cv2.VideoCapture(dvs_video_path)
            if not cap.isOpened():
                print(f"Błąd: Nie można otworzyć pliku wideo: {dvs_video_path}")
            else:
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_filename = os.path.join(extracted_frames_dir, f"{frame_count:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_count += 1
                    if frame_count % 100 == 0: print(f"Wyodrębniono {frame_count} klatek...", end='\r')
                cap.release()
                print(f"\nZakończono ekstrakcję. Zapisano {frame_count} klatek w folderze: {extracted_frames_dir}")

                # --- Usuwanie przetworzonego wideo i plików tekstowych ---
                print(f"Usuwanie przetworzonych plików z {output_base_dir}...")
                files_to_delete = [
                    dvs_video_path,
                    os.path.join(output_base_dir, "dvs-video-frame_times.txt"),
                    os.path.join(output_base_dir, "v2e-args.txt")
                ]
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        print(f" - Usunięto: {os.path.basename(file_path)}")
                    except FileNotFoundError:
                        print(f" - Ostrzeżenie: Nie znaleziono pliku do usunięcia: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f" - Błąd podczas usuwania {os.path.basename(file_path)}: {e}")

        else:
            print(f"Ostrzeżenie: Nie znaleziono pliku wideo {dvs_video_path}. Pomijanie ekstrakcji klatek i usuwania plików.")

    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas uruchamiania v2e dla {input_images_dir}:")
        print(f"Polecenie: {' '.join(e.cmd)}")
        print(f"Kod błędu: {e.returncode}")
        print("--- v2e stderr ---")
        print(e.stderr)
        # print("--- v2e stdout ---") # stdout może być już częściowo wydrukowany
        # print(e.stdout)
    except Exception as e:
        print(f"Niespodziewany błąd podczas przetwarzania {input_images_dir}: {e}")

# --- Główna funkcja ---
def main():
    args = parse_args()

    v2e_script_path = os.path.join("v2e", "v2e.py")
    main_input_dir = args.input_dir
    main_output_dir = args.output_dir
    global_fps = args.fps

    # Sprawdź, czy skrypt v2e istnieje
    if not os.path.isfile(v2e_script_path):
        print(f"Błąd: Nie znaleziono skryptu v2e.py w oczekiwanej lokalizacji: {v2e_script_path}")
        exit(1)

    # Sprawdź, czy główny folder wejściowy istnieje
    if not os.path.isdir(main_input_dir):
        print(f"Błąd: Główny folder wejściowy nie istnieje: {main_input_dir}")
        exit(1)

    # --- Znajdź wszystkie foldery 'images' rekurencyjnie ---
    # Używamy pathlib dla łatwiejszego składania ścieżek i wyszukiwania
    input_dir_path = pathlib.Path(main_input_dir)
    output_dir_path = pathlib.Path(main_output_dir)
    # Szukamy wszystkich folderów o nazwie 'images' wewnątrz input_dir_path
    image_folders = list(input_dir_path.rglob("images"))

    if not image_folders:
        print(f"Nie znaleziono żadnych podfolderów 'images' w {main_input_dir}")
        exit(0)

    print(f"Znaleziono {len(image_folders)} folderów 'images' do przetworzenia.")

    # --- Przetwarzanie każdego folderu 'images' ---
    for images_path in image_folders:
        # Upewnij się, że to rzeczywiście folder
        if not images_path.is_dir():
            continue

        # Konwertujemy ścieżkę pathlib na string dla reszty kodu
        input_images_dir_str = str(images_path)

        # Oblicz ścieżkę względną folderu nadrzędnego 'images'
        relative_parent_path = images_path.parent.relative_to(input_dir_path)

        # Skonstruuj pełną ścieżkę wyjściową dla wyników v2e (bez końcowego 'images')
        output_base_dir_path = output_dir_path / relative_parent_path
        output_base_dir_str = str(output_base_dir_path)

        # Uruchom przetwarzanie dla tego folderu
        try:
            run_v2e_for_folder(v2e_script_path, input_images_dir_str, output_base_dir_str, global_fps)
        except Exception as e:
            print(f"Krytyczny błąd podczas przetwarzania {input_images_dir_str}. Kontynuowanie z następnym folderem. Błąd: {e}")

    print(f"\nPrzetwarzanie wszystkich folderów zakończone. Wyniki znajdują się w {main_output_dir}")

# Uruchomienie głównej funkcji, jeśli skrypt jest wykonywany bezpośrednio
if __name__ == "__main__":
    main() 