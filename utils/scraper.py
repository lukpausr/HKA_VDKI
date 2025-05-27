
import os
import hashlib
from PIL import Image
import imagehash
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
# from google_images_search import GoogleImagesSearch  # Uncomment if using this part

# --- 1. Download images with iCrawler (Bing) ---
#crawler = BingImageCrawler(storage={'root_dir': 'Siebenschläfer'})
#crawler.crawl(keyword='Siebenschläfer', max_num=500)


# --- 2. Download images with iCrawler (Google) ---
#crawler = GoogleImageCrawler(storage={'root_dir': 'dwarf_rabbit_3'})
#crawler.crawl(keyword='dwarf rabbit', max_num=500)


# --- 3. Add a suffix to all filenames in a folder ---
def add_suffix(folder, suffix="_9"):
    files = os.listdir(folder)
    files = [f for f in files if os.path.isfile(os.path.join(folder, f))]

    for name in files:
        old_path = os.path.join(folder, name)
        name_no_ext, ext = os.path.splitext(name)
        new_name = f"{name_no_ext}{suffix}{ext}"
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"{name} → {new_name}")

# Example usage:
# add_suffix("coelho anão")


# --- 4. Remove exact duplicate images (based on SHA-256 hash) ---
def get_hash(file_path):
    "Returns the SHA-256 hash of a file (for content comparison)."
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def remove_duplicates(folder):
    "Removes exact duplicate image files in the folder."
    seen = {}
    duplicates = 0

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            file_hash = get_hash(path)
            if file_hash in seen:
                print(f"Duplicate found → deleting: {name}")
                os.remove(path)
                duplicates += 1
            else:
                seen[file_hash] = name

    print(f"\n{duplicates} duplicate(s) removed.")

# Example usage:
# remove_duplicates("Bilder_hase")


# --- 5. Remove duplicate images with tholerance ---
def remove_visual_duplicates(folder, threshold=1):
    """
    Detects and automatically removes visually similar images using perceptual hash (pHash).
    Keeps the first encountered image and deletes any subsequent duplicates.
    
    Args:
        folder (str): Path to the folder containing images.
        threshold (int): Maximum hash difference to consider images as duplicates.
    """
    hashes = {}
    duplicates = 0

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    h = imagehash.phash(img)
            except Exception as e:
                print(f"Error reading {name}: {e}")
                continue

            for existing_hash, existing_name in hashes.items():
                if h - existing_hash <= threshold:
                    print(f"Auto-deleting duplicate: {name} (duplicate of {existing_name})")
                    try:
                        os.remove(path)
                        duplicates += 1
                    except Exception as e:
                        print(f"Error deleting {name}: {e}")
                    break  # stop after first match
            else:
                hashes[h] = name  # only if no confirmed duplicate

    print(f"\n {duplicates} duplicate(s) deleted.")

#remove_visual_duplicates("Waschbär", threshold=8)


# --- 6. Remove visually similar images using perceptual hashing ---
def remove_visual_duplicates_interactive(folder, threshold=1):
    """
    Detects visually similar images using perceptual hash (pHash).
    Prompts user to confirm deletion of suspected duplicates.
    """
    hashes = {}
    duplicates = 0

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    h = imagehash.phash(img)
            except Exception as e:
                print(f"Error reading {name}: {e}")
                continue

            for existing_hash, existing_name in hashes.items():
                if h - existing_hash <= threshold:
                    print(f"\n Possible duplicate found:")
                    print(f"Original: {existing_name}")
                    print(f"Duplicate: {name}")
                    print(f"Hash difference: {h - existing_hash}")

                    # Display both images
                    try:
                        print("Displaying images...")
                        with Image.open(os.path.join(folder, existing_name)) as img1:
                            img1.show(title="Original")
                        with Image.open(path) as img2:
                            img2.show(title="Duplicate?")
                    except Exception as e:
                        print(f"Error displaying images: {e}")

                    choice = input("Delete duplicate? (y/n): ").strip().lower()
                    if choice == 'y':
                        os.remove(path)
                        print(f" Deleted: {name}")
                        duplicates += 1
                    else:
                        print(" Kept")

                    break  # stop after first match

            else:
                hashes[h] = name  # only if no confirmed duplicate

    print(f"\n {duplicates} duplicate(s) deleted.")

# Example usage:
#remove_visual_duplicates_interactive("Waschbär", threshold=18)


# --- 7. Rename all image files in a folder with ordered names ---
def rename_images(folder):
    # Accepted image file extensions
    extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    # List and filter image files
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
    files.sort()  # Alphabetical order

    # Rename
    for i, name in enumerate(files, start=1):
        ext = os.path.splitext(name)[1]
        new_name = f"Ratte_{i:04d}_nok{ext}"
        old_path = os.path.join(folder, name)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"{name} → {new_name}")

    print("All files have been renamed.")

# Example usage:
rename_images("Ratte")


