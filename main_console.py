"""
Optional console runner to smoke-test Phase 1 modules without Flask/Tkinter.
This replaces the old monolithic main.py with clean function calls.
"""

from __future__ import annotations
import sys

from backend.user_manager import add_user_record, delete_user_record, list_users
from backend.photo_capture import capture_user_images
from backend.embedding_manager import generate_and_save_embeddings_for_user
from backend.face_recognition import verify_face_live


def add_user_flow():
    name = input("Enter new user's name: ").strip()
    try:
        uid = add_user_record(name)
        print(f"Created user {name} (id={uid}).")
    except ValueError as e:
        print(f"Error: {e}")
        return

    raw_dir, cropped_dir, n = capture_user_images(name)
    if n == 0:
        print("No images captured; aborting.")
        return

    generate_and_save_embeddings_for_user(name, cropped_dir)


def delete_user_flow():
    key = input("Enter user NAME or ID to delete: ").strip()
    ok = delete_user_record(key)
    print("Deleted." if ok else "User not found.")


def list_users_flow():
    users = list_users()
    if not users:
        print("(no users)")
        return
    for uid, info in users.items():
        print(f"{uid}  |  {info['name']}")


def access_flow():
    verify_face_live()


def main():
    while True:
        print("\n🔹 Eagle Phase 1 Console 🔹")
        print("1) Add user (capture + embed)")
        print("2) Delete user")
        print("3) List users")
        print("4) Access (live verify)")
        print("0) Exit")
        choice = input("> ").strip()
        if choice == "1":
            add_user_flow()
        elif choice == "2":
            delete_user_flow()
        elif choice == "3":
            list_users_flow()
        elif choice == "4":
            access_flow()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
