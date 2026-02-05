from datetime import datetime


def main() -> None:
    print("✅ Electronics Hole Qualification - Project initialized")
    print(f"🕒 {datetime.now().isoformat(timespec='seconds')}")
    print("")
    print("Next steps:")
    print("1) Add model loaders (YOLO + SAM) in src/models/")
    print("2) Add inference service in src/services/")
    print("3) Build minimal UI in src/ui/")
    print("4) Add active learning + retraining pipeline")
    print("")


if __name__ == "__main__":
    main()
