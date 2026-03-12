import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|serve]")
        print("  train  - Fine-tune classifier on data/labeled_data.csv")
        print("  serve  - Start the web application (trains first if needed)")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        from src.train import train
        train()

    elif command == "serve":
        from src.classifier import CommentClassifier
        clf = CommentClassifier()
        if not clf.is_trained():
            print("No trained model found. Training first...")
            from src.train import train
            train()
        from app import app
        print("Starting web server at http://localhost:2273")
        app.run(debug=False, host="0.0.0.0", port=2273)

    else:
        print(f"Unknown command: {command!r}")
        print("Valid commands: train, serve")
        sys.exit(1)


if __name__ == "__main__":
    main()
