from ultralytics import YOLO

# Load trained model
model = YOLO("models/best (4).pt")

# Test image
results = model.predict(
    source="corroison.jpg",
    conf=0.25,
    show=True,
    save=True
)

print("Detection completed ✅")