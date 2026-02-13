from ocr_class import ImageOCRReader
from pathlib import Path


def simple_test():
    print("=== Simple Test for ImageOCRReader ===\n")

    # 自己本地的图片存储目录
    data_dir = Path("C:\\Users\\86157\\Desktop\\github_files\\jike_ai_engineer_training_homeworks_whw\\module3_homework_1\\data_files")
    image_files = list(data_dir.glob('*.png'))

    if not image_files:
        print("No images found in ocr_images")
        return

    print(f"Found {len(image_files)} images:")
    for img in image_files:
        print(f"  - {img.name} ,{img.absolute()}")

    print("\nInitializing ImageOCRReader...")
    reader = ImageOCRReader(lang='en', use_gpu=False)

    print("\nLoading images...")
    documents = reader.load_data(image_files)

    print(f"\nSuccessfully loaded {len(documents)} documents\n")

    for i, doc in enumerate(documents, 1):
        print(f"--- Document {i} ---")
        print(f"Image: {Path(doc.metadata['image_path']).name}")
        print(f"Text blocks: {doc.metadata['num_text_blocks']}")
        print(f"Avg confidence: {doc.metadata['avg_confidence']:.2f}")
        print(f"Text preview: {doc.text[:300]} ...")


if __name__ == "__main__":
    simple_test()
