from pathlib import Path
from PIL import Image

from solution.pipeline import predict_from_image, get_model_profile

print(get_model_profile())

roots = [
    Path('data/private_test/images_sample'),
    Path('data/private_test/images'),
]
imgs = []
for r in roots:
    if r.exists():
        imgs += sorted(r.glob('*.jpg')) + sorted(r.glob('*.png'))

if not imgs:
    raise SystemExit('No sample images found under data/private_test/images_sample or data/private_test/images')

p = imgs[0]
print('Image:', p)
pred = predict_from_image(Image.open(p).convert('RGB'))
print('OCR:', pred.get('ocr_text'))
print('Brand:', pred.get('brand_name'))
print('Product:', pred.get('product_name'))
print('Timing:', pred.get('timing_ms'))
print('Audit:', pred.get('audit'))
